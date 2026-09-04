"""
process_mp.py —— 多进程流水线版 TrafficAnalyzerMP（一次性阻塞 API）
=================================================================

为什么多进程：diagnose_parallel.py 实测本机"线程级多 GPU 并发"是灾难（单路 ~15ms
并发后变 ~180-330ms），而"进程级并发、各钉一张卡"正常（S∥L∥V ≈ 20ms）。
因此本模块把 S/L/V 三个模型各放到一个独立子进程，各自钉一张 GPU，
大块图像数据走共享内存，小结果走 mp.Queue，主进程做 P（CPU 后处理）。

拓扑：
    主进程(CPU): 读帧 → resize → 写共享内存 → 发 seq
                   ├→ [S 进程 cuda:4] 分割 → 写 masked/overlay/masks 到共享内存
                   ├→ [L 进程 cuda:6] 车道 track（ByteTrack 状态留进程内）
                   └→ [V 进程 cuda:5] 车辆 track+分组（ByteTrack 状态留进程内）
    主进程 P 线程: 按 seq 对齐 S/L/V 结果 → 密度/速度/拥堵 → List[ProcessFrameResult]

对外 API（一次性阻塞版）：
    an = TrafficAnalyzerMP(DetectionConfig(...), window=8)   # 启动 3 个子进程 + 加载模型（一次性成本）
    results = an.process_frames(iter_of_frames, reset_state=True)   # 阻塞，按帧序返回
    res = an.process_frame(img)                             # 单帧（等价 process_frames([img])）
    an.shutdown()

注意：
  - ByteTrack 状态常驻在 L/V 子进程内，跨 process_frames 调用保留（与 process.py 的
    TrafficAnalyzerLite 行为一致）；reset_state 只复位主进程 P 的速度/缩放跨帧状态。
  - 大图像数据（frame/masked/overlay/masks）用 ring 共享内存，避免每帧 pickle 数 MB。
"""
import os
import queue as _queue
import threading
import time
import multiprocessing as _mp
from multiprocessing.shared_memory import SharedMemory

import cv2
import numpy as np

from process import TrafficAnalyzer, ProcessFrameResult, DetectionConfig
import mp_workers as W


def _gpu_idx(device_str: str) -> int:
    return int(device_str.split(":")[1])


class _CPUPost(TrafficAnalyzer):
    """只复用 TrafficAnalyzer 的 CPU 后处理方法，不加载模型、不启动线程。"""

    def __init__(self, config=None):
        self.config = config or DetectionConfig()
        pts1 = np.float32([(450, 674), (750, 674), (500, 50), (700, 50)])
        pts2 = np.float32([(450, 674), (750, 674), (450, 0), (750, 0)])
        self.matrix = cv2.getPerspectiveTransform(pts1, pts2)
        self.input_width = self.INPUT_WIDTH
        self.input_height = self.INPUT_HEIGHT
        self._density_yf = None
        self._density_xf = None
        self._density_H = 0
        self._density_W = 0
        self.scaling_factor = 3.0
        self.past_scaling_factor = 3.0
        self.past_drone_speed = 10.0
        self.past_dis = {}
        self.previous_frame_lines = {}
        self.previous_frame_objects = {}
        self.count = 0
        self.output_dir = "./event"
        os.makedirs(self.output_dir, exist_ok=True)

    def shutdown(self):
        """CPU-only 版无资源需要释放。"""

    def _reset_cross_frame_state(self):
        self.past_dis.clear()
        self.previous_frame_objects = {}
        self.previous_frame_lines = {}
        self.past_drone_speed = 10.0
        self.past_scaling_factor = 3.0
        self.scaling_factor = 3.0


class _Slot:
    """流水线在途一帧的共享内存块。"""

    def __init__(self):
        self.frame = SharedMemory(create=True, size=W.FRAME_BYTES)
        self.masked = SharedMemory(create=True, size=W.FRAME_BYTES)
        self.overlay = SharedMemory(create=True, size=W.FRAME_BYTES)
        self.masks = SharedMemory(create=True, size=W.MASKS_BYTES)

    def close_unlink(self):
        for s in (self.frame, self.masked, self.overlay, self.masks):
            try:
                s.close()
            except Exception:
                pass
            try:
                s.unlink()
            except Exception:
                pass


class TrafficAnalyzerMP:
    """多进程流水线分析器（一次性阻塞 API）。"""

    DEFAULT_WINDOW = 8

    def __init__(self, config=None, window=None):
        self.window = window or self.DEFAULT_WINDOW
        if self.window < 1:
            raise ValueError(f"window 必须 >=1，实际 {self.window!r}")

        self._post = _CPUPost(config)
        self.input_width = self._post.input_width
        self.input_height = self._post.input_height

        # ---- 共享内存 ring（window 个在途帧）----
        self._slots = [_Slot() for _ in range(self.window)]

        # ---- 队列（fork 启动，父进程在 fork 前不碰 CUDA）----
        try:
            self._ctx = _mp.get_context("fork")
        except ValueError:
            self._ctx = _mp.get_context()   # 理论上 Linux 默认即 fork
        q = self._ctx.Queue
        cap = self.window * 2
        self.q_to_s = q(maxsize=cap)
        self.q_s_to_l = q(maxsize=cap)
        self.q_s_to_v = q(maxsize=cap)
        self.q_s_to_p = q(maxsize=cap)
        self.q_l_to_p = q(maxsize=cap)
        self.q_v_to_p = q(maxsize=cap)
        self._stop_ev = self._ctx.Event()

        def names(attr):
            return [getattr(s, attr).name for s in self._slots]

        # ---- 启动 3 个 worker 子进程（各钉一张卡，各加载一个模型）----
        base = os.path.abspath
        specs = (
            (W.seg_worker, (base(TrafficAnalyzer.SEGMENT_MODEL_PATH),
                            _gpu_idx(TrafficAnalyzer.SEGMENT_DEVICE))),
            (W.lane_worker, (base(TrafficAnalyzer.LANE_MODEL_PATH),
                             _gpu_idx(TrafficAnalyzer.LANE_DEVICE))),
            (W.veh_worker, (base(TrafficAnalyzer.VEHICLE_MODEL_PATH),
                            _gpu_idx(TrafficAnalyzer.VEHICLE_DEVICE))),
        )
        self._procs = []
        for target, (model_path, gpu) in specs:
            if target is W.seg_worker:
                args = (gpu, model_path, self.window,
                        names("frame"), names("masked"), names("overlay"), names("masks"),
                        self.q_to_s, self.q_s_to_l, self.q_s_to_v, self.q_s_to_p, self._stop_ev)
            elif target is W.lane_worker:
                args = (gpu, model_path, self.window, names("masked"),
                        self.q_s_to_l, self.q_l_to_p, self._stop_ev)
            else:
                args = (gpu, model_path, self.window, names("masked"), names("masks"),
                        self.q_s_to_v, self.q_v_to_p, self._stop_ev)
            p = self._ctx.Process(target=target, args=args, daemon=True)
            p.start()
            self._procs.append(p)

        # ---- 主进程 P 线程（CPU 后处理，按 seq 保序）----
        self._stop_thread = threading.Event()
        self._lock = threading.RLock()
        self._total = None
        self._next_seq = 0
        self._session_finished = threading.Event()
        self._results = {}
        self._slots_sem = None
        self._on_result = None
        self._p_thread = threading.Thread(target=self._post_loop, daemon=True)
        self._p_thread.start()

    # ------------------------------------------------------------------
    # 入口：一次性阻塞
    # ------------------------------------------------------------------
    def process_frames(self, frames, reset_state=True, on_result=None):
        """按帧序处理整段帧流，阻塞返回 List[ProcessFrameResult]。"""
        with self._lock:
            return self._process_frames_impl(frames, reset_state, on_result)

    def process_frame(self, frame):
        """单帧阻塞（等价 process_frames([frame], reset_state=True)）。"""
        return self.process_frames([frame], reset_state=True)[0]

    def _process_frames_impl(self, frames, reset_state, on_result):
        if reset_state:
            self._post._reset_cross_frame_state()

        self._results = {}
        self._next_seq = 0
        self._total = None
        self._on_result = on_result
        self._session_finished.clear()
        self._slots_sem = threading.BoundedSemaphore(self.window)

        seq = 0
        try:
            for frame in frames:
                if frame is None:
                    break
                self._slots_sem.acquire()          # 背压：在途帧 <= window
                self._write_frame(seq % self.window, frame)
                self.q_to_s.put(seq)
                seq += 1

            self._total = seq
            if not self._session_finished.wait(timeout=300.0):
                raise RuntimeError("多进程流水线处理超时（可能 worker 崩溃，请查看子进程报错）")
            return [self._results[i] for i in range(seq)]
        finally:
            self._slots_sem = None
            self._on_result = None
            self._total = None
            self._next_seq = 0

    def _write_frame(self, idx, frame):
        if frame.shape[0] != W.H or frame.shape[1] != W.W:
            frame = cv2.resize(frame, (W.W, W.H))
        arr = np.ndarray((W.H, W.W, W.C), dtype=np.uint8, buffer=self._slots[idx].frame.buf)
        arr[:] = frame

    # ------------------------------------------------------------------
    # P 线程：对齐 S/L/V 结果，按 seq 顺序做 CPU 后处理
    # ------------------------------------------------------------------
    def _post_loop(self):
        seg, lane, veh = {}, {}, {}
        while not self._stop_thread.is_set():
            got = False
            while True:
                try:
                    m = self.q_s_to_p.get_nowait()
                except _queue.Empty:
                    break
                seg[m[0]] = (m[1], m[2])
                got = True
            while True:
                try:
                    m = self.q_l_to_p.get_nowait()
                except _queue.Empty:
                    break
                lane[m[0]] = m[1]
                got = True
            while True:
                try:
                    m = self.q_v_to_p.get_nowait()
                except _queue.Empty:
                    break
                veh[m[0]] = m[1]
                got = True

            while (self._next_seq in seg and self._next_seq in lane
                   and self._next_seq in veh):
                seq = self._next_seq
                seg_ok, n = seg.pop(seq)
                lane_boxes = lane.pop(seq)
                veh_pkt = veh.pop(seq)
                try:
                    self._emit(seq, seg_ok, n, lane_boxes, veh_pkt)
                except Exception as e:
                    print(f"[MP P] 后处理异常 seq={seq}: {e!r}")
                    self._results[seq] = ProcessFrameResult(
                        frame=np.zeros((W.H, W.W, W.C), dtype=np.uint8), is_abnormal=True)
                if self._slots_sem is not None:
                    self._slots_sem.release()      # 释放共享内存槽位，允许后续帧复用
                self._next_seq += 1

            if self._total is not None and self._next_seq >= self._total:
                self._session_finished.set()

            if not got:
                time.sleep(0.0005)

    def _emit(self, seq, seg_ok, num_masks, lane_boxes, veh_pkt):
        slot = self._slots[seq % self.window]
        if not seg_ok:
            raw = np.ndarray((W.H, W.W, W.C), np.uint8, buffer=slot.frame.buf).copy()
            res = ProcessFrameResult(frame=raw, is_abnormal=True)
        else:
            overlay = np.ndarray((W.H, W.W, W.C), np.uint8, buffer=slot.overlay.buf).copy()
            masks_np = np.ndarray((num_masks, W.H, W.W), np.uint8,
                                  buffer=slot.masks.buf).astype(np.float32, copy=False)
            res = self._post._run_post_stage(lane_boxes, veh_pkt, overlay, masks_np)
        self._results[seq] = res
        if self._on_result is not None:
            self._on_result(seq, res)

    # ------------------------------------------------------------------
    def shutdown(self):
        self._stop_ev.set()
        self._stop_thread.set()
        for p in self._procs:
            p.join(timeout=5.0)
        for p in self._procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2.0)
        self._p_thread.join(timeout=2.0)
        for _q in (self.q_to_s, self.q_s_to_l, self.q_s_to_v,
                   self.q_s_to_p, self.q_l_to_p, self.q_v_to_p):
            try:
                _q.close()
                _q.join_thread()
            except Exception:
                pass
        for s in self._slots:
            s.close_unlink()
