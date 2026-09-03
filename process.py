"""
process.py — 交通视频帧间流水线引擎
====================================================

一帧的数据流（S → L∥V → P）
    一帧图
      → [S 分割   cuda:0] 产出 masked_frame / overlay_frame / road_masks(GPU) / masks_np(CPU)
      → 并行分发给两个 worker：
          ├→ [L 车道   cuda:2] 车道线 track → lane_boxes(numpy)
          └→ [V 车辆   cuda:1] 车辆 track + 按掩码分组 → (boxes, grouped)(numpy)
      → [P 后处理  CPU]  空检守卫 + 密度/速度/拥堵/异常事件（含跨帧状态）
      → ProcessFrameResult

帧间重叠（本文件的核心价值）
    阶段间用“单消费者 FIFO”队列串起来：Seg线程 → Lane/Veh worker → P线程。
    每个阶段同时只处理一帧，天然保序（ByteTrack/速度跨帧状态安全）且单飞（模型线程安全）；
    相邻帧可同时处于不同阶段 → S(N+1)、L/V(N)、P(N-1) 在 3 张 GPU + CPU 上重叠执行。

对外 API（server.py 只用这 4 个 + 2 个生命周期方法）
    start_pipeline()     开始一个流水线会话（线程都在 __init__ 常驻，这里只复位会话状态）
    submit_frame(frame)  提交一帧 → 返回 seq（有界队列满时阻塞 = 背压，不丢帧）
    get_result(timeout)  按帧号顺序取回 (seq, ProcessFrameResult)
    close_pipeline()     收尾（请先 get_result 取完全部结果再调用）
    reset_video_state()  一段视频/一次会话开始前调用（清跨帧状态）
    shutdown()           进程退出时关闭所有线程

# 节点地图（对照教学骨架 process_pseudocode.py）
#     伪代码节点            本文件实现位置                            说明
#     _node_segment        _segment_road()                            S：分割
#     _node_lane           _lane_worker() 线程内模型调用               L：车道线 → numpy
#     _node_vehicle        _vehicle_worker() 线程内调用 + _group_by_mask  V：车辆+分组 → numpy
#     _node_post_process   _run_post_stage()                          P：纯 CPU 后处理
#     process_pipeline     由 submit_frame()+get_result() 组合         伪代码=单帧阻塞，本文件=多帧流水线
"""
import time
import numpy as np
import os
import torch
import torch.nn.functional as F
import threading
import queue
import cv2
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO


@dataclass
class DetectionConfig:
    """客户端配置 - 仅暴露客户端需要的参数"""
    # 功能开关（客户端可选）
    enable_bike_person: bool = True    # 自行车/行人检测
    enable_stop_slow: bool = True      # 停止/慢速车辆检测


class AbnormalEventType(Enum):
    """异常事件类型"""
    BIKE = "bike"                      # 自行车检测
    PERSON = "person"                  # 行人检测
    STOPPED_VEHICLE = "stopped_vehicle"  # 停止的车辆
    SLOW_VEHICLE = "slow_vehicle"      # 慢速车辆


@dataclass
class AbnormalEvent:
    """异常事件"""
    event_type: AbnormalEventType
    detections: List = field(default_factory=list)   # 检测框列表


@dataclass
class ProcessFrameResult:
    """单帧处理结果封装类 - 优化版"""
    frame: np.ndarray
    traffic_congestion: bool = False                          # 交通拥堵标志（不再返回区域）

    abnormal_events: List[AbnormalEvent] = field(default_factory=list)  # 异常事件列表
    output_filename: Optional[str] = None  # 有异常事件时保存的唯一输出文件名
    metrics: Optional[Dict] = None                            # 本帧数值化指标（供前端展示；含密度/速度/拥堵区域/车辆数）

    is_abnormal: bool = False                                 # 异常返回标志：True 表示本帧未正常处理完成（代码或模型处理问题）



class TrafficAnalyzer:
    """多GPU多线程目标检测处理器"""

    # 部署参数（写死，客户端无需关心）
    SEGMENT_MODEL_PATH = "./runs/hz/viaduct/weights/best.pt"
    VEHICLE_MODEL_PATH = "./runs/hz/vehicle/weights/best.pt"
    LANE_MODEL_PATH = "./runs/hz/line/weights/best.pt"

    SEGMENT_DEVICE = "cuda:0"
    VEHICLE_DEVICE = "cuda:1"
    LANE_DEVICE = "cuda:2"

    INPUT_WIDTH = 960
    INPUT_HEIGHT = 540

    def __init__(self, config: Optional[DetectionConfig] = None):
        """
        初始化处理器 - 优化版

        Args:
            config: 客户端配置（仅功能开关与输出目录），默认使用默认配置
        """
        self.config = config or DetectionConfig()

        # 透视变换矩阵（固定标定，写死在类内）
        pts1 = np.float32([(450, 674), (750, 674), (500, 50), (700, 50)])
        pts2 = np.float32([(450, 674), (750, 674), (450, 0), (750, 0)])
        self.matrix = cv2.getPerspectiveTransform(pts1, pts2)

        self.input_width = self.INPUT_WIDTH
        self.input_height = self.INPUT_HEIGHT

        # 密度统计 CPU 坐标网格缓存（输入尺寸固定，只建一次）
        self._density_yf = None
        self._density_xf = None
        self._density_H = 0
        self._density_W = 0

        # 初始化设备
        self.segment_device = torch.device(self.SEGMENT_DEVICE)
        self.vehicle_device = torch.device(self.VEHICLE_DEVICE)
        self.lane_device = torch.device(self.LANE_DEVICE)

        # 加载模型（统一在 __init__ 一次性加载，worker 只取用，避免重复加载浪费显存）
        self.segment_model = YOLO(self.SEGMENT_MODEL_PATH).to(self.segment_device)
        self.vehicle_model = YOLO(self.VEHICLE_MODEL_PATH).to(self.vehicle_device)
        self.lane_model = YOLO(self.LANE_MODEL_PATH).to(self.lane_device)

        # ============ 全部队列统一在此创建（与线程同寿命，都由 __init__ 管理） ============
        self.seg_queue = queue.Queue(maxsize=4)      # S 输入：满则 submit_frame 阻塞（背压，不丢帧）
        self.lane_queue = queue.Queue(maxsize=2)     # → L worker
        self.vehicle_queue = queue.Queue(maxsize=2)  # → V worker
        self.out_queue = queue.Queue(maxsize=16)     # P 输出：按帧号顺序，供 get_result 取
        self.save_queue = queue.Queue(maxsize=30)    # → 写图线程（满了丢弃异常事件图，不阻塞）

        # 结果存储：按 seq 对齐各模型结果  L worker 和 V worker 是两个独立的线程、各跑各的 GPU，它们处理同一帧的快慢不一样
        self.lane_results = {}
        self.vehicle_results = {}

        self.results_cond = threading.Condition()   # 锁+门铃：保护共享字典，并让 P 线程 wait / 被 notify 唤醒

        # 会话状态（一次 start_pipeline ~ close_pipeline 为一个会话；线程常驻，会话只是“开关”）
        self._pipeline_on = False
        self._pipeline_seq = 0
        self._pipeline_total = None
        self.seg_payload = {}                 # seq -> 分割阶段产物（会话开始时清空）
        self._seg_done = False                # 本会话 Seg 线程是否已消费“结束哨兵”
        self._session_finished = threading.Event()  # 本会话全部帧处理完成信号

        # 控制标志
        self.stop_event = threading.Event()

        # 启动所有线程（同寿命：S/P/L/V/写图 全部在 __init__ 启动，之后不再创建线程）
        self._start_workers()

        self.scaling_factor = 3.0
        self.past_scaling_factor = 3.0
        self.past_drone_speed = 10.0
        self.past_dis: Dict[int, float] = {}
        self.previous_frame_lines: Dict = {}
        self.previous_frame_objects: Dict = {}

        # 输出计数（运行时状态，非配置）
        self.count = 0
        self.output_dir = "./event"
        os.makedirs(self.output_dir, exist_ok=True)



    def _start_workers(self):
        """启动全部线程（同一寿命，只在 __init__ 调用一次，之后不再创建线程）。"""
        self.segment_thread = threading.Thread(target=self._pipeline_segment_worker, name="Seg", daemon=True)
        self.post_thread = threading.Thread(target=self._pipeline_post_worker, name="P", daemon=True)
        self.lane_thread = threading.Thread(target=self._lane_worker, name="LaneWorker", daemon=True)
        self.vehicle_thread = threading.Thread(target=self._vehicle_worker, name="VehicleWorker", daemon=True)
        self.save_thread = threading.Thread(target=self._async_writer_worker, name="SaveWriter", daemon=True)
        for t in (self.segment_thread, self.post_thread, self.lane_thread,
                  self.vehicle_thread, self.save_thread):
            t.start()


    # ============================================================
    # L 节点 worker 线程（车道, GPU cuda:2）—— 对应伪代码 _node_lane
    # 消费 Seg 分发来的任务 (seq, masked_frame)，车道线 track 后把结果转成 numpy
    # 存入 lane_results[seq]，并 notify P 线程；任务帧号即流水线 seq。
    # ============================================================
    def _lane_worker(self):
        """车道线检测工作线程"""
        while not self.stop_event.is_set():
            try:
                # 非阻塞获取任务
                task = self.lane_queue.get(timeout=0.5)
                if task is None:
                    continue

                # 任务来自 Seg 线程： (seq, masked_frame)
                frame_id, masked_frame = task

                # 执行车道线检测
                results = self.lane_model.track(
                    masked_frame,
                    persist=True,
                    save=False,
                    verbose=False,
                    half=True,
                    device=self.lane_device,
                    conf=0.5
                )

                # CPU 化边界：只把下游所需的 boxes.data 转成 numpy 存入结果字典（不再传递 Results 对象）
                lane_boxes = None
                if results:
                    boxes = getattr(results[0], "boxes", None)
                    lane_boxes = boxes.data.cpu().numpy() if (boxes is not None and boxes.data.numel() > 0) else None

                # 按 frame_id 存入结果字典并通知等待方
                with self.results_cond:
                    self.lane_results[frame_id] = lane_boxes # 1. 把这一帧的结果放进盒子
                    self._prune_results(self.lane_results, frame_id) # 2. 顺手清理过期结果
                    self.results_cond.notify_all() # 3. 敲门：告诉 P 线程"有新货了"

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Lane worker error: {e}")

    # ============================================================
    # V 节点 worker 线程（车辆, GPU cuda:1）—— 对应伪代码 _node_vehicle
    # 消费 Seg 分发来的任务 (seq, masked_frame, road_masks)：
    # 车辆 track + _group_by_mask(按道路掩码分组，GPU 内完成) → numpy 数据包
    # (boxes, grouped) 存入 vehicle_results[seq]，并 notify P 线程。
    # ============================================================
    def _vehicle_worker(self):
        """车辆检测工作线程"""
        while not self.stop_event.is_set():
            try:
                # 非阻塞获取任务
                task = self.vehicle_queue.get(timeout=0.5)
                if task is None:
                    continue

                # 任务来自 Seg 线程： (seq, masked_frame, road_masks)
                frame_id, masked_frame, road_masks = task

                # 执行车辆检测和跟踪
                results = self.vehicle_model.track(
                    masked_frame,
                    persist=True,
                    save=False,
                    verbose=False,
                    half=True,
                    tracker="bytetrack.yaml",
                    device=self.vehicle_device
                )

                # 分组（_group_by_mask）在 worker 内于 GPU 完成，下游（P 线程）不再触碰 GPU 张量
                boxes_np, grouped_np = None, None
                if results:
                    boxes = getattr(results[0], "boxes", None)
                    boxes_np = boxes.data.cpu().numpy() if (boxes is not None and boxes.data.numel() > 0) else None
                    grouped_np = self._group_by_mask(road_masks, results).cpu().numpy()

                # CPU 化边界：存入 (原始框数组, 分组后数组) 数据包，不再传递 Results 对象
                with self.results_cond:
                    self.vehicle_results[frame_id] = (boxes_np, grouped_np)
                    self._prune_results(self.vehicle_results, frame_id)
                    self.results_cond.notify_all()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Vehicle worker error: {e}")

    def _async_writer_worker(self):
        """异步图片写入工作线程"""
        while not self.stop_event.is_set():
            try:
                item = self.save_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            path, img = item
            try:
                cv2.imwrite(path, img)
            except Exception as e:
                print(f"图片写入失败 {path}: {e}")
            finally:
                # 无论成败都标记完成，保证 shutdown() 的 join() 能正常返回
                self.save_queue.task_done()

    # ============================================================
    # P 节点入口（后处理, 纯 CPU）—— 对应伪代码 _node_post_process
    # 由流水线 P 线程按帧号顺序调用：空检守卫 → 自行车/行人 → 密度 → 速度 → 拥堵
    # → 异常事件 → ProcessFrameResult。所有跨帧状态都只在这里被串行更新。
    # ============================================================
    def _run_post_stage(self, raw_frame: np.ndarray, lane_boxes, veh_pkt,
                        overlay_frame: np.ndarray, masks_np: np.ndarray) -> ProcessFrameResult:
        """统一空检出守卫 + P 阶段（纯 CPU，含跨帧状态）。

        由流水线 P 线程调用；raw_frame 为已 resize 的原始帧（异常返回时作为画面）；
        lane_boxes / veh_pkt 为 worker 产出的 numpy 数据包。
        """
        enable_bike_person = self.config.enable_bike_person
        enable_stop_slow = self.config.enable_stop_slow

        # 统一空检出准则：分割模型用 r.masks is None 判空（见 _segment_road）；
        # 检测/追踪模型判空 = 结果数组行数为 0。
        # 任一模型未检出对象都视为本帧未正常处理完成（abnormal 返回，不进入下游后处理）：
        #   - 车道线模型没追踪到车道框 → 速度计算无车道依据；
        #   - 车辆模型没追踪到车辆框 → 与分割/车道结果矛盾，同样按异常帧处理。
        if lane_boxes is None or lane_boxes.shape[0] == 0:
            result = ProcessFrameResult(frame=raw_frame.copy())
            result.is_abnormal = True
            return result

        if veh_pkt is None:
            result = ProcessFrameResult(frame=raw_frame.copy())
            result.is_abnormal = True
            return result

        veh_boxes_np, vehicle_boxes = veh_pkt
        if veh_boxes_np is None or veh_boxes_np.shape[0] == 0:
            result = ProcessFrameResult(frame=raw_frame.copy())
            result.is_abnormal = True
            return result

        # 各类检测结果（先收集为局部变量，再统一映射为异常事件）
        bike_detections, person_detections = [], []
        stopped_vehicles, slow_vehicles = [], []
        density_info, speed_info = {}, {}

        # 自此以下全部为 CPU 数据处理：分组已在 vehicle worker 内完成，密度统计吃 CPU 掩码
        if enable_bike_person:
            bike_detections, person_detections = self._process_bike_person(overlay_frame, veh_boxes_np)

        overlay_frame, density_info = self._process_traffic_density(overlay_frame, masks_np, vehicle_boxes)

        # 速度计算（基本内容，始终执行；停止/慢速车辆由开关控制）
        overlay_frame, speed_info, stopped_vehicles, slow_vehicles = self._process_speed(
            overlay_frame, lane_boxes, vehicle_boxes, enable_stop_slow=enable_stop_slow
        )

        # 拥堵评估（基本内容，始终执行，只保留标志位）
        overlay_frame, _, congestion_regions = self._evaluate_congestion(
            overlay_frame, density_info, speed_info, vehicle_boxes
        )

        result = ProcessFrameResult(frame=overlay_frame)
        result.traffic_congestion = bool(congestion_regions)
        # 本帧数值化指标（供 FastAPI 前端展示/统计；纯 CPU 只读汇总，不参与绘图，不影响原结果）
        result.metrics = {
            "total_vehicles": int(vehicle_boxes.shape[0]),
            "mask_density": {str(k): round(float(v), 4) for k, v in density_info.items()},
            "mask_speed": {str(k): round(float(v), 2) for k, v in speed_info.items()},
            "congestion_regions": [[int(x1), int(y1), int(x2), int(y2)]
                                    for x1, y1, x2, y2 in congestion_regions],
        }
        # 汇总异常事件：自行车/行人/停止车辆/慢速车辆（事故检测暂时关闭）
        result.abnormal_events = self._collect_abnormal_events(
            bike_detections, person_detections, stopped_vehicles, slow_vehicles
        )
        # 有异常事件则保存一帧图像（一帧可有多个异常事件，但只保存一张，命名不含事件名称）
        if result.abnormal_events:
            result.output_filename = self._save_abnormal_event(overlay_frame)
        return result

    # ==================== 帧间流水线（异步编排 + 阶段线程执行，多帧重叠） ====================
    # 拓扑：  提交(S队列) → [Seg线程:cuda0] → {Lane队列:cuda2, Vehicle队列:cuda1} → [P线程:CPU]
    # 每个阶段都是“单消费者 FIFO”：天然保序（追踪器跨帧状态安全）且单飞（模型线程安全）；
    # 相邻帧可同时处于不同阶段 → 不同 GPU/CPU 资源重叠使用。
    # 本文件为“纯流水线版”：对外入口就是下面的 start_pipeline/submit_frame/get_result/close_pipeline。

    def start_pipeline(self):
        """开始一个新的流水线会话（线程已在 __init__ 常驻，这里只复位会话状态）。"""
        if self._pipeline_on:
            return
        # 复位会话状态（上一会话已通过 close_pipeline 结束，这里做防御性清理）
        with self.results_cond:
            self.lane_results.clear()
            self.vehicle_results.clear()
            self.seg_payload.clear()
        self._pipeline_seq = 0
        self._pipeline_total = None
        self._seg_done = False
        self._session_finished.clear()
        # 防御性排空上一会话可能残留的任务/结果（正常 close 后应为空；异常中止时防串会话）
        for _q in (self.seg_queue, self.out_queue, self.lane_queue, self.vehicle_queue):
            while True:
                try:
                    _q.get_nowait()
                except queue.Empty:
                    break
        self._pipeline_on = True

    def submit_frame(self, frame: np.ndarray) -> int:
        """提交一帧进入流水线（S 阶段队列），返回帧号。队列满时阻塞等待（背压）。"""
        if not self._pipeline_on:
            self.start_pipeline()
        seq = self._pipeline_seq
        self._pipeline_seq += 1
        self.seg_queue.put((seq, frame), timeout=60.0)
        return seq

    def get_result(self, timeout: float = 5.0):
        """按帧号顺序取回一个结果，返回 (seq, ProcessFrameResult)；超时返回 (None, None)。"""
        try:
            return self.out_queue.get(timeout=timeout)
        except queue.Empty:
            return None, None

    def close_pipeline(self):
        """结束本会话：给 Seg 线程发“结束哨兵”，等 P 线程处理完所有已提交帧后复位会话开关。
        线程并不退出（常驻，供下一会话复用）。调用前请先 get_result 取回全部结果。"""
        if not self._pipeline_on:
            return
        self._pipeline_total = self._pipeline_seq
        self.seg_queue.put(None)                     # 哨兵排在所有帧之后
        if not self._session_finished.wait(timeout=30.0):
            print("close_pipeline 等待本会话结束超时")
        self._pipeline_on = False

    def _pipeline_segment_worker(self):
        """S 阶段（常驻线程）：分割(GPU cuda:0) → 分发 L/V 队列。
        无会话时休眠；收到 None 哨兵表示本会话不再有新帧（并不退出线程）。"""
        while not self.stop_event.is_set():
            if not self._pipeline_on:
                time.sleep(0.05)
                continue
            try:
                task = self.seg_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if task is None:
                # 本会话帧已全部提交：置结束标志并叫醒 P 线程做收尾判断
                with self.results_cond:
                    self._seg_done = True
                    self.results_cond.notify_all()
                continue
            seq, frame = task
            try:
                if frame.shape[0] != self.input_height or frame.shape[1] != self.input_width:
                    frame = cv2.resize(frame, (self.input_width, self.input_height))
                masked_frame, overlay_frame, road_masks, masks_np = self._segment_road(frame)
            except Exception as e:
                print(f"PipeSeg error: {e}")
                with self.results_cond:
                    self.seg_payload[seq] = (".fail", frame)
                    self.results_cond.notify_all()
                continue

            if masks_np is None:
                # 分割模型未检出道路 → 无需 L/V，直接标记该帧按异常产出
                with self.results_cond:
                    self.seg_payload[seq] = (".fail", frame)
                    self.results_cond.notify_all()
                continue

            # 阻塞入队（超时保护），保证不丢帧、不破坏帧号顺序
            self.lane_queue.put((seq, masked_frame), timeout=60.0)
            self.vehicle_queue.put((seq, masked_frame, road_masks), timeout=60.0)
            with self.results_cond:
                self.seg_payload[seq] = (frame, overlay_frame, masks_np)
                self.results_cond.notify_all()

    def _pipeline_post_worker(self):
        """P 阶段（常驻线程）：按帧号顺序等待 (S/L/V) 就绪 → 执行 _run_post_stage（纯 CPU）→ 输出队列。
        单消费者保证跨帧状态（速度/缩放因子/历史目标）只被串行按序更新。
        无会话时休眠；本会话全部处理完则发完成信号并复位，等待下一会话。
        兜底：某帧长时间凑不齐 L/V 结果（worker 异常）时按异常帧产出，保证不挂起。"""
        seq = 0
        stall_since = None
        while not self.stop_event.is_set():
            if not self._pipeline_on:
                time.sleep(0.05)
                continue

            ready = None            # None=未就绪；("ok",...)/("fail",...) = 已取走该帧全部产物
            with self.results_cond:
                seg_ready = seq in self.seg_payload
                payload = self.seg_payload.get(seq)
                is_fail = isinstance(payload, tuple) and len(payload) == 2 and payload[0] == ".fail"

                if not seg_ready:
                    if (self._pipeline_total is not None and seq >= self._pipeline_total
                            and self._seg_done):
                        # 本会话全部帧已处理完：通知 close_pipeline，复位帧号，等开关关闭后进入空闲
                        self._session_finished.set()
                        seq = 0
                        stall_since = None
                        while self._pipeline_on and not self.stop_event.is_set():
                            self.results_cond.wait(timeout=0.1)
                        continue
                elif is_fail:
                    self.seg_payload.pop(seq)
                    ready = ("fail", payload[1])
                else:
                    lane_ok = seq in self.lane_results
                    veh_ok = seq in self.vehicle_results
                    if lane_ok and veh_ok:
                        lane_boxes = self.lane_results.pop(seq)
                        veh_pkt = self.vehicle_results.pop(seq)
                        self.seg_payload.pop(seq)
                        ready = ("ok", payload, lane_boxes, veh_pkt)

                if ready is None:
                    if stall_since is None:
                        stall_since = time.time()
                    stalled = time.time() - stall_since > 30.0
                    seg_finished = (self._pipeline_total is not None and self._seg_done)
                    if stalled and seg_finished:
                        # 兜底：分割已结束却仍凑不齐该帧 L/V 结果 → 按异常产出并推进
                        self.seg_payload.pop(seq, None)
                        self.lane_results.pop(seq, None)
                        self.vehicle_results.pop(seq, None)
                        ready = ("fail", None)
                        print(f"PipePost 兜底跳过 seq={seq}（缺少 L/V 结果）")
                    else:
                        self.results_cond.wait(timeout=0.5)
                        continue
                else:
                    stall_since = None

            # 持锁外执行（可能耗时）
            if ready[0] == "fail":
                raw = ready[1]
                if raw is None:
                    res = ProcessFrameResult(frame=np.zeros(
                        (self.input_height, self.input_width, 3), dtype=np.uint8), is_abnormal=True)
                else:
                    res = ProcessFrameResult(frame=raw.copy(), is_abnormal=True)
            else:
                _, payload, lane_boxes, veh_pkt = ready
                raw, overlay_frame, masks_np = payload
                try:
                    res = self._run_post_stage(raw, lane_boxes, veh_pkt, overlay_frame, masks_np)
                except Exception as e:
                    # 单帧后处理异常 → 按异常帧产出，保证流水线不断
                    print(f"PipePost error seq={seq}: {e}")
                    res = ProcessFrameResult(frame=np.zeros(
                        (self.input_height, self.input_width, 3), dtype=np.uint8), is_abnormal=True)
            self.out_queue.put((seq, res), timeout=60.0)
            seq += 1

    def _collect_abnormal_events(self, bike_detections, person_detections, stopped_vehicles,
                                 slow_vehicles) -> List[AbnormalEvent]:
        """将各类检测结果统一映射为异常事件列表"""
        events = []
        if bike_detections:
            events.append(AbnormalEvent(event_type=AbnormalEventType.BIKE, detections=bike_detections))
        if person_detections:
            events.append(AbnormalEvent(event_type=AbnormalEventType.PERSON, detections=person_detections))
        if stopped_vehicles:
            events.append(AbnormalEvent(event_type=AbnormalEventType.STOPPED_VEHICLE, detections=stopped_vehicles))
        if slow_vehicles:
            events.append(AbnormalEvent(event_type=AbnormalEventType.SLOW_VEHICLE, detections=slow_vehicles))
        return events

    def _save_abnormal_event(self, frame: np.ndarray) -> str:
        """保存异常事件对应帧（一帧一张图，命名不含事件名称），返回输出文件名"""
        self.count += 1
        filename = f"{self.count:05d}.jpg"
        absolute_filename = os.path.join(self.output_dir, filename)
        try:
            self.save_queue.put_nowait((absolute_filename, frame.copy()))
        except queue.Full:
            pass  # 队列满了丢弃，保证主流程不阻塞
        return filename

    def _prune_results(self, result_dict, current_frame_id: int, keep: int = 32):
        """清理过期帧的结果，避免长时间运行导致内存泄漏"""
        threshold = current_frame_id - keep
        for fid in [f for f in result_dict if f < threshold]:
            del result_dict[fid]

    def _extract_roi(self, masks: torch.Tensor) -> Optional[torch.Tensor]:
        """提取道路ROI"""
        small_mask = self.input_height * self.input_width / 100

        with torch.no_grad():
            areas = masks.view(masks.size(0), -1).sum(dim=1)
            filtered_masks = masks[areas > small_mask]

            if filtered_masks.size(0) == 0:
                return None

            masks_tensor = F.interpolate(
                filtered_masks.unsqueeze(1),
                size=(self.input_height, self.input_width),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)

            return masks_tensor

    # ============================================================
    # S 节点（分割, GPU cuda:0）—— 对应伪代码 _node_segment
    # 输入一帧 → 输出 masked_frame / overlay_frame / road_masks(GPU,给V) / masks_np(CPU,给P)；
    # 分割为空时返回 (frame, frame, None, None)，由上层按异常帧处理。
    # ============================================================
    def _segment_road(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[torch.Tensor], Optional[np.ndarray]]:
        """
        道路分割并生成两类输出图像。

        Args:
            frame: 输入帧（BGR，H×W×3，numpy 数组）

        Returns:
            (road_masked_frame, overlay_frame, road_masks, road_masks_np):
                road_masked_frame : 全遮蔽图像（道路外区域填充固定色），供目标检测模型使用
                overlay_frame     : 半透明叠加图像（全遮蔽帧与原始帧加权融合），供可视化使用
                road_masks        : 道路掩码张量（GPU，供 V worker 分组），未检测到道路时为 None
                road_masks_np     : road_masks 的 CPU 副本（纯 numpy，供密度统计，P 阶段不碰 GPU）

        说明：
            - 掩码提取、全遮蔽帧生成、半透明融合均在 segment_device 上完成，
              避免中间结果在 GPU/CPU 间来回搬运；仅在收尾处一次性产出 CPU 掩码副本。
            - 未检测到道路时回退为原帧，保证调用链不中断（road_masks 为 None 表示失败）。
        """
        if frame is None or frame.size == 0:
            raise ValueError("_segment_road: 输入 frame 为空")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"_segment_road: 输入 frame 应为三通道图像，实际 shape={frame.shape}")

        # 单次搬运到分割模型所在 GPU
        frame_tensor = torch.from_numpy(np.ascontiguousarray(frame)).to(
            self.segment_device
        ).permute(2, 0, 1).float()

        seg_results = self.segment_model.predict(
            frame, imgsz=960, verbose=False, half=True, device=self.segment_device
        )

        # 分割模型空检出判定准则：r.masks is None 表示本帧没有分割出道路对象
        if not seg_results or seg_results[0].masks is None:
            return frame, frame, None, None

        road_masks = self._extract_roi(seg_results[0].masks.data)
        if road_masks is None:
            return frame, frame, None, None

        # GPU 上同时生成全遮蔽帧与半透明叠加帧，一次前向 + 一次矩阵运算，无 CPU 往返
        with torch.no_grad():
            keep_road = road_masks.any(dim=0).unsqueeze(0)      # [1, H, W]
            keep_road_3c = keep_road.expand(3, -1, -1)          # [3, H, W]

            # 遮蔽色（BGR）：与历史实现保持一致 [255, 0, 0]
            mask_color = torch.tensor([255.0, 0.0, 0.0],
                                      device=self.segment_device,
                                      dtype=torch.float32).view(3, 1, 1)

            masked_tensor = torch.where(keep_road_3c, frame_tensor, mask_color)   # 全遮蔽
            overlay_tensor = masked_tensor * 0.3 + frame_tensor * 0.7             # 半透明融合

        masked_frame = masked_tensor.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
        overlay_frame = overlay_tensor.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
        # 收尾处一次性产出 CPU 掩码副本：P 阶段密度统计不再触碰 GPU；GPU 版 road_masks 留给 V worker 分组
        masks_np = road_masks.cpu().numpy()

        return masked_frame, overlay_frame, road_masks, masks_np

    def reset_video_state(self):
        """重置所有跨帧/跨视频状态，使同一实例可安全开始处理一段新视频。
        适用于服务端复用同一 TrafficAnalyzer 连续处理多段视频/多次流式会话。
        注意：不重置 self.count（事件保存文件名保持全局递增，避免重名覆盖）。"""
        with self.results_cond:
            self.lane_results.clear()
            self.vehicle_results.clear()
            if hasattr(self, 'seg_payload'):
                self.seg_payload.clear()
        # 跨帧追踪/速度状态（车道线、车辆轨迹、位移与缩放因子）
        self.past_dis.clear()
        self.previous_frame_objects = {}
        self.previous_frame_lines = {}
        self.past_drone_speed = 10.0
        self.past_scaling_factor = 3.0
        self.scaling_factor = 3.0

    # ---- P 子步骤 3：按车道/道路块算速度、停止车与慢速车（跨帧状态，P 线程独占保序）----
    def _process_speed(self, frame: np.ndarray, lane_boxes_np: np.ndarray, vehicle_boxes_np: np.ndarray,
                       enable_stop_slow: bool = True) -> Tuple[np.ndarray, Dict, List, List]:
        # 车道框与车辆分组数组均由 worker CPU 化（numpy），此处全部为 CPU 逻辑
        if lane_boxes_np.shape[0] == 0 or lane_boxes_np.shape[1] == 6:  # no id column
            drone_speed = self.past_drone_speed
        else:
            current_frame_lines = self.get_current_frame_objects(lane_boxes_np, 0, False)
            drone_speed = self.calculate_drone_speed(frame, current_frame_lines)
            self.past_drone_speed = drone_speed

        masks_speed_info = {}
        current_frame_objects = self.get_current_frame_objects(vehicle_boxes_np)
        track_id2v, stopped_car, slow_car = self.calculate_cars_speed(frame, current_frame_objects, drone_speed, enable_stop_slow=enable_stop_slow)  # 绘图函数

        if len(track_id2v) > 0:
            # 提取所有 track_id 和对应的 mask_id（列位置与今日 GPU 版本完全一致）
            track_ids = vehicle_boxes_np[:, 4]
            mask_ids = vehicle_boxes_np[:, -1]

            speed_values = np.full(len(track_ids), -1.0, dtype=np.float32)

            for i in range(len(track_ids)):
                # 将 track_id 转换为整数（解决 1.0 vs 1 的问题）
                track_id_int = int(track_ids[i])

                # 如果 track_id 在字典中，获取速度值
                if track_id_int in track_id2v:
                    speed_values[i] = track_id2v[track_id_int]
                else:
                    # 缺失 track_id：使用 -1 表示缺失值，后续计算平均速度时会排除
                    speed_values[i] = -1

            # 计算每个 mask_id 的平均速度
            for mask_id in np.unique(mask_ids):
                mask_id_int = int(mask_id)
                if mask_id_int == -1:
                    continue

                mask_sel = mask_ids == mask_id
                mask_speed = speed_values[mask_sel]

                # 排除无效值（-1）
                valid_speeds = mask_speed[mask_speed >= 0]

                if len(valid_speeds) > 0:
                    avg_speed = float(valid_speeds.mean())
                    masks_speed_info[mask_id_int] = avg_speed
                else:
                    # 没有有效速度数据
                    masks_speed_info[mask_id_int] = 100
        return frame, masks_speed_info, stopped_car, slow_car



    def shutdown(self):
        """安全关闭所有线程（线程全在 __init__ 启动，这里统一收尾）。"""
        # 先排空异步写入队列：趁写图线程还活着把已入队任务写完
        self.save_queue.join()
        self.save_queue.put(None)      # 写图线程哨兵
        self.stop_event.set()          # 举停止旗：所有线程跑完手头这轮后退出
        for t, to in ((self.segment_thread, 1.0), (self.post_thread, 1.0),
                      (self.lane_thread, 1.0), (self.vehicle_thread, 1.0),
                      (self.save_thread, 2.0)):
            t.join(timeout=to)

    def get_current_frame_objects(self, data, attention_id=2, get_scaling_factor=True):
        marge = 25
        current_frame_objects = {}

        points = []  # 存储左上角和右下角点
        valid_rows = []  # 存储对应的行数据

        # 第一遍遍历：收集所有需要处理的点
        for row in data:
            # 提取数据
            x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            if y1 < marge or y2 > self.input_height - marge:
                continue

            track_id = int(row[4])
            class_id = int(row[6])

            if class_id != attention_id:
                continue

            # 添加左上角和右下角点
            points.append([x1, y1])  # 左上角
            points.append([x2, y2])  # 右下角

            # 保存相关信息
            valid_rows.append((center_x, center_y, track_id, class_id))

        # 如果没有符合条件的行，直接返回
        if not points:
            return current_frame_objects

        points_array = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

        try:
            transformed_points = cv2.perspectiveTransform(points_array, self.matrix)
        except Exception as e:
            print(f"透视变换失败: {e}")
            transformed_points = points_array.copy()

        scaling_factor_lst = []
        for i, (center_x, center_y, track_id, class_id) in enumerate(valid_rows):
            # 获取变换后的点
            idx = i * 2  # 每个目标对应两个点
            top_left_t = transformed_points[idx, 0]
            bottom_right_t = transformed_points[idx + 1, 0]

            if (top_left_t[0] < 0 or top_left_t[0] > self.input_width or
                    top_left_t[1] < 0 or top_left_t[1] > self.input_height or
                    bottom_right_t[0] < 0 or bottom_right_t[0] > self.input_width or
                    bottom_right_t[1] < 0 or bottom_right_t[1] > self.input_height):
                # 任意一点超出范围，舍弃该目标
                continue

            # 计算变换后的框尺寸
            box_w_t = abs(bottom_right_t[0] - top_left_t[0])
            box_h_t = abs(bottom_right_t[1] - top_left_t[1])

            # 计算变换后的中心点
            center_x_t = (top_left_t[0] + bottom_right_t[0]) / 2
            center_y_t = (top_left_t[1] + bottom_right_t[1]) / 2

            # 计算缩放因子（基于变换后的框大小）
            scaling_factor_lst.append(2.25 / box_w_t * 36)

            # 存入字典
            current_frame_objects[track_id] = [
                int(center_x),
                int(center_y),
                int(center_x_t),
                int(center_y_t),
                int(box_w_t),
                int(box_h_t)
            ]

        if get_scaling_factor:
            if not scaling_factor_lst:
                self.scaling_factor = self.past_scaling_factor
            else:
                scaling_factor = sum(scaling_factor_lst) / (len(scaling_factor_lst) + 1e-10)
                self.scaling_factor = 0.2 * scaling_factor + 0.8 * self.past_scaling_factor
                self.past_scaling_factor = self.scaling_factor
        return current_frame_objects


    def calculate_cars_speed(self, frame, current_frame_objects, drone_speed, enable_stop_slow: bool = True):
        stopped_car, slow_car = [], []
        track_id2v = {}
        max_tracker_id = 0
        # 如果有前一帧的数据，则计算位移
        if self.previous_frame_objects:
            for track_id, current_pos in current_frame_objects.items():
                if track_id in self.previous_frame_objects:
                    prev_pos = self.previous_frame_objects[track_id]
                    if (track_id > max_tracker_id):
                        max_tracker_id = track_id
                    dx = current_pos[2] - prev_pos[2]
                    dy = current_pos[3] - prev_pos[3]
                    dis = abs(dy + 1e-5) / (dy + 1e-5) * np.sqrt(dx * dx + dy * dy)
                    # dis = abs(dy + 1e-5) / (dy + 1e-5) * dy
                    if track_id in self.past_dis:
                        past_dis = self.past_dis[track_id]
                        dis = np.clip(dis, past_dis - 5, past_dis + 5)
                    self.past_dis[track_id] = dis
                    v = abs(dis * self.scaling_factor - drone_speed)
                    v = int((-0.006 * v + 1.39) * v)
                    if v < 3:
                        v = 0
                        if enable_stop_slow:
                            stopped_car.append([int(current_pos[0] - current_pos[4] / 2), int(current_pos[1] - current_pos[5] / 2),
                                                int(current_pos[0] + current_pos[4] / 2), int(current_pos[1] + current_pos[5] / 2)])
                            cv2.rectangle(frame, (int(current_pos[0] - current_pos[4] / 2), int(current_pos[1] - current_pos[5] / 2)),
                                          (int(current_pos[0] + current_pos[4] / 2), int(current_pos[1] + current_pos[5] / 2)), (0, 0, 255), 1)
                            label = f'Stopped car v={v}'
                            # print('dy',dy,'v',v)
                            t_size = cv2.getTextSize(label, 0, fontScale=0.35, thickness=1)[0]

                            cv2.rectangle(frame, (int(current_pos[0] - t_size[0] / 2) , current_pos[1] - t_size[1] - 3),
                                          (int(current_pos[0] + t_size[0] / 2), current_pos[1] + 3), (0, 255, 0), -1)
                            cv2.putText(frame, label, (int(current_pos[0] - t_size[0] / 2), current_pos[1]),
                                        0, 0.35, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                    elif 3 <= v <20:
                        if enable_stop_slow:
                            slow_car.append([int(current_pos[0] - current_pos[4] / 2), int(current_pos[1] - current_pos[5] / 2),
                                                int(current_pos[0] + current_pos[4] / 2), int(current_pos[1] + current_pos[5] / 2)])
                            # cv2.rectangle(frame, (
                            # int(current_pos[0] - current_pos[4] / 2), int(current_pos[1] - current_pos[5] / 2)),
                            #               (int(current_pos[0] + current_pos[4] / 2),
                            #                int(current_pos[1] + current_pos[5] / 2)), (0, 0, 255), 1)
                            label = f'Slow car v={v}'
                            # print('dy',dy,'v',v)
                            t_size = cv2.getTextSize(label, 0, fontScale=0.35, thickness=1)[0]
                            cv2.rectangle(frame, (int(current_pos[0] - t_size[0] / 2), current_pos[1] - t_size[1] - 3),
                                          (int(current_pos[0] + t_size[0] / 2), current_pos[1] + 3), (0, 255, 0), -1)
                            cv2.putText(frame, label, (int(current_pos[0] - t_size[0] / 2), current_pos[1]),
                                        0, 0.35, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

                    else:
                        # 标记位移在当前帧图像上
                        label = f'v={v}'
                        # print('dy',dy,'v',v)
                        t_size = cv2.getTextSize(label, 0, fontScale=0.35, thickness=1)[0]
                        cv2.rectangle(frame, (int(current_pos[0] - t_size[0] / 2), current_pos[1] - t_size[1] - 3),
                                      (int(current_pos[0] + t_size[0] / 2), current_pos[1] + 3), (0, 255, 0), -1)
                        cv2.putText(frame, label, (int(current_pos[0] - t_size[0] / 2), current_pos[1]),
                                    0, 0.35, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                    track_id2v[track_id] = v
        self.previous_frame_objects = current_frame_objects.copy()
        return track_id2v, stopped_car, slow_car

    def calculate_drone_speed(self, frame, current_frame_objects):
        # 如果有前一帧的数据，则计算无人机速度
        if self.previous_frame_lines:
            v_lst = []
            for track_id, current_pos in current_frame_objects.items():
                if track_id in self.previous_frame_lines:
                    prev_pos = self.previous_frame_lines[track_id]
                    dy = current_pos[3] - prev_pos[3]
                    v = dy * self.scaling_factor
                    v_lst.append(v)
            v_average = sum(v_lst) / (len(v_lst) + 1e-10)

        else:
            v_average = 18

        v_average = np.clip(v_average, 0, 23)
        x_offset = 50
        y_offset = 50
        label = f'Drone speed: {int(0.2 * v_average + 0.8 * self.past_drone_speed)}'
        t_size = cv2.getTextSize(label, 0, fontScale=0.45, thickness=1)[0]
        cv2.rectangle(frame, (x_offset, y_offset - t_size[1] - 3), (x_offset + t_size[0], y_offset + 3), (0, 255, 0), -1)
        cv2.putText(frame, label, (x_offset, y_offset), 0, 0.45, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        self.previous_frame_lines = current_frame_objects.copy()
        return int(0.2 * v_average + 0.8 * self.past_drone_speed)

    # ---- V 节点子步骤：车辆框按道路掩码分组（GPU 内完成；末列为掩码索引）----
    def _group_by_mask(self, masks_tensor: torch.Tensor, vehicle_results: List) -> torch.Tensor:
        """按道路掩码给车辆检测框分组，返回 [N, 7] 张量（x1,y1,x2,y2,conf,cls + 掩码索引列，无掩码为 -1）。

        Args:
            masks_tensor: 道路掩码张量 [num_masks, H, W]，位于 segment_device
            vehicle_results: 车辆模型推理结果（Ultralytics Results 列表，非 Tensor），
                其 boxes.data 位于 vehicle_device，函数内统一搬运到 masks_tensor 所在设备后
                全程在 GPU 上索引计算，无 CPU 往返
        """
        device = masks_tensor.device
        boxes = getattr(vehicle_results[0], "boxes", None)
        if boxes is None or boxes.data.numel() == 0:
            # 无车辆框：返回空 [0, 7] 张量（列结构与正常结果一致，下游可统一处理）
            return torch.empty((0, 7), device=device, dtype=torch.float32)

        boxes_data_tensor = boxes.data.to(device)

        # 提取检测框左上角坐标 (x1, y1)
        top_left = boxes_data_tensor[:, :2].round().long()
        x_coords = top_left[:, 0].clamp(0, self.input_width - 1)
        y_coords = top_left[:, 1].clamp(0, self.input_height - 1)

        # 一次性获取所有掩码在这些点上的值 [num_masks, num_boxes]
        mask_values = masks_tensor[:, y_coords, x_coords] > 0.5

        # 找到每个框落在哪个掩码里
        has_mask = mask_values.any(dim=0)
        indices = torch.argmax(mask_values.float(), dim=0)
        indices[~has_mask] = -1

        return torch.cat([boxes_data_tensor, indices.unsqueeze(1)], dim=1)

    # ---- P 子步骤 1：自行车/行人检测框分类（可选，受 enable_bike_person 控制）----
    def _process_bike_person(self, frame: np.ndarray, boxes_np: Optional[np.ndarray]) -> Tuple[List, List]:
        """在 frame 上绘制自行车/行人检测框，返回各自框坐标列表。

        Args:
            frame: 可视化用图像（BGR, H×W×3, numpy, CPU 内存），会被原地绘制
            boxes_np: 车辆模型检测框数组（worker 内已 CPU 化，列含义与 boxes.data 完全一致）

        说明：
            直接对 numpy 数组做向量化按类别过滤，无 GPU→CPU 同步；
            目标类别固定为 bike=0 / person=3（bus/car/truck 不绘制）。
        """
        bike, person = [], []
        if boxes_np is None or boxes_np.size == 0:
            return bike, person

        cls = boxes_np[:, 5]

        # 车辆模型 5 类：0 bike / 1 bus / 2 car / 3 person / 4 truck
        for class_id, targets in ((0, bike), (3, person)):
            label = "bike" if class_id == 0 else "person"
            for x1, y1, x2, y2 in boxes_np[cls == class_id, :4].astype(int):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                targets.append([x1, y1, x2, y2])

        return bike, person

    # ---- P 子步骤 2：各道路块密度统计（纯 CPU，吃 masks_np + 车辆框）----
    def _process_traffic_density(self, image: np.ndarray, masks_np: np.ndarray, boxes_data_np: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        统计各道路掩码区域内的车辆密度并标注到图像上（纯 CPU numpy 实现）。

        Args:
            image: 待标注图像（BGR, numpy 数组）
            masks_np: 道路掩码数组 [num_masks, H, W]（float32，CPU），由 _segment_road 一并产出
            boxes_data_np: 车辆分组结果（末列为掩码索引，CPU numpy），由 vehicle worker 产出；
                可能为空（无车辆框），此时所有掩码密度补 0，保证返回结构完整

        Returns:
            image: 标注后的图像
            density_info: {mask_id: traffic_density}，无有效车辆时所有掩码密度为 0
        """
        height, width = image.shape[:2]
        font_scale = 0.45
        thickness = 1
        car_pixels = 16200 / self.scaling_factor / self.scaling_factor

        masks_f = masks_np.astype(np.float32, copy=False)

        # ===== CPU 统计部分（等价于原 GPU 向量化实现；坐标网格只生成一次并缓存，避免每帧重建） =====
        if (self._density_yf is None or self._density_H != height or self._density_W != width):
            # [H*W] 每像素对应的行号 / 列号（行主序展开，与 masks_np.reshape(N, -1) 对应）
            self._density_yf = np.repeat(np.arange(height), width).astype(np.float32)
            self._density_xf = np.tile(np.arange(width), height).astype(np.float32)
            self._density_H, self._density_W = height, width

        m = masks_f.reshape(masks_f.shape[0], -1)          # [N, H*W]
        yf, xf = self._density_yf, self._density_xf

        # 1. 一次矩阵乘法算出所有掩码的面积与质心（与 torch 的 sum // 后 .long() 等价）
        mask_sums = m.sum(axis=1)                          # [N]
        denom = np.maximum(mask_sums, 1.0)
        centroid_y = np.floor((m @ yf) / denom).astype(np.int64)
        centroid_x = np.floor((m @ xf) / denom).astype(np.int64)
        centroids = np.stack([centroid_y, centroid_x], axis=1).astype(np.int32)   # [N, 2]

        # 2. 统计每个掩码区域内的车辆信息
        area_statistics = {}
        total_vehicles = 0

        if boxes_data_np is None or boxes_data_np.size == 0:
            # 无车辆框：为每个掩码补 0 密度，保证返回的 density_info 结构完整（而非空字典）
            for mask_id in range(masks_f.shape[0]):
                area_statistics[mask_id] = {
                    'vehicle_count': 0,
                    'traffic_density': 0.0,
                    'centroid': centroids[mask_id]
                }
        else:
            # 末列为掩码索引；只统计落在道路掩码内（>=0）的车辆
            mask_col = boxes_data_np[:, -1]
            valid_mask_ids = np.unique(mask_col[mask_col >= 0])

            for mask_id in valid_mask_ids:
                mask_id_int = int(mask_id)
                mask_boxes = boxes_data_np[mask_col == mask_id]
                vehicle_count = int(mask_boxes.shape[0])
                total_vehicles += vehicle_count

                area_pixels = float(mask_sums[mask_id_int])
                traffic_density = car_pixels * vehicle_count / area_pixels if area_pixels > 0 else 0
                traffic_density = min(traffic_density, 0.99)  # 限制最大密度

                area_statistics[mask_id_int] = {
                    'vehicle_count': vehicle_count,
                    'traffic_density': traffic_density,
                    'centroid': centroids[mask_id_int]
                }

        # ===== CPU可视化部分 =====
        # 3. 在每个掩码区域上标注掩码ID（使用质心位置）
        centroids_cpu = centroids  # 已是 numpy 数组

        for mask_id in range(masks_np.shape[0]):
            # mask = masks_np[mask_id]
            # if not np.any(mask):
            #     continue

            # 获取GPU计算出的质心
            centroid_y, centroid_x = centroids_cpu[mask_id]

            # 确保质心在图像范围内
            centroid_x = max(0, min(centroid_x, width - 1))
            centroid_y = max(0, min(centroid_y, height - 1))

            # 在掩码质心绘制ID
            id_label = f"Area {mask_id + 1}"
            (id_width, id_height), _ = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX,
                                                       font_scale, thickness)

            # 绘制背景矩形
            cv2.rectangle(image,
                          (centroid_x - id_width // 2 - 5, centroid_y - id_height // 2 - 5),
                          (centroid_x + id_width // 2 + 5, centroid_y + id_height // 2 + 5),
                          (255, 255, 255), -1)

            # 绘制文本
            cv2.putText(image, id_label,
                        (centroid_x - id_width // 2, centroid_y + id_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 0, 0), thickness, lineType=cv2.LINE_AA)

        # 4. 在右上角显示统计信息
        # 总车辆数
        total_label = f"The total number of cars: {total_vehicles}"
        (total_width, total_height), _ = cv2.getTextSize(total_label, cv2.FONT_HERSHEY_SIMPLEX,
                                                         font_scale, thickness)
        x_offset = width - total_width - 50
        y_offset = 50

        # 绘制背景矩形
        cv2.rectangle(image,
                      (x_offset - 5, y_offset - total_height - 5),
                      (x_offset + total_width + 5, y_offset + 5),
                      (255, 255, 255), -1)  # 红色背景

        # 绘制总车辆数文本
        cv2.putText(image, total_label, (x_offset, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), thickness, lineType=cv2.LINE_AA)  # 黑色文本

        # 各区域统计信息
        y_offset += 30  # 下移一行
        density_statistics = {}
        for mask_id, info in area_statistics.items():
            vehicle_count = info['vehicle_count']
            traffic_density = info['traffic_density']
            density_statistics[mask_id] = traffic_density
            info_label = f"Area {mask_id + 1} vehicles:{vehicle_count} density:{traffic_density:.2f}"

            (text_width, text_height), _ = cv2.getTextSize(info_label, cv2.FONT_HERSHEY_SIMPLEX,
                                                           font_scale, thickness)
            # 绘制背景矩形
            cv2.rectangle(image,
                          (x_offset - 5, y_offset - text_height - 5),
                          (x_offset + text_width + 5, y_offset + 5),
                          (255, 255, 255), -1)

            # 绘制文本
            cv2.putText(image, info_label, (x_offset, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 0, 0), thickness, lineType=cv2.LINE_AA)

            y_offset += 30  # 下移一行
        return image, density_statistics

    # ---- P 子步骤 4：拥堵评估（读密度 + 速度结果，产出拥堵标志/区域）----
    def _evaluate_congestion(self, frame: np.ndarray, mask_density_info: Dict, mask_speed_info: Dict, vehicle_results_boxes: np.ndarray) -> Tuple[np.ndarray, Dict, List]:
        # vehicle_results_boxes 为 CPU numpy（末列为掩码索引）；numpy 同样支持 min/item/布尔索引，逻辑不变
        crowd = []
        common_keys = set(mask_density_info.keys()) & set(mask_speed_info.keys())
        combined_dict = {key: [mask_density_info[key], mask_speed_info[key] / 100, (1 - mask_density_info[key] + mask_speed_info[key] / 100) / 2] for key in common_keys}

        for mask_id, value in combined_dict.items():
            if (value[0] > 0.10 and value[1] < 0.10) or (value[0] > 0.25 and value[1] < 0.25):
            # if value < 0.45:
                # 2.1 提取当前掩码的所有框
                mask_mask = vehicle_results_boxes[:, -1] == mask_id
                mask_boxes = vehicle_results_boxes[mask_mask]

                # 2.2 检查是否有框
                if mask_boxes.shape[0] == 0:
                    continue

                # 2.3 计算包围矩形
                # 获取第一列的最小值 (x1_min)
                x1_min = mask_boxes[:, 0].min()
                # 获取第二列的最小值 (y1_min)
                y1_min = mask_boxes[:, 1].min()
                # 获取第三列的最大值 (x2_max)
                x2_max = mask_boxes[:, 2].max()
                # 获取第四列的最大值 (y2_max)
                y2_max = mask_boxes[:, 3].max()

                # 2.4 转换为整数坐标
                x1 = int(x1_min.item())
                y1 = int(y1_min.item())
                x2 = int(x2_max.item())
                y2 = int(y2_max.item())

                # 2.5 绘制矩形
                crowd.append([x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return frame, combined_dict, crowd

