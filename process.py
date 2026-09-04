"""
process.py — 交通视频帧间流水线引擎
====================================================

一帧的数据流（S → L∥V → P）
    一帧图
      → [S 分割   cuda:0] 产出 masked_frame / overlay_frame / masks_tensor(GPU) / masks_np(CPU)
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
    process_frame(frame) 单帧阻塞版：帧内 DAG 编排（S → L∥V → P），单图/调试用；勿与流水线会话并发使用
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
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
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

    SEGMENT_DEVICE = "cuda:4"
    VEHICLE_DEVICE = "cuda:5"
    LANE_DEVICE = "cuda:6"

    INPUT_WIDTH = 960
    INPUT_HEIGHT = 540

    # L/V 检测推理分辨率（长边）。
    # 注意：不显式传 imgsz 时，ultralytics 会沿用 best.pt 训练参数里的 960（960x960 方形），
    # 实测 yolo11m 在该尺寸约 180-190ms/帧（还是与另一路并发）。调小立即可加速，代价是精度：
    #   640 → 像素量约为 960 的 44%，约 2.2x 加速；512 → 约 28%，约 3.5x。
    # 推荐从 640 起测（视频本身就是 960x540，640 长边足够），远小目标多时再酌情上调。
    TRACK_IMGSZ = 640

    # 推理串行开关。本机（多 RTX4080 共享机）实测：Python 线程同时驱动多张 GPU 推理会
    # 灾难性串行化——单路 predict ~15ms，一旦两路并发每路变 ~250ms（且与限制 torch/cv2
    # 线程数无关，S∥V、L∥V 都如此）；而顺序执行两路仅 ~46ms。
    # 因此默认 True：S/L/V 任一时刻只跑一个模型（单帧 S→L→V→P，~65ms/帧）。
    # 若部署机线程级多 GPU 并发正常，可置 False 恢复三卡并行。
    SERIAL_INFERENCE = True

    def __init__(self, config: Optional[DetectionConfig] = None):
        """
        初始化处理器 - 优化版

        Args:
            config: 客户端配置（仅功能开关与输出目录），默认使用默认配置
        """
        self.config = config or DetectionConfig()
        # 推理串行锁：SERIAL_INFERENCE=True 时 S/L/V 的 GPU 推理互斥（见类常量说明）
        self._infer_lock = threading.RLock()

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

        # 启动所有线程（同寿命：S/P/L/V 全部在 __init__ 启动，之后不再创建线程）
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



    @contextmanager
    def _infer_guard(self):
        """推理互斥上下文：SERIAL_INFERENCE=True 时串行化 S/L/V 的 GPU 推理。
        本机实测线程级多 GPU 并发是负优化（~15ms→~250ms/路），串行反而最快；
        并发正常的机器可把 SERIAL_INFERENCE 置 False 直接透传。
        """
        if self.SERIAL_INFERENCE:
            with self._infer_lock:
                yield
        else:
            yield

    def _start_workers(self):
        """启动全部线程（同一寿命，只在 __init__ 调用一次，之后不再创建线程）。"""
        self.segment_thread = threading.Thread(target=self._pipeline_segment_worker, name="Seg", daemon=True)
        self.post_thread = threading.Thread(target=self._pipeline_post_worker, name="P", daemon=True)
        self.lane_thread = threading.Thread(target=self._lane_worker, name="LaneWorker", daemon=True)
        self.vehicle_thread = threading.Thread(target=self._vehicle_worker, name="VehicleWorker", daemon=True)
        for t in (self.segment_thread, self.post_thread, self.lane_thread,
                  self.vehicle_thread):
            t.start()

    # ============================================================
    # L 节点（车道, GPU cuda:2）—— 对应伪代码 _node_lane
    # 单帧同步节点：track 车道线后 CPU 化，供流水线 worker 与单帧 DAG（process_frame）复用。
    # 线程安全前提：同一模型对象只能被一个调用方串行使用（worker 单飞 / 单帧与流水线互斥）。
    # ============================================================
    def _node_lane(self, masked_frame: np.ndarray) -> Optional[np.ndarray]:
        """车道线检测节点（推理串行锁入口）。"""
        with self._infer_guard():
            return self._node_lane_impl(masked_frame)

    def _node_lane_impl(self, masked_frame: np.ndarray) -> Optional[np.ndarray]:
        """车道线 track → numpy；返回 (N,6) 或含 id 列的框数组；无检出返回 None。"""
        results = self.lane_model.track(
            masked_frame,
            persist=True,
            save=False,
            verbose=False,
            half=True,
            imgsz=self.TRACK_IMGSZ,
            tracker="bytetrack.yaml",
            device=self.lane_device,
        )
        # CPU 化边界：只把下游所需的 boxes.data 转成 numpy（不再传递 Results 对象）
        lane_boxes = None
        if results:
            boxes = getattr(results[0], "boxes", None)
            lane_boxes = boxes.data.cpu().numpy() if (boxes is not None and boxes.data.numel() > 0) else None
        return lane_boxes

    # ============================================================
    # V 节点（车辆, GPU cuda:1）—— 对应伪代码 _node_vehicle
    # 单帧同步节点：车辆 track + 按道路掩码分组（GPU 内完成）→ numpy 数据包，
    # 供流水线 worker 与单帧 DAG（process_frame）复用。
    # ============================================================
    def _node_vehicle(self, masked_frame: np.ndarray, masks_tensor: torch.Tensor) -> Optional[Tuple]:
        """车辆检测+分组节点（推理串行锁入口）。"""
        with self._infer_guard():
            return self._node_vehicle_impl(masked_frame, masks_tensor)

    def _node_vehicle_impl(self, masked_frame: np.ndarray, masks_tensor: torch.Tensor) -> Optional[Tuple]:
        """车辆 track + 按掩码分组（GPU 内完成）→ numpy 数据包；无车辆检出时 boxes 为 None。"""
        results = self.vehicle_model.track(
            masked_frame,
            persist=True,
            save=False,
            verbose=False,
            half=True,
            imgsz=self.TRACK_IMGSZ,
            tracker="bytetrack.yaml",
            device=self.vehicle_device
        )
        # 分组在节点内于 GPU 完成，下游（P）不再触碰 GPU 张量
        boxes_np, grouped_np = None, None
        if results:
            boxes = getattr(results[0], "boxes", None)
            boxes_np = boxes.data.cpu().numpy() if (boxes is not None and boxes.data.numel() > 0) else None
            grouped_np = self._group_by_mask(masks_tensor, results).cpu().numpy()
        return (boxes_np, grouped_np)

    # ============================================================
    # L 节点 worker 线程（车道, GPU cuda:2）—— 调用 _node_lane
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

                # L 节点逻辑在 _node_lane 内（单飞由本 worker 串行消费保证）
                lane_boxes = self._node_lane(masked_frame)

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
    # 消费 Seg 分发来的任务 (seq, masked_frame, masks_tensor)：
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

                # 任务来自 Seg 线程： (seq, masked_frame, masks_tensor)
                frame_id, masked_frame, masks_tensor = task

                # V 节点逻辑在 _node_vehicle 内（单飞由本 worker 串行消费保证）
                boxes_np, grouped_np = self._node_vehicle(masked_frame, masks_tensor)

                # CPU 化边界：存入 (原始框数组, 分组后数组) 数据包，不再传递 Results 对象
                with self.results_cond:
                    self.vehicle_results[frame_id] = (boxes_np, grouped_np)
                    self._prune_results(self.vehicle_results, frame_id)
                    self.results_cond.notify_all()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Vehicle worker error: {e}")

    # ============================================================
    # P 节点入口（后处理, 纯 CPU）—— 对应伪代码 _node_post_process
    # 由流水线 P 线程按帧号顺序调用：空检守卫 → 自行车/行人 → 密度 → 速度 → 拥堵
    # → 异常事件 → ProcessFrameResult。所有跨帧状态都只在这里被串行更新。
    # ============================================================
    def _run_post_stage(self, lane_boxes, veh_pkt,
                        overlay_frame: np.ndarray, masks_np: np.ndarray) -> ProcessFrameResult:
        """统一空检出守卫 + P 阶段（纯 CPU，含跨帧状态）。

        由流水线 P 线程调用；lane_boxes / veh_pkt 为 worker 产出的 numpy 数据包；
        overlay_frame 为分割阶段产出的叠加帧，正常与异常返回都作为画面。
        """
        enable_bike_person = self.config.enable_bike_person
        enable_stop_slow = self.config.enable_stop_slow

        # 统一空检出准则：分割模型用 r.masks is None 判空（见 _segment_road）；
        # 检测/追踪模型判空 = 结果数组行数为 0。
        # 任一模型未检出对象都视为本帧未正常处理完成（abnormal 返回，不进入下游后处理）：
        #   - 车道线模型没追踪到车道框 → 速度计算无车道依据；
        #   - 车辆模型没追踪到车辆框 → 与分割/车道结果矛盾，同样按异常帧处理。
        if lane_boxes is None or lane_boxes.shape[0] == 0:
            result = ProcessFrameResult(frame=overlay_frame.copy())
            result.is_abnormal = True
            return result

        if veh_pkt is None:
            result = ProcessFrameResult(frame=overlay_frame.copy())
            result.is_abnormal = True
            return result

        veh_boxes_np, vehicle_boxes = veh_pkt
        if veh_boxes_np is None or veh_boxes_np.shape[0] == 0:
            result = ProcessFrameResult(frame=overlay_frame.copy())
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

    # ==================== 单帧入口（帧内 DAG 编排，同步阻塞） ====================
    # 依赖图：帧 → [S 分割 cuda:0] → {L 车道 cuda:2 ∥ V 车辆 cuda:1} → [P 后处理 CPU]
    # fork：S 完成后把 L / V 派发到两个线程并行执行（两张不同 GPU）；
    # join：两条支路都完成后才进入 P（P 依赖 L、V 的全部产物）。
    # 与下面的帧间流水线（start_pipeline/submit_frame/get_result）为两种使用方式：
    # 本函数一次调用同步处理一帧并返回结果（等价伪代码 process_pipeline 的阻塞版），
    # 不占用任何流水线队列，适合单图/单帧调试；调用时勿与流水线会话并发（模型单飞约束）。

    def process_frame(self, frame: np.ndarray) -> ProcessFrameResult:
        """单帧同步处理（帧内 DAG 编排：S → L∥V → P）。

        Args:
            frame: 输入帧（BGR, H×W×3, numpy 数组），尺寸不限（内部 resize 到输入尺寸）

        Returns:
            ProcessFrameResult：与 get_result 产出同构；本帧未正常处理时 is_abnormal=True。
        """
        # ---- 第 1 步：S 节点（分割, GPU cuda:0）----
        if frame.shape[0] != self.input_height or frame.shape[1] != self.input_width:
            frame = cv2.resize(frame, (self.input_width, self.input_height))

        masked_frame, overlay_frame, masks_tensor, masks_np = self._segment_road(frame)

        # 分割未检出道路 → 该帧按异常产出（与流水线行为一致，不进入下游 L/V/P）
        if masks_np is None:
            return ProcessFrameResult(frame=frame.copy(), is_abnormal=True)

        # ---- 第 2 步：L、V 节点 ----
        # 本机实测（多 GPU 共享机）：每帧新建线程跑推理 ~300ms/路，而主线程/常驻线程顺序
        # 执行只要 ~30ms（SERIAL_INFERENCE 说明见类常量）。串行模式下直接在调用线程顺序跑
        # L→V；只有并发正常的机器（SERIAL_INFERENCE=False）才走线程 fork 并行。
        lane_box, veh_pkt = {}, {}

        if self.SERIAL_INFERENCE:
            try:
                lane_box["v"] = self._node_lane(masked_frame)
            except Exception as e:
                print(f"process_frame L 节点错误: {e}")
                lane_box["v"] = None
            try:
                veh_pkt["v"] = self._node_vehicle(masked_frame, masks_tensor)
            except Exception as e:
                print(f"process_frame V 节点错误: {e}")
                veh_pkt["v"] = None
        else:
            def _run_l():
                try:
                    lane_box["v"] = self._node_lane(masked_frame)
                except Exception as e:
                    print(f"process_frame L 节点错误: {e}")
                    lane_box["v"] = None

            def _run_v():
                try:
                    veh_pkt["v"] = self._node_vehicle(masked_frame, masks_tensor)
                except Exception as e:
                    print(f"process_frame V 节点错误: {e}")
                    veh_pkt["v"] = None

            t_l = threading.Thread(target=_run_l)
            t_v = threading.Thread(target=_run_v)
            t_l.start()
            t_v.start()
            # ---- join —— 等待两条支路全部完成，再进入 P ----
            t_l.join()
            t_v.join()

        # ---- 第 3 步：P 节点（纯 CPU 后处理；内部含空检守卫）----
        try:
            return self._run_post_stage(
                lane_box["v"], veh_pkt["v"], overlay_frame, masks_np
            )
        except Exception as e:
            print(f"process_frame P 节点错误: {e}")
            return ProcessFrameResult(frame=frame.copy(), is_abnormal=True)

    # ==================== 帧间流水线（异步编排 + 阶段线程执行，多帧重叠） ====================
    # 拓扑：  提交(S队列) → [Seg线程:cuda0] → {Lane队列:cuda2, Vehicle队列:cuda1} → [P线程:CPU]
    # 每个阶段都是“单消费者 FIFO”：天然保序（追踪器跨帧状态安全）且单飞（模型线程安全）；
    # 相邻帧可同时处于不同阶段 → 不同 GPU/CPU 资源重叠使用。
    # 本文件为“纯流水线版”：对外入口就是下面的 start_pipeline/submit_frame/get_result/close_pipeline。
    # 单图/单帧调试请用上面的 process_frame（阻塞式，勿与流水线会话并发调用）。

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
                masked_frame, overlay_frame, masks_tensor, masks_np = self._segment_road(frame)
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
            self.vehicle_queue.put((seq, masked_frame, masks_tensor), timeout=60.0)
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
                        # 本会话全部帧已处理完：通知 close_pipeline，复位帧号，回循环顶。
                        # 注意：不能在这里 while _pipeline_on 空等——close_pipeline 置 False 后
                        # 若新会话已 start（_pipeline_on 又变 True），本线程会永远空转，
                        # 导致下一会话的帧全部卡在 L/V 结果里、P 永不产出（与 process_product 同源 bug）。
                        self._session_finished.set()
                        seq = 0
                        stall_since = None
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
                _, overlay_frame, masks_np = payload
                try:
                    res = self._run_post_stage(lane_boxes, veh_pkt, overlay_frame, masks_np)
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
        """保存异常事件对应帧（一帧一张图，命名不含事件名称），返回输出文件名

        在 P 阶段（_run_post_stage）内同步写盘：异常事件帧属于低频输出，
        直接 cv2.imwrite 即可，无需再维护一个常驻写图线程。
        """
        self.count += 1
        filename = f"{self.count:05d}.jpg"
        absolute_filename = os.path.join(self.output_dir, filename)
        try:
            cv2.imwrite(absolute_filename, frame)
        except Exception as e:
            print(f"图片写入失败 {absolute_filename}: {e}")
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
    # 输入一帧 → 输出 masked_frame / overlay_frame / masks_tensor(GPU,给V) / masks_np(CPU,给P)；
    # 分割为空时返回 (frame, frame, None, None)，由上层按异常帧处理。
    # ============================================================
    def _segment_road(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[torch.Tensor], Optional[np.ndarray]]:
        """分割节点（推理串行锁入口）：详见 _segment_road_impl。"""
        with self._infer_guard():
            return self._segment_road_impl(frame)

    def _segment_road_impl(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[torch.Tensor], Optional[np.ndarray]]:
        """
        道路分割并生成两类输出图像。

        Args:
            frame: 输入帧（BGR，H×W×3，numpy 数组）

        Returns:
            (road_masked_frame, overlay_frame, masks_tensor, masks_tensor_np):
                road_masked_frame : 全遮蔽图像（道路外区域填充固定色），供目标检测模型使用
                overlay_frame     : 半透明叠加图像（全遮蔽帧与原始帧加权融合），供可视化使用
                masks_tensor        : 道路掩码张量（GPU，供 V worker 分组），未检测到道路时为 None
                masks_tensor_np     : masks_tensor 的 CPU 副本（纯 numpy，供密度统计，P 阶段不碰 GPU）

        说明：
            - 掩码提取、全遮蔽帧生成、半透明融合均在 segment_device 上完成，
              避免中间结果在 GPU/CPU 间来回搬运；仅在收尾处一次性产出 CPU 掩码副本。
            - 未检测到道路时回退为原帧，保证调用链不中断（masks_tensor 为 None 表示失败）。
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

        masks_tensor = self._extract_roi(seg_results[0].masks.data)
        if masks_tensor is None:
            return frame, frame, None, None

        # GPU 上同时生成全遮蔽帧与半透明叠加帧，一次前向 + 一次矩阵运算，无 CPU 往返
        with torch.no_grad():
            keep_road = masks_tensor.any(dim=0).unsqueeze(0)      # [1, H, W]
            keep_road_3c = keep_road.expand(3, -1, -1)          # [3, H, W]

            # 遮蔽色（BGR）：与历史实现保持一致 [255, 0, 0]
            mask_color = torch.tensor([255.0, 0.0, 0.0],
                                      device=self.segment_device,
                                      dtype=torch.float32).view(3, 1, 1)

            masked_tensor = torch.where(keep_road_3c, frame_tensor, mask_color)   # 全遮蔽
            overlay_tensor = masked_tensor * 0.3 + frame_tensor * 0.7             # 半透明融合

        masked_frame = masked_tensor.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
        overlay_frame = overlay_tensor.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
        # 收尾处一次性产出 CPU 掩码副本：P 阶段密度统计不再触碰 GPU；GPU 版 masks_tensor 留给 V worker 分组
        masks_np = masks_tensor.cpu().numpy()

        return masked_frame, overlay_frame, masks_tensor, masks_np

    # def reset_video_state(self):
    #     """重置所有跨帧/跨视频状态，使同一实例可安全开始处理一段新视频。
    #     适用于服务端复用同一 TrafficAnalyzer 连续处理多段视频/多次流式会话。
    #     注意：不重置 self.count（事件保存文件名保持全局递增，避免重名覆盖）。"""
    #     with self.results_cond:
    #         self.lane_results.clear()
    #         self.vehicle_results.clear()
    #         if hasattr(self, 'seg_payload'):
    #             self.seg_payload.clear()
    #     # 跨帧追踪/速度状态（车道线、车辆轨迹、位移与缩放因子）
    #     self.past_dis.clear()
    #     self.previous_frame_objects = {}
    #     self.previous_frame_lines = {}
    #     self.past_drone_speed = 10.0
    #     self.past_scaling_factor = 3.0
    #     self.scaling_factor = 3.0

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
        self.stop_event.set()          # 举停止旗：所有线程跑完手头这轮后退出
        for t, to in ((self.segment_thread, 1.0), (self.post_thread, 1.0),
                      (self.lane_thread, 1.0), (self.vehicle_thread, 1.0)):
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


# ============================================================================
# TrafficAnalyzerLite —— 线程池精简版（帧内 DAG + 帧间流水线）
# ----------------------------------------------------------------------------
# 设计目标：可读性优先。对外只暴露 2 个入口，其余“会话 API”全部取消：
#   process_frame(frame)     单帧：帧内 DAG（S → L∥V → P），同步阻塞
#   process_frames(frames)   整段视频/帧流：帧间流水线，同步返回按帧序的结果列表
#       （S(n+2) ∥ L/V(n+1) ∥ P(n) 三段阶梯重叠，每阶段单飞保序、模型永不并发调用）
# 相比 TrafficAnalyzer 删除：start_pipeline / submit_frame / get_result /
#   close_pipeline / reset_video_state 以及常驻线程/哨兵/会话开关 ——
#   一段视频 = 一次调用，会话状态（跨帧追踪/速度缓存）在调用内开始并自动复位。
# 单飞/保序由两个小原语统一表达（_SingleFlightStage / _OrderedStage），
# S/L/V/P 的 DAG 拓扑集中在 4 个 work 函数里，一处即可看懂整条流水线。
#
# 注意：process_frame / process_frames 共用同一批模型与跨帧状态，已用 _entry_lock
#   互斥：并发调用会阻塞等待，等效“一次只跑一个视频/会话”；跨帧状态每次调用前自动复位。
# ============================================================================


class _SingleFlightStage:
    """阶段原语①：单飞 + FIFO 保序（供 S / L / V 使用）。

    同一时刻本阶段至多执行一帧；待处理任务按投递顺序排队，即天然保帧序。
    空闲时不占用线程——有活才向池子借一条线程，干完/干到队空即归还。
    """

    def __init__(self, pool: ThreadPoolExecutor, work):
        self._pool = pool
        self._work = work
        self._q = deque()          # 待处理任务（FIFO）
        self._busy = False         # 本阶段是否正有一帧在跑
        self._lock = threading.Lock()

    def put(self, job):
        """投递任务：立即返回（不阻塞调用者）。"""
        with self._lock:
            self._q.append(job)
            if not self._busy:
                self._busy = True
                self._pool.submit(self._run)

    def _run(self):
        while True:
            with self._lock:
                if not self._q:          # 排空则“下班”，线程归还池子
                    self._busy = False
                    return
                job = self._q.popleft()
            try:
                self._work(job)          # 计算全程不持锁（可耗时）
            except Exception as e:
                # 防御：单帧任务异常不允许卡死本阶段（否则 _busy 恒 True，流水线停摆）。
                # work 函数内部已各自捕获，这里兜底并继续处理后续排队任务。
                print(f"_SingleFlightStage 任务异常: {e}")


class _OrderedStage:
    """阶段原语②：单飞 + 按帧号门控（供 P 使用）。

    P 含跨帧状态（previous_frame_objects / past_dis…），必须严格按 seq 串行更新；
    而 S 失败帧与正常 join 帧可能乱序到达 P，因此不能只靠 FIFO 到达序，
    需按 expected 帧号门控：乱序到达的任务先进 ready 缓冲，轮到才执行。
    """

    def __init__(self, pool: ThreadPoolExecutor, work):
        self._pool = pool
        self._work = work
        self._ready = {}            # seq -> job（乱序缓冲）
        self._expected = 0          # 下一个应执行帧号
        self._busy = False
        self._lock = threading.Lock()

    def put(self, job):
        with self._lock:
            self._ready[job.seq] = job
            self._kick()

    def _kick(self):
        if self._busy or self._expected not in self._ready:
            return
        self._busy = True
        job = self._ready.pop(self._expected)
        self._expected += 1
        self._pool.submit(self._run, job)

    def _run(self, job):
        try:
            self._work(job)
        except Exception as e:
            # 防御：记录异常（否则被线程池的 Future 吞掉后不可见）；
            # finally 仍会复位 _busy 并推进下一帧，阶段不会停摆。
            print(f"_OrderedStage 任务异常: {e}")
        finally:
            with self._lock:
                self._busy = False
                self._kick()             # 跑完这一号，立刻推进下一号


class _FrameJob:
    """一帧在流水线中流转的工单：存 S→L∥V→P 的中间产物与 join 计数。"""

    __slots__ = ("seq", "raw", "on_done", "masked", "overlay",
                 "masks", "masks_np", "lane", "veh", "failed",
                 "joined", "lock")

    def __init__(self, seq: int, frame: np.ndarray, on_done):
        self.seq = seq
        self.raw = frame               # 原帧（S 阶段 resize 后回写）
        self.on_done = on_done         # 完成回调（释放背压坑位）
        self.masked = self.overlay = None
        self.masks = self.masks_np = None   # masks_tensor(GPU→V) / masks_np(CPU→P)
        self.lane = self.veh = None
        self.failed = False            # S 失败标记（跳过 L/V，直接按异常帧产出）
        self.joined = 0                # L/V join 计数
        self.lock = threading.Lock()


class TrafficAnalyzerLite(TrafficAnalyzer):
    """线程池精简版：复用父类全部节点/后处理逻辑，只换“编排层”。

    用法：
        a = TrafficAnalyzerLite(config)
        # ① 单帧（帧内 DAG）
        res = a.process_frame(img)
        # ② 整段视频/帧流（帧间流水线，内部 3 卡 + CPU 并行，阻塞返回按序结果）
        results = a.process_frames(iter_of_frames)
        a.shutdown()
    """

    POOL_WORKERS = 8        # 同时最多 S/L/V/P 各 1 条在忙（=4），留余量
    DEFAULT_WINDOW = 8      # 在途帧上限（背压窗口）≈ 流水线可重叠帧数

    def __init__(self, config: Optional[DetectionConfig] = None):
        # 复用父类 __init__：加载模型/标定/跨帧状态缓存；
        # 但父类末尾会调 _start_workers()，这里被子类重写为空 → 不启动常驻线程/队列会话。
        super().__init__(config)
        self._pool = ThreadPoolExecutor(max_workers=self.POOL_WORKERS)
        # 入口互斥锁：process_frame / process_frames 共用同一批模型对象与跨帧状态，
        # 二者必须互斥执行（跨线程并发会自动阻塞等待），保证模型“永不并发调用”。
        self._entry_lock = threading.RLock()

    # ------------------------------------------------------------------
    # 生命周期钩子
    # ------------------------------------------------------------------
    def _start_workers(self):
        """精简版不启动父类的常驻流水线线程（父类 __init__ 会调用此钩子）。"""
        pass

    def shutdown(self):
        """进程退出：关闭线程池（本类没有任何常驻线程/写图线程）。"""
        self._pool.shutdown(wait=True)

    def _save_abnormal_event(self, frame: np.ndarray) -> str:
        """精简版：异常事件帧直接同步写盘（父类走异步写图队列，本类没有写图线程）。"""
        self.count += 1
        filename = f"{self.count:05d}.jpg"
        path = os.path.join(self.output_dir, filename)
        try:
            cv2.imwrite(path, frame)
        except Exception as e:
            print(f"图片写入失败 {path}: {e}")
        return filename

    def _reset_cross_frame_state(self):
        """复位跨帧追踪/速度缓存（等价父类 reset_video_state 的核心部分）。
        每段视频 / 每次单帧调用前自动执行，换视频无需手动复位。"""
        self.past_dis.clear()
        self.previous_frame_objects = {}
        self.previous_frame_lines = {}
        self.past_drone_speed = 10.0
        self.past_scaling_factor = 3.0
        self.scaling_factor = 3.0

    # ------------------------------------------------------------------
    # 入口①：单帧（帧内 DAG：S → L∥V → P）——复用父类，仅先复位跨帧状态
    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> ProcessFrameResult:
        """处理单张图像（视为独立场景，帧内 DAG 并发：L、V 各占一张 GPU）。"""
        with self._entry_lock:                 # 与 process_frames 互斥（共享模型，不并发）
            self._reset_cross_frame_state()
            return super().process_frame(frame)

    # ------------------------------------------------------------------
    # 入口②：整段视频/帧流（帧间流水线）—— start/submit/get/close 全被此调用吸收
    # ------------------------------------------------------------------
    def process_frames(self, frames, window: Optional[int] = None,
                       on_result=None, reset_state: bool = True) -> List[ProcessFrameResult]:
        """按帧序处理整个视频/帧迭代器，阻塞返回按帧序的 List[ProcessFrameResult]。

        内部把 S/L/V/P 排成阶梯流水线（每阶段单飞保序）：
            S(n+2) 与 L(n+1)/V(n+1) 与 P(n) 三路并行，跨帧状态只在 P 内按帧序更新；
        有界在途窗口提供背压：frames 是惰性生成器时也最多比处理进度领先 window 帧。

        Args:
            frames: 帧迭代器（BGR numpy，尺寸不限）；遇 None 视为输入结束。
            window: 在途帧上限，默认 DEFAULT_WINDOW。
            on_result: 可选回调 on_result(seq, result)。每帧结果产出的瞬间由 P 阶段
                调用（线程池线程内，勿做重活），用于外部测量逐帧完成时刻 / 流式推送；
                默认 None。
            reset_state: 是否在本次调用开头复位跨帧追踪/速度状态。默认 True：
                一段新视频 = 一次干净会话。传 False 表示接续上一次调用的状态
                （模拟同一段连续视频分两次喂入，状态连续流动）。

        线程安全：process_frame 与本方法共用同一批模型对象与跨帧状态，由
        _entry_lock 互斥（跨线程并发调用会阻塞等待），保证模型“永不并发调用”。
        注意：on_result 在 P 阶段线程内触发，回调里不要再调 process_frames /
        process_frame（会与持有 _entry_lock 的主线程互相等待而死锁）。
        """
        with self._entry_lock:                   # 生命周期守卫：与 process_frame 互斥
            return self._process_frames_impl(frames, window, on_result, reset_state)

    def _process_frames_impl(self, frames, window: Optional[int] = None,
                             on_result=None, reset_state: bool = True) -> List[ProcessFrameResult]:
        """process_frames 的加锁主体：真正的流水线编排（仅由上面的入口调用）。"""
        if window is None:
            window = self.DEFAULT_WINDOW
        elif not isinstance(window, int) or window < 1:
            # window=0/负数会让 BoundedSemaphore 永久阻塞 → 显式报错，不做静默兜底
            raise ValueError(f"window 必须为 >=1 的整数，实际为 {window!r}")
        if reset_state:
            self._reset_cross_frame_state()      # 一段新视频 = 一次会话

        # 结果容器：显式按 seq 装配。P 虽为“单飞 + 帧号门控”（天然按序、仅 P 线程写），
        # 这里仍用 dict 收集 + 末尾按序展开，从结构上不依赖 P 的执行顺序，
        # 将来即使 P 改为并行执行，返回序与线程安全也不受影响。
        results_by_seq: Dict[int, ProcessFrameResult] = {}

        # ---- 4 个节点的 work（DAG 拓扑集中在这里：S → fork(L,V) → join → P）----
        def work_S(job: _FrameJob):
            frame = job.raw
            try:
                if frame.shape[0] != self.input_height or frame.shape[1] != self.input_width:
                    frame = cv2.resize(frame, (self.input_width, self.input_height))
                masked, overlay, masks_tensor, masks_np = self._segment_road(frame)
            except Exception as e:
                print(f"Lite S 节点错误 seq={job.seq}: {e}")
                masked = overlay = masks_tensor = masks_np = None

            if masks_np is None:
                # 分割失败/未检出道路：跳过 L/V，直接作为异常包进 P（P 按帧号门控，保输出有序）
                job.failed = True
                job.raw = frame
                p.put(job)
            else:
                job.raw = frame                 # resize 后的原帧（异常兜底/画面上用）
                job.masked, job.overlay = masked, overlay
                job.masks, job.masks_np = masks_tensor, masks_np
                l.put(job)                      # fork：同一工单进入 L、V 两条支路
                v.put(job)

        def work_L(job: _FrameJob):
            try:
                lane = self._node_lane(job.masked)
            except Exception as e:
                print(f"Lite L 节点错误 seq={job.seq}: {e}")
                lane = None                     # 置空交给 P 的空检守卫按异常帧处理
            with job.lock:
                job.lane = lane
                job.joined += 1
                if job.joined == 2:
                    p.put(job)                  # join：L、V 都完成才放行 P（锁保证只放行一次）

        def work_V(job: _FrameJob):
            try:
                veh = self._node_vehicle(job.masked, job.masks)
            except Exception as e:
                print(f"Lite V 节点错误 seq={job.seq}: {e}")
                veh = None
            with job.lock:
                job.veh = veh
                job.joined += 1
                if job.joined == 2:
                    p.put(job)

        def work_P(job: _FrameJob):
            # 本函数保证不抛异常：任何失败路径都产出 is_abnormal=True 的结果并释放坑位
            if job.failed or job.raw is None:
                res = ProcessFrameResult(
                    frame=(job.raw.copy() if job.raw is not None else np.zeros(
                        (self.input_height, self.input_width, 3), dtype=np.uint8)),
                    is_abnormal=True)
            else:
                try:
                    res = self._run_post_stage(
                        job.lane, job.veh, job.overlay, job.masks_np)
                except Exception as e:
                    print(f"Lite P 节点错误 seq={job.seq}: {e}")
                    res = ProcessFrameResult(frame=job.raw.copy(), is_abnormal=True)
            # 记录结果 + 触发回调。try/finally 保证 on_done 必然执行：
            # 若回调抛异常而不释放槽位，主线程 drain 阶段会永久阻塞（死锁）。
            try:
                results_by_seq[job.seq] = res     # 先落结果，回调失败也不丢帧
                if on_result is not None:
                    on_result(job.seq, res)
            except Exception as e:
                print(f"Lite P 阶段 on_result 回调错误 seq={job.seq}: {e}")
            finally:
                job.on_done()                     # 兜底：任何路径都释放背压坑位

        # ---- 装配阶段（同一线程池，空闲阶段不占线程）----
        s = _SingleFlightStage(self._pool, work_S)   # 分割 cuda:0（单飞保序）
        l = _SingleFlightStage(self._pool, work_L)   # 车道 cuda:2
        v = _SingleFlightStage(self._pool, work_V)   # 车辆 cuda:1
        p = _OrderedStage(self._pool, work_P)        # 后处理 CPU（按帧号门控）

        # ---- 投帧（有界背压：最多 window 帧在途，不丢帧、不无界堆积）----
        slot = threading.BoundedSemaphore(window)
        seq = 0
        for frame in frames:
            if frame is None:
                break
            slot.acquire()                        # 坑满则阻塞：源生成器不会跑太快
            s.put(_FrameJob(seq, frame, slot.release))
            seq += 1

        # 等全部处理完：把 window 个名额重新取回 == 每个在途任务都已释放。
        # work_P 已保证 on_done 必然执行，故这里的 acquire 不会永久阻塞。
        for _ in range(window):
            slot.acquire()

        # 不变式校验 + 按 seq 展开为有序列表：每个任务都“先记结果、后放槽位”，
        # keys 必然恰好覆盖 0..seq-1；此校验用于尽早暴露未来的回归。
        if len(results_by_seq) != seq:
            missing = [i for i in range(seq) if i not in results_by_seq]
            raise RuntimeError(f"流水线结果缺失（{len(missing)} 帧）: {missing}")
        return [results_by_seq[i] for i in range(seq)]


