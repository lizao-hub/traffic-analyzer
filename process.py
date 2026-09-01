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
    enable_accident: bool = True       # 事故检测


class AbnormalEventType(Enum):
    """异常事件类型"""
    BIKE = "bike"                      # 自行车检测
    PERSON = "person"                  # 行人检测
    STOPPED_VEHICLE = "stopped_vehicle"  # 停止的车辆
    SLOW_VEHICLE = "slow_vehicle"      # 慢速车辆
    ACCIDENT = "accident"              # 事故检测


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
    output_filename: Optional[str] = None                     # 有异常事件时保存的唯一输出文件名
    is_abnormal: bool = False                                 # 异常返回标志：True 表示本帧未正常处理完成（代码或模型处理问题）



class TrafficAnalyzer:
    """多GPU多线程目标检测处理器"""

    # 部署参数（写死，客户端无需关心）
    SEGMENT_MODEL_PATH = "./runs/hz/viaduct/weights/best.pt"
    VEHICLE_MODEL_PATH = "./runs/hz/vehicle/weights/best.pt"
    LANE_MODEL_PATH = "./runs/hz/line/weights/best.pt"
    ACCIDENT_MODEL_PATH = "./runs/hz/accident/weights/best.pt"

    SEGMENT_DEVICE = "cuda:0"
    VEHICLE_DEVICE = "cuda:1"
    LANE_DEVICE = "cuda:2"
    ACCIDENT_DEVICE = "cuda:3"

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

        # 初始化设备
        self.segment_device = torch.device(self.SEGMENT_DEVICE)
        self.vehicle_device = torch.device(self.VEHICLE_DEVICE)
        self.lane_device = torch.device(self.LANE_DEVICE)
        # self.accident_device = torch.device(self.ACCIDENT_DEVICE)

        # 加载模型（统一在 __init__ 一次性加载，worker 只取用，避免重复加载浪费显存）
        self.segment_model = YOLO(self.SEGMENT_MODEL_PATH).to(self.segment_device)
        self.vehicle_model = YOLO(self.VEHICLE_MODEL_PATH).to(self.vehicle_device)
        self.lane_model = YOLO(self.LANE_MODEL_PATH).to(self.lane_device)
        # self.accident_model = YOLO(self.ACCIDENT_MODEL_PATH).to(self.accident_device)

        # 线程通信队列（输入队列使用双缓冲避免数据丢失）
        self.lane_queue = queue.Queue(maxsize=2)
        self.vehicle_queue = queue.Queue(maxsize=2)
        # self.accident_queue = queue.Queue(maxsize=2)

        # 结果存储：按 frame_id 对齐各模型结果（解决多线程结果错帧问题）
        self.lane_results = {}
        self.vehicle_results = {}
        # self.accident_results = {}
        self.results_lock = threading.Lock()
        self.results_cond = threading.Condition(self.results_lock)

        # 异步图片保存队列
        self.save_queue = queue.Queue(maxsize=30)

        # 控制标志
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()

        # 帧计数器（整型递增，替代 time.time() 作为帧 ID）
        self._frame_counter = 0

        # 启动工作线程
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
        """启动工作线程"""
        self.lane_thread = threading.Thread(target=self._lane_worker, name="LaneWorker", daemon=True)
        self.vehicle_thread = threading.Thread(target=self._vehicle_worker, name="VehicleWorker", daemon=True)
        # self.accident_thread = threading.Thread(target=self._accident_worker, name="AccidentWorker", daemon=True)
        self.save_thread = threading.Thread(target=self._async_writer_worker, name="SaveWriter", daemon=True)
        self.lane_thread.start()
        self.vehicle_thread.start()
        # self.accident_thread.start()
        self.save_thread.start()

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，确保资源释放"""
        self.shutdown()
        return False

    def _lane_worker(self):
        """车道线检测工作线程"""
        while not self.stop_event.is_set():
            if self.pause_event.is_set():
                time.sleep(0.1)
                continue

            try:
                # 非阻塞获取任务
                task = self.lane_queue.get(timeout=0.5)
                if task is None:
                    continue

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

                # 按 frame_id 存入结果字典并通知等待方
                with self.results_cond:
                    self.lane_results[frame_id] = results
                    self._prune_results(self.lane_results, frame_id)
                    self.results_cond.notify_all()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Lane worker error: {e}")

    def _vehicle_worker(self):
        """车辆检测工作线程"""
        while not self.stop_event.is_set():
            if self.pause_event.is_set():
                time.sleep(0.1)
                continue

            try:
                # 非阻塞获取任务
                task = self.vehicle_queue.get(timeout=0.5)
                if task is None:
                    continue

                frame_id, masked_frame = task

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

                # 按 frame_id 存入结果字典并通知等待方
                with self.results_cond:
                    self.vehicle_results[frame_id] = results
                    self._prune_results(self.vehicle_results, frame_id)
                    self.results_cond.notify_all()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Vehicle worker error: {e}")

    # def _accident_worker(self):
    #     """事故检测工作线程"""
    #     while not self.stop_event.is_set():
    #         if self.pause_event.is_set():
    #             time.sleep(0.1)
    #             continue
    #
    #         try:
    #             # 非阻塞获取任务
    #             task = self.accident_queue.get(timeout=0.5)
    #             if task is None:
    #                 continue
    #
    #             frame_id, masked_frame = task
    #
    #             # 执行事故检测
    #             results = self.accident_model.predict(
    #                 masked_frame,
    #                 imgsz=960,
    #                 verbose=False,
    #                 half=True,
    #                 device=self.accident_device
    #             )
    #
    #             # 按 frame_id 存入结果字典并通知等待方
    #             with self.results_cond:
    #                 self.accident_results[frame_id] = results
    #                 self._prune_results(self.accident_results, frame_id)
    #                 self.results_cond.notify_all()
    #
    #         except queue.Empty:
    #             continue
    #         except Exception as e:
    #             print(f"Accident worker error: {e}")

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

    def process_frame(self, frame: np.ndarray) -> ProcessFrameResult:
        # 输入为空：属异常返回（代码/输入问题）
        if frame is None or frame.size == 0:
            result = ProcessFrameResult(frame=frame, is_abnormal=True)
            return result

        # 从配置读取功能开关
        enable_bike_person = self.config.enable_bike_person
        enable_stop_slow = self.config.enable_stop_slow
        # enable_accident = self.config.enable_accident  # 事故检测暂时关闭，后续再改

        # 输入尺寸与类定义尺寸不一致时，先 resize 到统一尺寸
        if frame.shape[0] != self.input_height or frame.shape[1] != self.input_width:
            frame = cv2.resize(frame, (self.input_width, self.input_height))

        result = ProcessFrameResult(frame=frame.copy())


        # 道路分割：返回全遮蔽帧（目标检测用）、半透明叠加帧（可视化用）与道路掩码
        masked_frame, overlay_frame, road_masks = self._segment_road(frame)
        if road_masks is None:
            # 未检测到道路，属异常返回（模型处理问题）
            result.is_abnormal = True
            return result

        # 整型帧 ID（替代 time.time()，用于多模型结果对齐）
        frame_id = self._next_frame_id()

        # 并行提交车道/车辆两个任务到各自 worker（各自 GPU 上并行推理）
        # masked_frame 为只读输入，两个队列复用同一引用，避免重复拷贝
        self._submit_task(self.lane_queue, frame_id, masked_frame)
        self._submit_task(self.vehicle_queue, frame_id, masked_frame)
        # if enable_accident:
        #     self._submit_task(self.accident_queue, frame_id, masked_frame)

        # 按 frame_id 对齐取回各模型结果
        lane_results = self._wait_result(self.lane_results, frame_id)
        vehicle_results = self._wait_result(self.vehicle_results, frame_id)
        # accident_results = self._wait_result(self.accident_results, frame_id) if enable_accident else None

        # worker 超时或异常时 _wait_result 返回 None，直接异常返回。
        if not lane_results or not vehicle_results:
            result.is_abnormal = True
            return result

        # 车道未检出任何 box 时直接返回：速度计算依赖车道框，缺车道则无速度依据。
        # 车辆未检出 box 则放行继续（空道路属正常结果，下游各函数均对空数据有守卫）。
        lane_boxes = getattr(lane_results[0], "boxes", None)
        if lane_boxes is None or lane_boxes.data.numel() == 0:
            result.is_abnormal = True
            return result

        # 各类检测结果（先收集为局部变量，再统一映射为异常事件）
        bike_detections, person_detections = [], []
        stopped_vehicles, slow_vehicles = [], []
        # accident_detections = []
        density_info, speed_info = {}, {}

        if enable_bike_person:
            bike_detections, person_detections = self._process_bike_person(overlay_frame, vehicle_results)

        # # 事故检测（开关控制）
        # if enable_accident and accident_results is not None:
        #     accident_detections = self._process_accident(overlay_frame, accident_results[0].boxes)

        # 车辆检测（速度/密度/拥堵为基本内容，始终执行）
        # if vehicle_results is not None and vehicle_results[0].boxes.data.size(0) != 0:
            # 自行车/行人检测（开关控制）
            # if enable_bike_person:
            #     bike_detections, person_detections = self._process_bike_person(overlay_frame, vehicle_results[0].boxes)

        vehicle_boxes = self._group_by_mask(road_masks, vehicle_results)

        overlay_frame, density_info = self._process_traffic_density(overlay_frame, road_masks, vehicle_boxes)


        # 速度计算（基本内容，始终执行；停止/慢速车辆由开关控制）
        if enable_stop_slow:
            overlay_frame, speed_info, stopped_vehicles, slow_vehicles = self._process_speed(
                overlay_frame, lane_results, vehicle_boxes, enable_stop_slow=enable_stop_slow
            )
        else:
            overlay_frame, speed_info, stopped_vehicles, slow_vehicles = self._process_speed(
                overlay_frame, lane_results, vehicle_boxes, enable_stop_slow=enable_stop_slow
            )

        # 密度分析（基本内容，始终执行）


        # 拥堵评估（基本内容，始终执行，只保留标志位）
        overlay_frame, _, congestion_regions = self._evaluate_congestion(
            overlay_frame, density_info, speed_info, vehicle_boxes
        )
        result.traffic_congestion = bool(congestion_regions)

        # 汇总异常事件：自行车/行人/停止车辆/慢速车辆（事故检测暂时关闭）
        result.abnormal_events = self._collect_abnormal_events(
            bike_detections, person_detections, stopped_vehicles, slow_vehicles
        )
        # 有异常事件则保存一帧图像（一帧可有多个异常事件，但只保存一张，命名不含事件名称）
        if result.abnormal_events:
            result.output_filename = self._save_abnormal_event(overlay_frame)

        result.frame = overlay_frame

        return result


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
        # 事故检测暂时关闭，后续再改
        # if accident_detections:
        #     events.append(AbnormalEvent(event_type=AbnormalEventType.ACCIDENT, detections=accident_detections))
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

    def _submit_task(self, task_queue: queue.Queue, frame_id: int, data: np.ndarray):
        """提交任务到队列（非阻塞）"""
        try:
            task_queue.put((frame_id, data), timeout=0.1)
        except queue.Full:
            # 丢弃最旧的任务
            try:
                task_queue.get_nowait()
            except queue.Empty:
                pass
            task_queue.put((frame_id, data), timeout=0.1)

    def _next_frame_id(self) -> int:
        """返回递增的整型帧 ID"""
        self._frame_counter += 1
        return self._frame_counter

    def _wait_result(self, result_dict, frame_id: int, timeout: float = 1.0):
        """按 frame_id 从结果字典取回指定帧的结果，超时返回 None"""
        with self.results_cond:
            deadline = time.time() + timeout
            while frame_id not in result_dict:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self.results_cond.wait(remaining)
            return result_dict.pop(frame_id)

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

    def _segment_road(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[torch.Tensor]]:
        """
        道路分割并生成两类输出图像。

        Args:
            frame: 输入帧（BGR，H×W×3，numpy 数组）

        Returns:
            (road_masked_frame, overlay_frame, road_masks):
                road_masked_frame : 全遮蔽图像（道路外区域填充固定色），供目标检测模型使用
                overlay_frame     : 半透明叠加图像（全遮蔽帧与原始帧加权融合），供可视化使用
                road_masks        : 道路掩码张量（车辆分组 / 密度统计需要），未检测到道路时为 None

        说明：
            - 掩码提取、全遮蔽帧生成、半透明融合均在 segment_device 上完成，
              避免中间结果在 GPU/CPU 间来回搬运。
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

        # 健壮性：未检出道路掩码时回退为原帧
        if not seg_results or getattr(seg_results[0], "masks", None) is None:
            return frame, frame, None

        road_masks = self._extract_roi(seg_results[0].masks.data)
        if road_masks is None:
            return frame, frame, None

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

        return masked_frame, overlay_frame, road_masks

    def clear_queues(self):
        """清空队列，确保只处理最新帧"""
        while not self.lane_queue.empty():
            try:
                self.lane_queue.get_nowait()
            except queue.Empty:
                break

        while not self.vehicle_queue.empty():
            try:
                self.vehicle_queue.get_nowait()
            except queue.Empty:
                break

        # 事故检测暂时关闭，后续再改
        # while not self.accident_queue.empty():
        #     try:
        #         self.accident_queue.get_nowait()
        #     except queue.Empty:
        #         break

        with self.results_cond:
            self.lane_results.clear()
            self.vehicle_results.clear()
            # self.accident_results.clear()

    def get_scaling_factor(self, data):
        marge = 25
        points = []
        valid_rows = []  # 存储对应的行数据

        # 第一遍遍历：收集所有需要处理的点
        for row in data:
            # 提取数据
            x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
            if y1 < marge or y2 > self.input_height - marge:
                continue

            class_id = int(row[-1])

            if class_id != 2:
                continue

            # 添加左上角和右下角点
            points.append([x1, y1])  # 左上角
            points.append([x2, y2])  # 右下角

            # 保存相关信息
            valid_rows.append(row)

        # 如果没有符合条件的行，直接返回
        if not points:
            return self.past_scaling_factor

        # 转换为正确的NumPy数组格式 (N, 1, 2)
        points_array = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

        # 批量应用透视变换
        try:
            transformed_points = cv2.perspectiveTransform(points_array, self.matrix)
        except Exception as e:
            print(f"透视变换失败: {e}")
            # 如果变换失败，使用原始坐标作为后备
            transformed_points = points_array.copy()

        # 第二遍遍历：处理结果
        scaling_factor_lst = []
        for i, row in enumerate(valid_rows):
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

            aspect_ratio = box_h_t / box_w_t if box_w_t > 0 else 0
            if aspect_ratio < 1.5:
                continue

            center_x_t = (top_left_t[0] + bottom_right_t[0]) / 2
            if self.input_width // 4 < center_x_t < 3 * self.input_width // 4:
                scaling_factor_lst.append(2.25 / box_w_t * 36)
            else:
                scaling_factor_lst.append(2.25 / box_w_t * 36)

        if not scaling_factor_lst:
            return self.past_scaling_factor
        scaling_factor = sum(scaling_factor_lst) / (len(scaling_factor_lst) + 1e-10)
        scaling_factor = 0.2 * scaling_factor + 0.8 * self.past_scaling_factor
        return scaling_factor

    def _process_speed(self, frame: np.ndarray, lane_results: List, vehicle_boxes: torch.Tensor,
                       enable_stop_slow: bool = True) -> Tuple[np.ndarray, Dict, List, List]:
        # 车道结果已在调用前确认有 box 数据（process_frame 中的 lane_boxes 守卫），直接取 boxes.data
        line_results = lane_results[0].boxes.data
        if line_results.size(0) == 0 or line_results.shape[1] == 6: # no id column
            drone_speed = self.past_drone_speed
        else:
            current_frame_lines = self.get_current_frame_objects(line_results.cpu().numpy(), 0,False)
            drone_speed = self.calculate_drone_speed(frame, current_frame_lines)
            self.past_drone_speed = drone_speed

        masks_speed_info = {}
        current_frame_objects = self.get_current_frame_objects(vehicle_boxes.cpu().numpy())
        track_id2v, stopped_car, slow_car = self.calculate_cars_speed(frame, current_frame_objects, drone_speed, enable_stop_slow=enable_stop_slow)  # 绘图函数

        if len(track_id2v) > 0:
            # 提取所有track_id和对应的mask_id
            track_ids = vehicle_boxes[:, 4]
            mask_ids = vehicle_boxes[:, -1]

            speed_values = torch.zeros(len(track_ids), device=vehicle_boxes.device)

            for i, track_id in enumerate(track_ids):
                # 将 track_id 转换为整数（解决 1.0 vs 1 的问题）
                track_id_int = int(track_id.item())

                # 如果 track_id 在字典中，获取速度值
                if track_id_int in track_id2v:
                    speed_values[i] = track_id2v[track_id_int]
                else:
                    # 处理缺失的 track_id（使用默认值或跳过）
                    # 这里使用 -1 表示缺失值，后续计算平均速度时会排除
                    speed_values[i] = -1

            # 计算每个 mask_id 的平均速度
            unique_mask_ids = torch.unique(mask_ids)

            for mask_id in unique_mask_ids:
                if mask_id.item() == -1:
                    continue

                mask = mask_ids == mask_id
                mask_speed = speed_values[mask]

                # 排除无效值（-1）
                valid_speeds = mask_speed[mask_speed >= 0]

                if len(valid_speeds) > 0:
                    avg_speed = valid_speeds.float().mean().item()
                    masks_speed_info[int(mask_id.item())] = avg_speed
                else:
                    # 没有有效速度数据
                    masks_speed_info[int(mask_id.item())] = 100
        return frame, masks_speed_info, stopped_car, slow_car



    def pause(self):
        """暂停处理"""
        self.pause_event.set()

    def resume(self):
        """恢复处理"""
        self.pause_event.clear()

    def shutdown(self):
        """安全关闭所有线程"""
        # 先排空异步写入队列：趁写图线程还活着把已入队任务写完（避免 stop_event 提前终止导致 join() 死锁）
        self.save_queue.join()
        # 放入哨兵，让写图线程立即退出（而非空等 0.5s 超时）
        self.save_queue.put(None)
        # 再置停止标志，让车道/车辆 worker 退出循环
        self.stop_event.set()
        self.lane_thread.join(timeout=1.0)
        self.vehicle_thread.join(timeout=1.0)
        # self.accident_thread.join(timeout=1.0)  # 事故检测暂时关闭，后续再改
        self.save_thread.join(timeout=2.0)

    # ==================== 向后兼容方法 ====================

    def process_frame_compat(self, frame: np.ndarray, width: int, height: int,
                            stop_car_flag: bool = True, slow_car_flag: bool = True,
                            crowd_flag: bool = True, accident_flag: bool = True) -> Tuple[np.ndarray, List, List, List, List, List, List, Optional[str]]:
        """
        向后兼容的 process_frame 接口

        Returns:
            (frame, bike, person, stop_car, slow_car, crowd, accident, filename)
        """
        # 映射旧参数到新功能开关（拥堵/密度现为基本内容，始终启用）；临时写入 config，处理完恢复
        saved_stop_slow = self.config.enable_stop_slow
        saved_accident = self.config.enable_accident
        self.config.enable_stop_slow = stop_car_flag or slow_car_flag
        self.config.enable_accident = accident_flag
        try:
            result = self.process_frame(frame)
        finally:
            self.config.enable_stop_slow = saved_stop_slow
            self.config.enable_accident = saved_accident

        # 从统一的异常事件中还原旧的分类结果（拥堵区域已改为标志位，返回空列表）
        def _detections(event_type: AbnormalEventType) -> List:
            for e in result.abnormal_events:
                if e.event_type == event_type:
                    return e.detections
            return []

        return (
            result.frame,
            _detections(AbnormalEventType.BIKE),
            _detections(AbnormalEventType.PERSON),
            _detections(AbnormalEventType.STOPPED_VEHICLE),
            _detections(AbnormalEventType.SLOW_VEHICLE),
            [],  # 拥堵区域不再返回，仅保留标志位 result.traffic_congestion
            _detections(AbnormalEventType.ACCIDENT),
            result.output_filename
        )



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

    def _process_bike_person(self, frame: np.ndarray, vehicle_results: List) -> Tuple[List, List]:
        """在 frame 上绘制自行车/行人检测框，返回各自框坐标列表。

        Args:
            frame: 可视化用图像（BGR, H×W×3, numpy, CPU 内存），会被原地绘制
            vehicle_results: 车辆模型推理结果（Ultralytics Results 列表，非 Tensor）

        说明：
            先对 boxes.data 做一次批量 .cpu() 取回，再用向量化掩码按类别过滤，
            替代原实现逐框 int(box.cls)/int(box.xyxy) 触发的 N 次 GPU→CPU 同步；
            目标类别固定为 bike=0 / person=3（bus/car/truck 不绘制）。
        """
        bike, person = [], []
        boxes = getattr(vehicle_results[0], "boxes", None)
        if boxes is None or boxes.data.numel() == 0:
            return bike, person

        # [N, 6]: x1,y1,x2,y2,conf,cls —— 单次批量搬运到 CPU，后续无设备同步
        data_cpu = boxes.data.cpu().numpy()
        cls = data_cpu[:, 5]

        # 车辆模型 5 类：0 bike / 1 bus / 2 car / 3 person / 4 truck
        for class_id, targets in ((0, bike), (3, person)):
            label = "bike" if class_id == 0 else "person"
            for x1, y1, x2, y2 in data_cpu[cls == class_id, :4].astype(int):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                targets.append([x1, y1, x2, y2])

        return bike, person

    # 事故检测暂时关闭，后续再改
    # def _process_accident(self, frame: np.ndarray, boxes_data) -> List:
    #     """处理事故检测结果"""
    #     accidents = []
    #     if boxes_data is None or len(boxes_data) == 0:
    #         return accidents
    #
    #     for box in boxes_data:
    #         x1, y1, x2, y2 = map(int, box.xyxy[0])
    #         conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
    #         label = f"accident {conf:.2f}"
    #         # 绘制红色框标记事故
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    #         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    #         accidents.append([x1, y1, x2, y2, conf])
    #     return accidents

    def _process_traffic_density(self, image: np.ndarray, masks_tensor: torch.Tensor, boxes_data_tensor: torch.Tensor) -> Tuple[np.ndarray, Dict]:
        """
        统计各道路掩码区域内的车辆密度并标注到图像上。

        Args:
            image: 待标注图像（BGR, numpy 数组）
            masks_tensor: 道路掩码张量 [num_masks, H, W]，位于 segment_device
            boxes_data_tensor: 车辆分组结果 [N, 7]（x1,y1,x2,y2,conf,cls,掩码索引），位于 GPU；
                可能为空（无车辆框），此时所有掩码密度补 0，保证返回结构完整

        Returns:
            image: 标注后的图像
            density_info: {mask_id: traffic_density}，无有效车辆时所有掩码密度为 0
        """
        # 将数据转移到GPU
        device = masks_tensor.device
        height, width = image.shape[:2]
        font_scale = 0.45
        thickness = 1
        car_pixels = 16200 / self.scaling_factor / self.scaling_factor


        # ===== GPU加速部分 =====
        # 1. 向量化计算每个掩码的质心（消除 Python for 循环）
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing='ij'
        )

        # masks_tensor: [N, H, W] -> weights: [N, H, 1] * [1, H, W] and [N, 1, W] * [1, H, W]
        mask_sums = masks_tensor.sum(dim=(1, 2)).clamp(min=1)  # [N]
        centroid_y = (masks_tensor * y_coords.unsqueeze(0)).sum(dim=(1, 2)) // mask_sums
        centroid_x = (masks_tensor * x_coords.unsqueeze(0)).sum(dim=(1, 2)) // mask_sums
        centroids = torch.stack([centroid_y.long(), centroid_x.long()], dim=1).to(torch.int32)  # [N, 2]

        # 2. 在GPU上统计车辆信息
        area_statistics = {}
        total_vehicles = 0

        if boxes_data_tensor.numel() == 0:
            # 无车辆框：为每个掩码补 0 密度，保证返回的 density_info 结构完整（而非空字典）
            for mask_id in range(masks_tensor.shape[0]):
                area_statistics[mask_id] = {
                    'vehicle_count': 0,
                    'traffic_density': 0.0,
                    'centroid': centroids[mask_id].cpu().numpy()
                }
        else:
            # 获取有效的掩码ID（忽略 -1）
            valid_mask_ids = boxes_data_tensor[boxes_data_tensor[:, -1] >= 0, -1]
            unique_mask_ids = torch.unique(valid_mask_ids).long() if len(valid_mask_ids) > 0 else torch.tensor([],
                                                                                                               device=device)

            for mask_id in unique_mask_ids:
                mask_id_int = mask_id.item()
                mask_boxes = boxes_data_tensor[boxes_data_tensor[:, -1] == mask_id]
                vehicle_count = mask_boxes.shape[0]
                total_vehicles += vehicle_count

                mask = masks_tensor[mask_id_int]
                area_pixels = mask.sum().item()
                traffic_density = car_pixels * vehicle_count / area_pixels if area_pixels > 0 else 0
                traffic_density = min(traffic_density, 0.99)  # 限制最大密度

                area_statistics[mask_id_int] = {
                    'vehicle_count': vehicle_count,
                    'traffic_density': traffic_density,
                    'centroid': centroids[mask_id_int].cpu().numpy()
                }

        # ===== CPU可视化部分 =====
        # 3. 在每个掩码区域上标注掩码ID（使用质心位置）
        centroids_cpu = centroids.cpu().numpy()

        for mask_id in range(masks_tensor.shape[0]):
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

    def _evaluate_congestion(self, frame: np.ndarray, mask_density_info: Dict, mask_speed_info: Dict, vehicle_results_boxes: torch.Tensor) -> Tuple[np.ndarray, Dict, List]:
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

