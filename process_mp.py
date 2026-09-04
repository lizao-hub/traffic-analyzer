"""
process_mp.py —— 多进程流水线版 TrafficAnalyzerMP
=================================================================

拓扑：
    主进程(CPU): 读帧 → resize → 写共享内存 → 发 seq
                   ├→ [S 进程 cuda:4] 分割 → 写 masked/overlay/masks 到共享内存
                   ├→ [L 进程 cuda:6] 车道 track（ByteTrack 状态留进程内）
                   └→ [V 进程 cuda:5] 车辆 track+分组（ByteTrack 状态留进程内）
    主进程 P 线程: 按 seq 对齐 S/L/V 结果 → 密度/速度/拥堵 → List[ProcessFrameResult]

对外 API：
    # 一次性阻塞版（有限视频/批处理）
    an = TrafficAnalyzerMP(DetectionConfig(...), window=8)
    results = an.process_frames(iter_of_frames, reset_state=True)   # 阻塞，按帧序返回
    res = an.process_frame(img)                                     # 单帧
    # 会话式流式版（无限流/相机/WebSocket 逐帧）
    an.start_pipeline()
    seq = an.submit_frame(frame)          # 提交一帧，满时背压阻塞
    seq, res = an.get_result(timeout)     # 按序取回一帧结果
    an.close_pipeline()
    an.shutdown()

"""
import os
import queue as _queue
import threading
import time
import multiprocessing as _mp
from multiprocessing.shared_memory import SharedMemory

import cv2
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import mp_workers as W


# ============================================================================
# 自包含定义：以下内容从 process.py 原样复制（不再 from process import ...）。
#   · 数据类/枚举：DetectionConfig / AbnormalEventType / AbnormalEvent / ProcessFrameResult
#   · TrafficAnalyzer：CPU 后处理子集（类常量 + P 阶段方法），无模型加载、无 CUDA。
# 注意：父进程在 fork 前不初始化 CUDA；S/L/V 模型推理在 mp_workers 子进程中完成，
#   主进程这里只需要 CPU 后处理逻辑与共享内存调度，故只内嵌 CPU 侧代码。
# ============================================================================
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
    """process.py 中 TrafficAnalyzer 的 CPU 后处理子集（自包含复制）。

    仅含类常量与 P 阶段（含跨帧速度/密度/拥堵）方法，供 _CPUPost 继承复用；
    不含任何 GPU/模型加载逻辑（S/L/V 推理在 mp_workers 子进程里完成）。
    """
    SEGMENT_MODEL_PATH = "./runs/hz/viaduct/weights/best.pt"
    VEHICLE_MODEL_PATH = "./runs/hz/vehicle/weights/best.pt"
    LANE_MODEL_PATH = "./runs/hz/line/weights/best.pt"
    SEGMENT_DEVICE = "cuda:4"
    VEHICLE_DEVICE = "cuda:5"
    LANE_DEVICE = "cuda:6"
    INPUT_WIDTH = 960
    INPUT_HEIGHT = 540
    TRACK_IMGSZ = 640
    SERIAL_INFERENCE = True
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
        self.out_queue = _queue.Queue(maxsize=self.window * 2)  # 会话式 API 的结果队列
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
        # 会话式 API 状态
        self._pipeline_on = False
        self._submitted = 0
        self._consumed = 0
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

    # ------------------------------------------------------------------
    # 会话式 API：流式视频（逐帧 submit / 逐帧 get_result），无限流也可用
    # ------------------------------------------------------------------
    def start_pipeline(self):
        """开始一个流式会话（复位跨帧状态与帧序号）。线程已常驻，这里只复位会话状态。"""
        with self._lock:
            if self._pipeline_on:
                return
            self._post._reset_cross_frame_state()
            self._results = {}
            self._next_seq = 0
            self._total = None
            self._submitted = 0
            self._consumed = 0
            self._on_result = None
            self._session_finished.clear()
            self._slots_sem = threading.BoundedSemaphore(self.window)
            self._pipeline_on = True

    def submit_frame(self, frame):
        """提交一帧进入流水线，返回帧号。流水线满时阻塞（背压，不丢帧）。"""
        if not self._pipeline_on:
            self.start_pipeline()
        seq = self._submitted
        self._submitted += 1
        self._slots_sem.acquire()            # 在途帧 <= window，满则阻塞
        self._write_frame(seq % self.window, frame)
        self.q_to_s.put(seq)
        return seq

    def get_result(self, timeout=1.0):
        """按帧号顺序取回一个 (seq, ProcessFrameResult)；超时返回 (None, None)。"""
        try:
            seq, res = self.out_queue.get(timeout=timeout)
        except _queue.Empty:
            return None, None
        self._consumed += 1
        return seq, res

    def close_pipeline(self):
        """结束流式会话：排空剩余结果，保证下次会话状态干净。"""
        if not self._pipeline_on:
            return
        remaining = self._submitted - self._consumed
        for _ in range(remaining):
            try:
                self.out_queue.get(timeout=60.0)
            except _queue.Empty:
                break
        self._pipeline_on = False
        self._slots_sem = None
        self._submitted = 0
        self._consumed = 0

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
                    self._store_result(seq, ProcessFrameResult(
                        frame=np.zeros((W.H, W.W, W.C), dtype=np.uint8), is_abnormal=True))
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
        self._store_result(seq, res)

    def _store_result(self, seq, res):
        """结果落盘：会话式走 out_queue，批处理走 _results；并触发 on_result。"""
        if self._pipeline_on:
            self.out_queue.put((seq, res))
        else:
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
