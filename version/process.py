import time

import numpy as np
import os
import torch
import torch.nn.functional as F
import threading
import queue
import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans



class Process:
    def __init__(self,device, matrix):
        self.segment_model = YOLO("./runs/segment/segment/weights/best.pt").to(device)
        self.vehicle_model_nt = YOLO('./runs/detect/vehicle/weights/best.pt').to(device)
        self.lane_queue = queue.Queue(maxsize=1)  # 只保留最新任务
        self.vehicle_queue = queue.Queue(maxsize=1)  # 只保留最新任务

        # 结果队列
        self.lane_result_queue = queue.Queue(maxsize=1)
        self.vehicle_result_queue = queue.Queue(maxsize=1)

        # 停止标志
        self.stop_event = threading.Event()

        # 启动工作线程
        self.lane_thread = threading.Thread(target=self.lane_worker)
        self.vehicle_thread = threading.Thread(target=self.vehicle_worker)
        self.lane_thread.daemon = True
        self.vehicle_thread.daemon = True
        self.lane_thread.start()
        self.vehicle_thread.start()
        self.frame_lock = threading.Lock()

        # self.cars_model = cars_model
        self.accident_model = YOLO('./runs/detect/accident/weights/best.pt').to(device)
        # self.line_model = line_model
        self.device = device
        self.matrix= matrix

        self.in_width, self.in_height = 1200, 675
        self.scaling_factor = 3
        self.past_drone_speed = 10
        self.previous_frame_lines = {}

        self.previous_frame_objects = {}

        self.count = 0
        self.output_dir = "./event"
        os.makedirs(self.output_dir, exist_ok=True)

    def lane_worker(self):
        """车道线检测线程工作函数"""
        line_model = YOLO('./runs/detect/line/weights/best.pt').to(self.device)

        while not self.stop_event.is_set():
            try:
                # 从队列获取道路分割结果
                segmented_frame = self.lane_queue.get(timeout=0.5)

                # 处理车道线检测
                line_results = line_model.track(segmented_frame, persist=True, save=False, verbose=False, half=True,
                                                     device=self.device)
                # line_results =1
                # 存储结果
                self.lane_result_queue.put(line_results)

            except queue.Empty:
                continue

    def vehicle_worker(self):
        """车辆检测线程工作函数"""
        vehicle_model = YOLO('./runs/detect/vehicle/weights/best.pt').to(self.device)

        while not self.stop_event.is_set():
            try:
                # 从队列获取道路分割结果
                segmented_frame = self.vehicle_queue.get(timeout=0.5)

                # 处理车辆检测
                vehicle_results = vehicle_model.track(segmented_frame, persist=True, save=False, verbose=False, half=True,
                                 device=self.device) # 目标追踪

                # 存储结果
                self.vehicle_result_queue.put(vehicle_results)

            except queue.Empty:
                continue



    def process_frame(self, frame, width, height, detect_accident=True, calculate_speed=True):
        acc, bike, person, stop_car, slow_car, crowd, filename = [], [], [], [], [], [], None
        frame = cv2.resize(frame, (self.in_width, self.in_height))
        frame_tensor = torch.from_numpy(frame).to(self.device).permute(2, 0, 1).float()
        ########################################### segment
        small_mask = self.in_width * self.in_height / 100
        # start_time = time.time()
        seg_results = self.segment_model.predict(frame, imgsz=960, verbose=False, half=True, device=self.device)
        # print(seg_results[0].masks)
        # print(seg_results[0])
        ########################################### extract_road_roi
        with torch.no_grad():  # 禁用梯度计算
            if seg_results[0].masks.data.size(0) ==0:
                frame = cv2.resize(frame, (width, height))
                return frame, acc, bike, person, stop_car, slow_car, crowd, filename
            masks = seg_results[0].masks.data
            areas = masks.view(masks.size(0), -1).sum(dim=1)
            filtered_masks = masks[areas > small_mask]
            # print(filtered_masks.size())
            if filtered_masks.size(0) == 0:
                frame = cv2.resize(frame, (width, height))
                return frame, acc, bike, person, stop_car, slow_car, crowd, filename
            masks_tensor = F.interpolate(
                filtered_masks.unsqueeze(1),
                size=(self.in_height, self.in_width),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)

            combined_mask_tensor = masks_tensor.any(dim=0)
            color_tensor = torch.tensor([255.0, 0.0, 0.0],
                                        device=self.device,
                                        dtype=torch.float32).view(1, 1, 3)
            mask_expanded = combined_mask_tensor.unsqueeze(0).unsqueeze(-1)
            mask_expanded = mask_expanded.expand(-1, -1, -1, 3).permute(3, 1, 2, 0).squeeze(-1)
            segmented_frame_tensor = torch.where(mask_expanded,
                                         frame_tensor,
                                         color_tensor.permute(2, 0, 1))
            segmented_frame = segmented_frame_tensor.permute(1, 2, 0).byte().cpu().numpy()

        ########################################### draw_masks
        frame = cv2.addWeighted(segmented_frame, 0.3, frame, 0.7, 0)
        ########################################### detect_accident
        if detect_accident:
            acc_results = self.accident_model(segmented_frame, save=False, verbose=False, half=True, device=self.device)
            if acc_results[0].boxes.data.size(0) !=0:
                frame = acc_results[0].plot(img=frame, labels=False)
                acc = acc_results[0].boxes.xyxy.cpu().numpy()

        if calculate_speed:
            self.clear_queues()
            ########################################### 将分割结果放入两个队列供并行处理
            self.lane_queue.put(segmented_frame.copy())
            self.vehicle_queue.put(segmented_frame.copy())

            ########################################### 获取并处理结果
            lane_results = self.lane_result_queue.get()
            vehicle_results = self.vehicle_result_queue.get()

            frame = lane_results[0].plot(img=frame, labels=False)
            if vehicle_results[0].boxes.data.size(0) !=0:
                bike, person = self.bike_preson_processing(frame, vehicle_results[0].boxes)
                vehicle_results_boxes = self.group_boxes(masks_tensor, vehicle_results[0].boxes.data)
                frame, mask_density_info = self.density_post_processing(frame, masks_tensor, vehicle_results_boxes)
                frame, mask_speed_info, stop_car, slow_car= self.speed_post_processing(frame, lane_results[0].boxes.data, vehicle_results_boxes)
                frame, info, crowd = self.evaluate(frame, mask_density_info, mask_speed_info, vehicle_results_boxes)
                # print(info)
        else:
            vehicle_results = self.vehicle_model_nt.predict(segmented_frame, save=False, verbose=False, half=True, device=self.device)  # 目标追踪
            if vehicle_results[0].boxes.data.size(0) != 0:
                frame = vehicle_results[0].plot(img=frame, labels=False)
                vehicle_results_boxes = self.group_boxes(masks_tensor, vehicle_results[0].boxes.data)
                frame, mask_density_info = self.density_post_processing(frame, masks_tensor, vehicle_results_boxes)

        if len(acc)>0 or len(bike)>0 or len(person)>0 or len(stop_car)>0 or len(slow_car)>0 or len(crowd)>0:
            self.count += 1
            absolute_filename = os.path.join(self.output_dir, f"{self.count:04d}.jpg")
            frame = cv2.resize(frame, (width, height))
            cv2.imwrite(absolute_filename, frame)
            filename = f"{self.count:04d}.jpg"
            return frame, acc, bike, person, stop_car, slow_car, crowd, filename
        frame = cv2.resize(frame, (width, height))
        return frame, acc, bike, person, stop_car, slow_car, crowd, filename

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

        while not self.lane_result_queue.empty():
            try:
                self.lane_result_queue.get_nowait()
            except queue.Empty:
                break

        while not self.vehicle_result_queue.empty():
            try:
                self.vehicle_result_queue.get_nowait()
            except queue.Empty:
                break

    def speed_post_processing(self, frame, line_results, vehicle_results):
        stopped_car, slow_car = [], []
        if line_results.size(0) ==0 or  line_results.shape[1] == 6: # no id column
            self.scaling_factor /= 2
            drone_speed = self.past_drone_speed
        else:
            current_frame_lines, _ = self.get_current_frame_objects(line_results.cpu().numpy(), [0], False)
            drone_speed = self.calculate_drone_speed(frame, current_frame_lines)
            self.past_drone_speed = drone_speed

        masks_speed_info = {}
        current_frame_objects, scaling_factor = self.get_current_frame_objects(vehicle_results.cpu().numpy())
        if scaling_factor is not None:
            self.scaling_factor = scaling_factor
            track_id2v, stopped_car, slow_car = self.calculate_cars_speed(current_frame_objects, frame, drone_speed)  # 绘图函数

            if len(track_id2v) > 0:
                # 提取所有track_id和对应的mask_id
                track_ids = vehicle_results[:, 4]
                mask_ids = vehicle_results[:, -1]

                speed_values = torch.zeros(len(track_ids), device=vehicle_results.device)

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
        else:
            self.scaling_factor /= 2
        return frame, masks_speed_info, stopped_car, slow_car



    def shutdown(self):
        """安全关闭所有线程"""
        self.stop_event.set()
        self.lane_thread.join(timeout=1.0)
        self.vehicle_thread.join(timeout=1.0)



    def get_current_frame_objects(self, data, attention_id=None, get_scaling_factor=True, marge=25):
        # 如果没有检测到目标，直接返回空字典


        if attention_id is None:
            attention_id = [2]
        current_frame_objects = {}
        scaling_factor = None

        # 如果没有检测到目标，直接返回空字典
        if len(data) == 0:
            return current_frame_objects, scaling_factor

        # 准备所有角点的数组 (批量处理更高效)
        points = []  # 存储左上角和右下角点
        valid_rows = []  # 存储对应的行数据

        # 第一遍遍历：收集所有需要处理的点
        for row in data:
            # 提取数据
            x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
            if y1 < marge or y2 > self.in_height - marge:
                continue

            track_id = int(row[4])
            class_id = int(row[6])

            if class_id not in attention_id:
                continue

            # 添加左上角和右下角点
            points.append([x1, y1])  # 左上角
            points.append([x2, y2])  # 右下角

            # 保存相关信息
            valid_rows.append((row, track_id, class_id))

        # 如果没有符合条件的行，直接返回
        if not points:
            return current_frame_objects, scaling_factor

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
        for i, (row, track_id, class_id) in enumerate(valid_rows):
            # 计算原始中心点（用于显示或调试）
            x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            box_w = abs(x2 - x1)
            box_h = abs(y2 - y1)


            # 获取变换后的点
            idx = i * 2  # 每个目标对应两个点
            top_left_t = transformed_points[idx, 0]
            bottom_right_t = transformed_points[idx + 1, 0]

            if (top_left_t[0] < 0 or top_left_t[0] > self.in_width or
                    top_left_t[1] < 0 or top_left_t[1] > self.in_height or
                    bottom_right_t[0] < 0 or bottom_right_t[0] > self.in_width or
                    bottom_right_t[1] < 0 or bottom_right_t[1] > self.in_height):
                # 任意一点超出范围，舍弃该目标
                continue

            # 计算变换后的框尺寸
            box_w_t = abs(bottom_right_t[0] - top_left_t[0])
            box_h_t = abs(bottom_right_t[1] - top_left_t[1])

            aspect_ratio = box_h_t / box_w_t if box_w_t > 0 else 0
            if aspect_ratio < 1.05:
                continue

            # 计算变换后的中心点
            center_x_t = (top_left_t[0] + bottom_right_t[0]) / 2
            center_y_t = (top_left_t[1] + bottom_right_t[1]) / 2

            # 计算缩放因子（基于变换后的框大小）
            scaling_factor_lst.append(2.15 / box_w_t * 36)

            # 存入字典
            current_frame_objects[track_id] = (
                int(center_x),
                int(center_y),
                int(center_x_t),
                int(center_y_t),
                int(box_w),
                int(box_h)
            )
        if get_scaling_factor:
            scaling_factor = sum(scaling_factor_lst) / (len(scaling_factor_lst) + 1e-10)
        return current_frame_objects, scaling_factor

    def calculate_cars_speed(self, current_frame_objects, frame, drone_speed):
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
                    v = abs(int(dis * self.scaling_factor) - drone_speed)
                    if v < 3:
                        v = 0
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
                        slow_car.append([int(current_pos[0] - current_pos[4] / 2), int(current_pos[1] - current_pos[5] / 2),
                                            int(current_pos[0] + current_pos[4] / 2), int(current_pos[1] + current_pos[5] / 2)])
                        cv2.rectangle(frame, (
                        int(current_pos[0] - current_pos[4] / 2), int(current_pos[1] - current_pos[5] / 2)),
                                      (int(current_pos[0] + current_pos[4] / 2),
                                       int(current_pos[1] + current_pos[5] / 2)), (0, 0, 255), 1)
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
                # H, W = frame.shape[:2]
                # x_offset = 280
                # y_offset = 50
                # label = f'Total number of cars tracked: {max_tracker_id}'
                # t_size = cv2.getTextSize(label, 0, fontScale=0.45, thickness=1)[0]
                # cv2.rectangle(frame, (W - x_offset, y_offset - t_size[1] - 3), (W - x_offset + t_size[0], y_offset + 3),
                #               (0, 0, 255), -1)
                # cv2.putText(frame, label, (W - x_offset, y_offset),
                #             0, 0.45, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
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
            v_average = 10

        if v_average > 20:
            v_average = 20
        if v_average < 0:
            v_average = 0

        x_offset = 50
        y_offset = 50
        label = f'Drone speed: {int((v_average + self.past_drone_speed) / 2)}'
        t_size = cv2.getTextSize(label, 0, fontScale=0.45, thickness=1)[0]
        cv2.rectangle(frame, (x_offset, y_offset - t_size[1] - 3), (x_offset + t_size[0], y_offset + 3), (0, 255, 0), -1)
        cv2.putText(frame, label, (x_offset, y_offset), 0, 0.45, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        self.previous_frame_lines = current_frame_objects.copy()
        return int((v_average + self.past_drone_speed) / 2)

    def group_boxes(self, masks_tensor, boxes_data_tensor):
        device = masks_tensor.device
        boxes_data_tensor = boxes_data_tensor.to(device)

        masks_bool = masks_tensor > 0.5
        # 获取图像尺寸
        num_masks, H, W = masks_bool.shape
        # 提取检测框左上角坐标 (x1, y1)
        top_left = boxes_data_tensor[:, :2].round().long()  # 四舍五入取整

        # 确保坐标在图像范围内
        x_coords = top_left[:, 0].clamp(0, W - 1)
        y_coords = top_left[:, 1].clamp(0, H - 1)

        # 创建坐标张量 (num_boxes, 2)
        points = torch.stack([y_coords, x_coords], dim=1)

        # 为每个检测点生成掩码索引图 (H, W)
        # 值为-1表示无掩码，≥0表示掩码索引
        index_map = torch.full((H, W), -1, dtype=torch.long, device=device)

        # 遍历每个掩码并更新索引图
        for idx in range(num_masks):
            # 当前掩码区域设置为当前索引
            index_map[masks_bool[idx]] = idx

        # 获取每个点对应的掩码索引
        mask_indices = index_map[points[:, 0], points[:, 1]]

        new_boxes_data = torch.cat([
            boxes_data_tensor,
            mask_indices.unsqueeze(1)
        ], dim=1)

        return new_boxes_data

    def bike_preson_processing(self, frame, boxes_data_tensor):
        bike, person = [], []
        for box in boxes_data_tensor:
            class_id = int(box.cls)  # 类别ID
            if class_id == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 边界框坐标
                label = "bike"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                bike.append([x1, y1, x2, y2])
                # 只保留目标类别
            if class_id == 3:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 边界框坐标
                label = "person"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                person.append([x1, y1, x2, y2])
        return bike, person

    def density_post_processing(self, image, masks_tensor, boxes_data_tensor):
        """
        优化版本：在GPU上加速掩码处理和统计计算

        参数:
            image: 原始图像 (numpy数组, BGR格式)
            masks_np: 分割掩码 (numpy数组, 形状为[num_masks, height, width])
            boxes_data_np: 检测框数据 (numpy数组, 最后一列为掩码索引)

        返回:
            标注后的图像 (numpy数组, BGR格式)
            掩码信息字典
        """
        # 将数据转移到GPU
        device = masks_tensor.device
        height, width = image.shape[:2]
        font_scale = 0.45
        thickness = 1
        car_pixels = 4500

        # ===== GPU加速部分 =====
        # 1. 在GPU上计算每个掩码的质心
        centroids = torch.zeros((masks_tensor.shape[0], 2), dtype=torch.int32, device=device)

        # 创建网格坐标
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing='ij'
        )

        for mask_id in range(masks_tensor.shape[0]):
            mask = masks_tensor[mask_id]
            if mask.any():
                # 计算质心
                total = mask.sum()
                centroid_y = (y_coords * mask).sum() // total
                centroid_x = (x_coords * mask).sum() // total
                centroids[mask_id] = torch.tensor([centroid_y, centroid_x], device=device)

        # 2. 在GPU上统计车辆信息
        mask_info = {}
        total_vehicles = 0

        # 获取有效的掩码ID（忽略-1）
        valid_mask_ids = boxes_data_tensor[boxes_data_tensor[:, -1] >= 0, -1]
        unique_mask_ids = torch.unique(valid_mask_ids).long() if len(valid_mask_ids) > 0 else torch.tensor([],
                                                                                                           device=device)

        for mask_id in unique_mask_ids:
            mask_id_int = mask_id.item()
            mask_boxes = boxes_data_tensor[boxes_data_tensor[:, -1] == mask_id]
            count = mask_boxes.shape[0]
            total_vehicles += count

            mask = masks_tensor[mask_id_int]
            area = mask.sum().item()
            density = car_pixels * count / area if area > 0 else 0
            density = min(density, 0.99)  # 限制最大密度

            mask_info[mask_id_int] = {
                'count': count,
                'density': density,
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
        new_mask_info = {}
        for mask_id, info in mask_info.items():
            count = info['count']
            density = info['density']
            new_mask_info[mask_id] = density
            info_label = f"Area {mask_id + 1} numbers:{count} density:{density:.2f}"

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
        return image, new_mask_info

    def evaluate(self, frame, mask_density_info, mask_speed_info, vehicle_results_boxes):
        crowd = []
        common_keys = set(mask_density_info.keys()) & set(mask_speed_info.keys())
        combined_dict = {key: 1 - mask_density_info[key] + mask_speed_info[key] / 100 for key in common_keys}

        for mask_id, value in combined_dict.items():
            if value < 0.45:
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

