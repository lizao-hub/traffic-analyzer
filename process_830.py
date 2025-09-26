import time

import numpy as np
import os
import torch
import torch.nn.functional as F
import threading
import queue
import cv2
from ultralytics import YOLO
# from PIL import Image, ImageDraw, ImageFont
# from sklearn.cluster import KMeans



class Process:
    def __init__(self):
        self.segment_model = YOLO("./runs/dahua/viaduct/weights/best.pt").to(torch.device('cuda:0'))
        self.lane_queue = queue.Queue(maxsize=1)  # 只保留最新任务
        self.vehicle_queue = queue.Queue(maxsize=1)  # 只保留最新任务

        self.lane_result_queue = queue.Queue(maxsize=1)
        self.vehicle_result_queue = queue.Queue(maxsize=1)

        self.stop_event = threading.Event()

        self.lane_thread = threading.Thread(target=self.lane_worker)
        self.vehicle_thread = threading.Thread(target=self.vehicle_worker)
        self.lane_thread.daemon = True
        self.vehicle_thread.daemon = True
        self.lane_thread.start()
        self.vehicle_thread.start()
        self.in_width, self.in_height = 1200, 675

        self.previous_frame_cars = {}
        self.previous_frame_lines = {}

        self.scaling_factor = 3
        self.last_drone_speed = 10
        pts1 = np.float32([(450, 674), (750, 674), (500, 50), (700, 50)])
        pts2 = np.float32([(450, 674), (750, 674), (450, 0), (750, 0)])
        self.matrix = cv2.getPerspectiveTransform(pts1, pts2)


    def lane_worker(self):
        """车道线检测线程工作函数"""
        line_model = YOLO('./runs/dahua/line/weights/best.pt').to(torch.device('cuda:2'))

        while not self.stop_event.is_set():
            try:
                # 从队列获取道路分割结果
                segmented_frame = self.lane_queue.get(timeout=0.5)

                # 处理车道线检测
                line_results = line_model.track(segmented_frame, persist=True, save=False, verbose=False, half=True,
                                                     device=torch.device('cuda:2'), conf=0.5)
                # line_results =1
                # 存储结果
                self.lane_result_queue.put(line_results)

            except queue.Empty:
                continue

    def vehicle_worker(self):
        """车辆检测线程工作函数"""
        vehicle_model = YOLO('./runs/dahua/vehicle/weights/best.pt').to(torch.device('cuda:1'))
        while not self.stop_event.is_set():
            try:
                # 从队列获取道路分割结果
                segmented_frame = self.vehicle_queue.get(timeout=0.5)
                # 处理车辆检测
                vehicle_results = vehicle_model.track(segmented_frame, persist=True, save=False, verbose=False, half=True, tracker="bytetrack.yaml",
                                 device=torch.device('cuda:1')) # 目标追踪
                # 存储结果
                self.vehicle_result_queue.put(vehicle_results)

            except queue.Empty:
                continue



    def process_frame(self, frame, width, height):
        frame = cv2.resize(frame, (self.in_width, self.in_height))
        frame_tensor = torch.from_numpy(frame).to(torch.device('cuda:0')).permute(2, 0, 1).float()
        ########################################### segment
        small_mask = self.in_width * self.in_height / 100
        seg_results = self.segment_model.predict(frame, imgsz=960, verbose=False, half=True, device=torch.device('cuda:0'))
        if seg_results[0].masks is None:
            frame = cv2.resize(frame, (width, height))
            return frame
        ########################################### extract_road_roi
        with torch.no_grad():  # 禁用梯度计算
            masks = seg_results[0].masks.data
            areas = masks.view(masks.size(0), -1).sum(dim=1)
            filtered_masks = masks[areas > small_mask]
            if filtered_masks.size(0) == 0:
                frame = cv2.resize(frame, (width, height))
                return frame
            masks_tensor = F.interpolate(filtered_masks.unsqueeze(1), size=(self.in_height, self.in_width), mode='bilinear', align_corners=False).squeeze(1)
            combined_mask_tensor = masks_tensor.any(dim=0)
            color_tensor = torch.tensor([255.0, 0.0, 0.0], device=torch.device('cuda:0'), dtype=torch.float32).view(1, 1, 3)
            mask_expanded = combined_mask_tensor.unsqueeze(0).unsqueeze(-1)
            mask_expanded = mask_expanded.expand(-1, -1, -1, 3).permute(3, 1, 2, 0).squeeze(-1)
            segmented_frame_tensor = torch.where(mask_expanded, frame_tensor, color_tensor.permute(2, 0, 1))
            segmented_frame = segmented_frame_tensor.permute(1, 2, 0).byte().cpu().numpy() # cpu

        ########################################### draw_masks
        frame = cv2.addWeighted(segmented_frame, 0.3, frame, 0.7, 0) # cpu

        ########################################### detect_accident

        self.clear_queues()
        ########################################### 将分割结果放入两个队列供并行处理
        self.lane_queue.put(segmented_frame.copy())
        self.vehicle_queue.put(segmented_frame.copy())

        ########################################### 获取并处理结果
        lane_results = self.lane_result_queue.get()
        vehicle_results = self.vehicle_result_queue.get()
        # print("vehicle_results:", vehicle_results[0])
        # print("vehicle_results:", vehicle_results[0].boxes.data)
        if vehicle_results[0].boxes.data.size(0) != 0:
            vehicle_results_boxes = self.group_boxes(masks_tensor, vehicle_results[0].boxes.data)
            frame = self.density_post_processing(frame, masks_tensor, vehicle_results_boxes)




            current_frame_lines = self.get_current_frame_objects(lane_results[0].boxes.data.cpu().numpy(), False)
            current_frame_cars = self.get_current_frame_objects(vehicle_results[0].boxes.data.cpu().numpy())
            frame = self.speed_post_processing(frame, current_frame_lines, current_frame_cars)


        # print("current_frame_cars:", current_frame_cars)
        # frame = vehicle_results[0].plot(img=frame, labels=False)
        # frame = lane_results[0].plot(img=frame, labels=False)
        # print("vehicle_results:", vehicle_results[0].boxes.data.device)
        # print("lane_results:", lane_results[0].boxes.data.device)
        return frame

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



    def shutdown(self):
        """安全关闭所有线程"""
        self.stop_event.set()
        self.lane_thread.join(timeout=1.0)
        self.vehicle_thread.join(timeout=1.0)

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

    def density_post_processing(self, image, masks_tensor, boxes_data_tensor):
        """
        优化版本：在GPU上加速掩码处理和统计计算

        参数:
            image: 原始图像
            masks: 分割掩码 (形状为[num_masks, height, width])
            boxes_data: 检测框数据 (最后一列为掩码索引)

        返回:
            标注后的图像 (numpy数组, BGR格式)
            掩码信息字典
        """
        # 将数据转移到GPU
        device = masks_tensor.device
        height, width = image.shape[:2]
        font_scale = 0.45
        thickness = 1
        car_pixels = 26200 / self.scaling_factor / self.scaling_factor


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
                'density': density
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
        for mask_id, info in mask_info.items():
            count = info['count']
            density = info['density']
            # new_mask_info[mask_id] = density
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
        return image

    def speed_post_processing(self, frame, current_frame_lines, current_frame_cars):
        drone_speed = self.calculate_drone_speed(frame, current_frame_lines)
        frame = self.calculate_cars_speed(frame, current_frame_cars, drone_speed)
        return frame

    def get_current_frame_objects(self, data, scale=True):
        marge = 25
        current_frame_objects = {}
        if data.shape[1] != 7:
            return  current_frame_objects
        valid_ids = []
        valid_points = []  # 存储对应的行数据
        scaling_factor_lst = []
        # 第一遍遍历：收集所有需要处理的点
        for row in data:
            # 提取数据
            x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
            if y1 < marge or y2 > self.in_height - marge:
                continue

            track_id = int(row[4])
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            valid_ids.append(track_id)
            valid_points.append((center_x, center_y))
            # 保存cars的宽度
            if int(row[6]) == 2:
                scaling_factor_lst.append(2.75 / (x2 - x1) * 36)
        scaling_factor = sum(scaling_factor_lst) / (len(scaling_factor_lst) + 1e-10)

        if valid_points:
            points_array = np.array([valid_points], dtype=np.float32)
            transformed_points = cv2.perspectiveTransform(points_array, self.matrix)
            # transformed_points.shape = (1, N, 2) → 我们取 [0] 得到 (N, 2)
            transformed_points = transformed_points[0]  # shape: (N, 2)

            # 构建结果字典
            for i, track_id in enumerate(valid_ids):
                new_x, new_y = transformed_points[i]
                orig_x, orig_y = valid_points[i]
                current_frame_objects[track_id] = (
                    int(new_x),
                    int(new_y),
                    int(orig_x),
                    int(orig_y),
                    scaling_factor
                )
        # for track_id, center_x, center_y in valid_rows:
        #     current_frame_objects[track_id] = (
        #         int(center_x),
        #         int(center_y),
        #         scaling_factor
        # )
        if scale:
            if scaling_factor != 0:
                self.scaling_factor = scaling_factor
        return current_frame_objects

    def calculate_drone_speed(self, frame, current_frame_lines):
        v_lst = []
        if self.previous_frame_lines:
            for track_id, current_pos in current_frame_lines.items():
                if track_id in self.previous_frame_lines:
                    prev_pos = self.previous_frame_lines[track_id]
                    dy = current_pos[1] - prev_pos[1]
                    v = dy * self.scaling_factor
                    v_lst.append(v)
        if not v_lst:
            v_average = self.last_drone_speed
        else:
            v_average = sum(v_lst) / (len(v_lst) + 1e-10)

        x_offset = 50
        y_offset = 50
        label = f'Drone speed: {int(v_average)}'
        t_size = cv2.getTextSize(label, 0, fontScale=0.45, thickness=1)[0]
        cv2.rectangle(frame, (x_offset, y_offset - t_size[1] - 3), (x_offset + t_size[0], y_offset + 3), (0, 255, 0), -1)
        cv2.putText(frame, label, (x_offset, y_offset), 0, 0.45, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        self.previous_frame_lines = current_frame_lines.copy()
        self.last_drone_speed = v_average
        return int(v_average)

    def calculate_cars_speed(self, frame, current_frame_cars, drone_speed):
        # 如果有前一帧的数据，则计算位移
        if self.previous_frame_cars:
            for track_id, current_pos in current_frame_cars.items():
                if track_id in self.previous_frame_cars:
                    prev_pos = self.previous_frame_cars[track_id]
                    dx = current_pos[0] - prev_pos[0]
                    dy = current_pos[1] - prev_pos[1]
                    dis = abs(dy + 1e-5) / (dy + 1e-5) * np.sqrt(dx * dx + dy * dy)
                    # dis = abs(dy + 1e-5) / (dy + 1e-5) * dy
                    v = abs(dis * self.scaling_factor - drone_speed)
                    v = int((-0.006 * v + 1.49) * v)
                    if v > 10:
                        label = f'v={v}'
                    else:
                        label = f'v={0}'
                    # print('dy',dy,'v',v)
                    t_size = cv2.getTextSize(label, 0, fontScale=0.35, thickness=1)[0]
                    cv2.rectangle(frame, (int(current_pos[2] - t_size[0] / 2), current_pos[3] - t_size[1] - 3),
                                  (int(current_pos[2] + t_size[0] / 2), current_pos[3] + 3), (0, 255, 0), -1)
                    cv2.putText(frame, label, (int(current_pos[2] - t_size[0] / 2), current_pos[3]),
                                0, 0.35, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        self.previous_frame_cars = current_frame_cars.copy()
        return frame