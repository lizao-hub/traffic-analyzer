"""
mp_workers.py —— process_mp.py 的 GPU worker 进程（每进程钉一张卡）。

用 fork 启动（Linux 默认）：父进程在 fork 前不初始化 CUDA（P 为纯 CPU），
worker 内先把 CUDA_VISIBLE_DEVICES 钉到指定卡，再用 cuda:0，
这样连 YOLO 加载模型时也不会误触默认的 cuda:0（避免踩到邻居的卡）。
"""
import os
import time
import queue as _queue

import numpy as np

# 输入分辨率 / 掩码上限（与 process.py 的 INPUT_WIDTH/HEIGHT 一致）
H, W, C = 540, 960, 3
MAX_MASKS = 8
FRAME_BYTES = H * W * C
MASKS_BYTES = MAX_MASKS * H * W
IMGSZ_SEG = 960
IMGSZ_TRACK = 640


def _attach(shm_names):
    from multiprocessing.shared_memory import SharedMemory
    return [SharedMemory(name=n) for n in shm_names]


def _view(shm, shape, dtype):
    return np.ndarray(shape, dtype=dtype, buffer=shm.buf)


def _extract_roi(masks, device):
    """复刻 process.py 的 _extract_roi（按面积过滤 + bilinear 上采样到 HxW）。"""
    import torch
    import torch.nn.functional as F
    small_mask = H * W / 100
    with torch.no_grad():
        areas = masks.view(masks.size(0), -1).sum(dim=1)
        filtered = masks[areas > small_mask]
        if filtered.size(0) == 0:
            return None
        return F.interpolate(filtered.unsqueeze(1), size=(H, W),
                             mode="bilinear", align_corners=False).squeeze(1)


def run_segment(model, frame_np, device):
    """复刻 process.py 的 _segment_road_impl。返回 (masked, overlay, masks_u8|None)。"""
    import torch
    frame_tensor = torch.from_numpy(np.ascontiguousarray(frame_np)).to(device).permute(2, 0, 1).float()

    seg_results = model.predict(frame_np, imgsz=IMGSZ_SEG, verbose=False, half=True, device=device)
    if not seg_results or seg_results[0].masks is None:
        return frame_np, frame_np, None

    masks_tensor = _extract_roi(seg_results[0].masks.data, device)
    if masks_tensor is None:
        return frame_np, frame_np, None

    with torch.no_grad():
        keep_road = masks_tensor.any(dim=0).unsqueeze(0)
        keep_road_3c = keep_road.expand(3, -1, -1)
        mask_color = torch.tensor([255.0, 0.0, 0.0], device=device, dtype=torch.float32).view(3, 1, 1)
        masked_t = torch.where(keep_road_3c, frame_tensor, mask_color)
        overlay_t = masked_t * 0.3 + frame_tensor * 0.7

    masked = masked_t.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
    overlay = overlay_t.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
    masks_u8 = (masks_tensor > 0.5).byte().cpu().numpy()
    if masks_u8.shape[0] > MAX_MASKS:
        masks_u8 = masks_u8[:MAX_MASKS]
    return masked, overlay, masks_u8


def group_by_mask_cpu(masks_u8, boxes_np):
    """CPU 版按道路掩码分组（等价 process.py 的 GPU _group_by_mask）。

    masks_u8: [N, H, W] uint8；boxes_np: [M, >=4]，列 0/1 为 x1,y1。
    返回 [M, cols+1]，末列为掩码索引（无掩码 -1）。
    """
    M = boxes_np.shape[0]
    if M == 0:
        return np.empty((0, boxes_np.shape[1] + 1), dtype=np.float32)
    xs = boxes_np[:, 0].round().astype(np.int64).clip(0, W - 1)
    ys = boxes_np[:, 1].round().astype(np.int64).clip(0, H - 1)
    vals = masks_u8[:, ys, xs]                 # [N, M] uint8
    has = vals.any(axis=0)
    idx = vals.argmax(axis=0).astype(np.float32)
    idx[~has] = -1.0
    return np.concatenate([boxes_np.astype(np.float32), idx[:, None]], axis=1)


# ----------------------------------------------------------------------
# 三个常驻 worker 进程
# ----------------------------------------------------------------------
def _pin_and_get_device(gpu):
    """把当前进程钉到物理卡 gpu，返回其上的 cuda:0 设备。
    必须在任何 CUDA 初始化（含 YOLO 加载）之前调用。"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError(
            f"worker 启动时 CUDA 已被初始化（父进程 fork 前触了 CUDA），无法钉卡到 {gpu}")
    return torch.device("cuda:0")


def seg_worker(gpu, model_path, window, frame_names, masked_names, overlay_names,
               masks_names, q_to_s, q_s_to_l, q_s_to_v, q_s_to_p, stop_ev):
    from ultralytics import YOLO
    device = _pin_and_get_device(gpu)
    model = YOLO(model_path)
    model.to(device)

    frames = _attach(frame_names)
    maskeds = _attach(masked_names)
    overlays = _attach(overlay_names)
    masks_all = _attach(masks_names)
    try:
        while not stop_ev.is_set():
            try:
                seq = q_to_s.get(timeout=0.5)
            except _queue.Empty:
                continue
            if seq is None:
                continue
            slot = seq % window
            frame = _view(frames[slot], (H, W, C), np.uint8)
            masked, overlay, masks_u8 = run_segment(model, frame, device)

            if masks_u8 is None:
                q_s_to_l.put((seq, False, 0))
                q_s_to_v.put((seq, False, 0))
                q_s_to_p.put((seq, False, 0))
            else:
                n = int(masks_u8.shape[0])
                _view(maskeds[slot], (H, W, C), np.uint8)[:] = masked
                _view(overlays[slot], (H, W, C), np.uint8)[:] = overlay
                buf = _view(masks_all[slot], (MAX_MASKS, H, W), np.uint8)
                buf[:] = 0
                buf[:n] = masks_u8
                q_s_to_l.put((seq, True, n))
                q_s_to_v.put((seq, True, n))
                q_s_to_p.put((seq, True, n))
    except Exception as e:
        print(f"[seg_worker] 异常: {e!r}")
    finally:
        for s in frames + maskeds + overlays + masks_all:
            s.close()


def lane_worker(gpu, model_path, window, masked_names, q_s_to_l, q_l_to_p, stop_ev):
    from ultralytics import YOLO
    device = _pin_and_get_device(gpu)
    model = YOLO(model_path)
    model.to(device)

    maskeds = _attach(masked_names)
    try:
        while not stop_ev.is_set():
            try:
                seq, seg_ok, n = q_s_to_l.get(timeout=0.5)
            except _queue.Empty:
                continue
            if not seg_ok:
                q_l_to_p.put((seq, None))
                continue

            masked = _view(maskeds[seq % window], (H, W, C), np.uint8)
            results = model.track(masked, persist=True, save=False, verbose=False,
                                  half=True, imgsz=IMGSZ_TRACK, tracker="bytetrack.yaml",
                                  device=device)
            lane_boxes = None
            if results:
                boxes = getattr(results[0], "boxes", None)
                if boxes is not None and boxes.data.numel() > 0:
                    lane_boxes = boxes.data.cpu().numpy()
            q_l_to_p.put((seq, lane_boxes))
    except Exception as e:
        print(f"[lane_worker] 异常: {e!r}")
    finally:
        for s in maskeds:
            s.close()


def veh_worker(gpu, model_path, window, masked_names, masks_names,
               q_s_to_v, q_v_to_p, stop_ev):
    from ultralytics import YOLO
    device = _pin_and_get_device(gpu)
    model = YOLO(model_path)
    model.to(device)

    maskeds = _attach(masked_names)
    masks_all = _attach(masks_names)
    try:
        while not stop_ev.is_set():
            try:
                seq, seg_ok, n = q_s_to_v.get(timeout=0.5)
            except _queue.Empty:
                continue
            if not seg_ok:
                q_v_to_p.put((seq, None))
                continue

            masked = _view(maskeds[seq % window], (H, W, C), np.uint8)
            results = model.track(masked, persist=True, save=False, verbose=False,
                                  half=True, imgsz=IMGSZ_TRACK, tracker="bytetrack.yaml",
                                  device=device)
            veh_pkt = None
            if results:
                boxes = getattr(results[0], "boxes", None)
                boxes_np = None
                if boxes is not None and boxes.data.numel() > 0:
                    boxes_np = boxes.data.cpu().numpy()
                if boxes_np is not None:
                    masks_u8 = _view(masks_all[seq % window], (n, H, W), np.uint8)
                    grouped = group_by_mask_cpu(masks_u8, boxes_np)
                else:
                    grouped = np.empty((0, 7), dtype=np.float32)
                veh_pkt = (boxes_np, grouped)
            q_v_to_p.put((seq, veh_pkt))
    except Exception as e:
        print(f"[veh_worker] 异常: {e!r}")
    finally:
        for s in maskeds + masks_all:
            s.close()
