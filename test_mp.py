"""
test_mp.py —— 用 TrafficAnalyzerMP（多进程流水线）处理视频，间隔取帧，保存结果图。

用法:
    python test_mp.py
    python test_mp.py --video test_videos/cypp.mp4 --stride 2 --out ./test_mp
"""
import argparse
import os

import cv2

from process_mp import TrafficAnalyzerMP
from process import DetectionConfig


def main():
    ap = argparse.ArgumentParser(description="多进程流水线处理视频并保存结果图")
    ap.add_argument("--video", default="test_videos/cypp.mp4")
    ap.add_argument("--stride", type=int, default=2,
                    help="取帧间隔：2 表示每隔 1 帧处理 1 帧（0,2,4,...）")
    ap.add_argument("--out", default="./test_mp")
    args = ap.parse_args()

    if args.stride < 1:
        raise SystemExit("--stride 必须 >=1")

    os.makedirs(args.out, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {args.video}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    indices = []          # 被处理帧在源视频中的帧号

    def gen_frames():
        """惰性读取源视频，按 stride 间隔取帧（process_frames 有背压，不会把整段视频读进内存）。"""
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % args.stride == 0:
                indices.append(idx)
                yield frame
            idx += 1

    n_sel = (total_frames + args.stride - 1) // args.stride
    print(f"视频: {args.video}（共 {total_frames} 帧），间隔 {args.stride} 帧取 1 帧，"
          f"预计处理 {n_sel} 帧")
    print("启动 TrafficAnalyzerMP（含 3 个子进程加载模型，一次性成本）...")

    an = TrafficAnalyzerMP(DetectionConfig(enable_bike_person=True, enable_stop_slow=True))
    try:
        done = {"n": 0}

        def on_result(seq, res):
            done["n"] += 1
            if done["n"] % 20 == 0 or done["n"] == n_sel:
                print(f"  已处理 {done['n']}/{n_sel}")

        results = an.process_frames(gen_frames(), reset_state=True, on_result=on_result)

        saved = 0
        abnormal = 0
        for i, res in enumerate(results):
            frame = getattr(res, "frame", None)
            if frame is None:
                continue
            if getattr(res, "is_abnormal", False):
                abnormal += 1
            src_idx = indices[i] if i < len(indices) else i
            path = os.path.join(args.out, f"frame_{src_idx:05d}.jpg")
            cv2.imwrite(path, frame)
            saved += 1

        print(f"完成：处理 {len(results)} 帧（源帧号 "
              f"{indices[0] if indices else '-'}~{indices[-1] if indices else '-'}），"
              f"保存 {saved} 张到 {args.out}，异常帧 {abnormal}")
    finally:
        cap.release()
        an.shutdown()


if __name__ == "__main__":
    main()
