"""
benchmark_mp.py —— 测 TrafficAnalyzerMP（多进程流水线）的帧率，热启动口径。
================================================================================

热启动：TrafficAnalyzerMP.__init__ 已包含一次性进程启动 + 模型加载成本（不计时），
之后先跑 warmup 帧预热（tracker/速度状态热身），再对 test 帧计时。

用法:
    python benchmark_mp.py --warmup 10 --test 30
    python benchmark_mp.py --warmup 20 --test 60 --window 6
"""
import argparse
import time

import cv2

from process_mp import TrafficAnalyzerMP, DetectionConfig


def read_frames(video, n):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video}")
    frames = []
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser(description="TrafficAnalyzerMP 多进程流水线帧率测试（热启动）")
    ap.add_argument("--video", default="test_videos/cypp.mp4")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--test", type=int, default=30)
    ap.add_argument("--window", type=int, default=None)
    args = ap.parse_args()

    total = args.warmup + args.test
    print(f"读取视频前 {total} 帧: {args.video}")
    frames = read_frames(args.video, total)
    if len(frames) <= args.warmup:
        raise RuntimeError(f"视频帧数 {len(frames)} <= 预热帧数 {args.warmup}")
    n_test = min(args.test, len(frames) - args.warmup)
    print(f"实际测试帧数: {n_test}（预热 {args.warmup} 帧）")
    print(f"启动 TrafficAnalyzerMP（含 3 个子进程加载模型，一次性成本，不计时）...")

    t_init0 = time.perf_counter()
    an = TrafficAnalyzerMP(DetectionConfig(enable_bike_person=True, enable_stop_slow=True),
                           window=args.window)
    init_time = time.perf_counter() - t_init0
    print(f"  初始化耗时 {init_time:.2f}s（不计入 FPS）")

    try:
        warm = [cv2.resize(f, (an.input_width, an.input_height))
                for f in frames[:args.warmup]]
        tst = [cv2.resize(f, (an.input_width, an.input_height))
               for f in frames[args.warmup:args.warmup + n_test]]

        # 热启动：warm 预热（模型已加载、tracker 热身），test 接续状态计时
        t0 = time.perf_counter()
        an.process_frames(warm, reset_state=True)
        warmup_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        res = an.process_frames(tst, reset_state=False)
        test_time = time.perf_counter() - t0

        n_abn = sum(1 for r in res if getattr(r, "is_abnormal", False))
        print("=" * 70)
        print(f"[TrafficAnalyzerMP 多进程流水线]")
        print(f"  设备 S/V/L = {an._post.SEGMENT_DEVICE}/{an._post.VEHICLE_DEVICE}/{an._post.LANE_DEVICE}"
              f"（各一个子进程）  window={an.window}")
        print(f"  预热 {len(warm)} 帧 {warmup_time:.2f}s | 测试 {len(res)} 帧 {test_time:.2f}s → "
              f"{len(res) / test_time:.2f} FPS ({test_time / len(res) * 1000:.1f} ms/帧)")
        print(f"  异常帧(is_abnormal): {n_abn}/{len(res)}")
        print("=" * 70)
        print(f"  [对照] 线程串行基线 ~16-20 FPS（50-60ms/帧）；理论 DAG 上限 ~50 FPS")
    finally:
        an.shutdown()


if __name__ == "__main__":
    main()
