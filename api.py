"""
WebSocket 实时逐帧分析服务（精简版）
====================================
只保留一个对外接口：WS /ws/analyze —— 前端逐帧推送 JPEG，服务端逐帧回结果。

已移除（简化阅读，实际生产可能缺这些，先不管）：
  - GET  /health          健康检查
  - POST /video/analyze   整段视频上传分析
  - POST /frame/analyze   单帧快照分析
  - asyncio.Lock / busy 会话互斥锁（多路会话并发会互相干扰）
  - CORS 跨域中间件

完整版请见 server_backup.py，需要时还原： cp server_backup.py server.py

运行：
  python server.py --host 0.0.0.0 --port 8000

前端 WebSocket 协议（帧以 base64 文本发送）：
  1) 连接后先发一条文本消息： {"action":"start","send_images":true,"quality":85}
  2) 之后逐帧发文本消息：     {"frame": "<JPEG 图像的 base64>"}   （每帧一条）
  3) 发送完毕发：            {"action":"end"}   （或 {"action":"cancel"} 中止）
  4) 服务端每完成一帧回一条 JSON 文本；end 后回一条 {"type":"done", ...summary}
"""
import argparse
import asyncio
import base64
import json
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from process import DetectionConfig, TrafficAnalyzer


# =========================================================================== #
# 小工具                                                                    #
# =========================================================================== #
class SessionError(Exception):
    """会话级错误（会中止当前分析并清理流水线）"""


class SessionOutcome:
    def __init__(self):
        self.submitted = 0      # 已投递进流水线的帧数
        self.truncated = False  # 因 max_frames 上限被截断（本简化版不使用）
        self.aborted = False    # 客户端中途断开/取消


def _encode_jpg(frame: np.ndarray, quality: int = 85) -> str:
    """numpy(BGR) -> JPEG -> base64 字符串"""
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return base64.b64encode(buf).decode("ascii")


def _decode_jpeg(data: bytes) -> np.ndarray:
    """JPEG 字节 -> numpy(BGR)；失败返回 None"""
    if not data:
        return None
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame if frame is not None else None


def _event_to_dict(event) -> dict:
    return {
        "type": event.event_type.value,
        "detections": [[int(v) for v in det] for det in event.detections],
    }


def _result_to_dict(seq: int, res, want_images: bool, quality: int) -> dict:
    """把一帧 ProcessFrameResult 转成可 JSON 序列化的字典"""
    d = {
        "seq": seq,
        "ok": not res.is_abnormal,
        "is_abnormal": res.is_abnormal,
        "traffic_congestion": bool(res.traffic_congestion),
        "abnormal_events": [_event_to_dict(e) for e in res.abnormal_events],
        "output_filename": res.output_filename,
        "metrics": res.metrics,
    }
    if want_images:
        d["image"] = _encode_jpg(res.frame, quality)
    return d


# =========================================================================== #
# 流水线会话核心                                                            #
# =========================================================================== #
async def _consumer_core(app, outcome: SessionOutcome, completed: asyncio.Event,
                         results: list, emit, want_images: bool, quality: int,
                         progress: dict, stall_timeout: float = 45.0):
    """消费者：按序取回流水线结果 → 序列化（可选图片）→ emit(result_dict)。
    emit 由调用方提供：本服务为发 JSON 文本到 WebSocket。
    progress 由生产者/消费者共同更新（投帧或产出都算“有进展”），
    用于区分“真卡死”与“输入侧暂时无帧”。"""
    analyzer = app.state.analyzer
    got = 0
    while True:
        seq, res = await asyncio.to_thread(analyzer.get_result, 1.0)
        if seq is not None:
            got += 1
            progress["last"] = time.monotonic()
            item = await asyncio.to_thread(
                _result_to_dict, seq, res, want_images, quality)
            await emit(item)
            if completed.is_set() and got >= outcome.submitted:
                break
            continue

        # 超时无结果
        if completed.is_set() and got >= outcome.submitted:
            break  # 正常结束：已全部取回
        if outcome.submitted > got and time.monotonic() - progress["last"] > stall_timeout:
            raise SessionError("流水线长时间无进展（可能某 worker 异常），已中止")
        await asyncio.sleep(0.05)


async def _close_pipeline_safe(app):
    """安全关闭本会话流水线（无论是否启动/已关闭均可调用）"""
    analyzer = app.state.analyzer
    try:
        if analyzer._pipeline_on:
            await asyncio.to_thread(analyzer.close_pipeline)
    except Exception as e:
        print(f"[server] close_pipeline 异常: {e}")


def _build_summary(outcome: SessionOutcome, results: list, elapsed_s: float,
                   abnormal_frames: int) -> dict:
    n = len(results)
    return {
        "ok": True,
        "submitted": outcome.submitted,
        "returned": n,
        "abnormal_frames": abnormal_frames,
        "elapsed_ms": round(elapsed_s * 1000.0, 1),
        "fps": round(n / elapsed_s, 2) if elapsed_s > 0 else None,
        "note": "",
    }


# =========================================================================== #
# FastAPI 应用 + 生命周期                                                    #
# =========================================================================== #
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[server] 正在加载 3 个 GPU 模型（分割/车道/车辆）...")
    # 模型加载是耗时同步操作，放到线程里避免阻塞事件循环
    analyzer = await asyncio.to_thread(
        lambda: TrafficAnalyzer(DetectionConfig(
            enable_bike_person=True, enable_stop_slow=True)))
    app.state.analyzer = analyzer
    print("[server] 模型加载完成，开始对外服务")
    try:
        yield
    finally:
        print("[server] 关闭中...")
        await asyncio.to_thread(analyzer.shutdown)


app = FastAPI(title="交通视频 WebSocket 实时分析服务(精简版)", version="1.1", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# 唯一接口：WebSocket 前端逐帧推送 JPEG，服务端逐帧回结果                     #
# --------------------------------------------------------------------------- #
@app.websocket("/ws/analyze")
async def ws_analyze(ws: WebSocket):
    await ws.accept()

    # 等待客户端先发 {"action":"start", ...}
    try:
        first = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
        cfg = json.loads(first)
    except Exception:
        await ws.send_text(json.dumps({"type": "error", "message": "请先发送 start 配置"}))
        await ws.close(code=1008)
        return

    try:
        send_images = bool(cfg.get("send_images", True))
        quality = int(cfg.get("quality", 85))
        idle_s = float(cfg.get("idle_s", 90.0))
    except (ValueError, TypeError):
        await ws.send_text(json.dumps(
            {"type": "error", "message": "start 配置字段不合法"}, ensure_ascii=False))
        await ws.close(code=1008)
        return

    analyzer = app.state.analyzer
    outcome = SessionOutcome()
    completed = asyncio.Event()
    results = []
    abnormal_frames = 0
    progress = {"last": time.monotonic()}

    async def emit(item: dict):
        nonlocal abnormal_frames
        if item["is_abnormal"]:
            abnormal_frames += 1
        results.append(item)
        await ws.send_text(json.dumps(item, ensure_ascii=False))

    async def producer():
        """读取前端 JSON 文本：{"frame": b64} 投帧；{"action":"end"/"cancel"} 结束"""
        try:
            while True:
                try:
                    text = await asyncio.wait_for(ws.receive_text(), timeout=idle_s)
                except asyncio.TimeoutError:
                    raise SessionError("前端空闲超时，已结束会话")
                except WebSocketDisconnect:
                    outcome.aborted = True
                    break
                try:
                    obj = json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    raise SessionError("协议错误：仅接受 JSON 文本消息")

                action = obj.get("action")
                if action == "end":
                    break
                if action == "cancel":
                    outcome.aborted = True
                    break

                b64 = obj.get("frame")
                if b64:
                    raw = base64.b64decode(b64)
                    frame = await asyncio.to_thread(_decode_jpeg, raw)
                    if frame is None:
                        raise SessionError("收到无法解码的 JPEG 帧（frame 字段应为 JPEG base64）")
                    await asyncio.to_thread(analyzer.submit_frame, frame)
                    outcome.submitted += 1
                    progress["last"] = time.monotonic()
        finally:
            completed.set()

    error_msg = None
    t0 = None
    try:
        await asyncio.to_thread(analyzer.reset_video_state)
        await asyncio.to_thread(analyzer.start_pipeline)
        t0 = time.perf_counter()
        try:
            await asyncio.gather(
                producer(),
                _consumer_core(app, outcome, completed, results, emit,
                               send_images, quality, progress))
        except SessionError as e:
            error_msg = str(e)          # 会话级错误（空闲超时/协议/流水线无输出等）
        except WebSocketDisconnect:
            outcome.aborted = True      # 客户端中途断开：正常静默收尾
        except Exception as e:
            error_msg = str(e)
    except Exception as e:
        error_msg = f"流水线启动失败: {e}"
    finally:
        await _close_pipeline_safe(app)

    if error_msg is not None:
        try:
            await ws.send_text(json.dumps(
                {"type": "error", "message": error_msg}, ensure_ascii=False))
        except Exception:
            pass
        try:
            await ws.close(code=1011)
        except Exception:
            pass
        return

    # 正常结束：结果已逐帧推过，这里只补统计摘要
    elapsed = time.perf_counter() - t0
    done_msg = _build_summary(outcome, results, elapsed, abnormal_frames)
    done_msg["type"] = "done"
    done_msg.pop("results", None)          # 结果已逐帧推过
    done_msg["returned"] = len(results)
    try:
        await ws.send_text(json.dumps(done_msg, ensure_ascii=False))
        await ws.close()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# 启动入口                                                                     #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="交通视频 WebSocket 实时分析服务(精简版)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print("=" * 62)
    print("WebSocket 实时分析服务（精简版，仅保留 /ws/analyze）")
    print(f"  逐帧推送    : WS   ws://{args.host}:{args.port}/ws/analyze")
    print("=" * 62)
    uvicorn.run(app, host=args.host, port=args.port)
