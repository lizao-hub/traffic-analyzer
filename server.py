"""
server.py —— FastAPI + WebSocket 流式视频分析服务（逐帧，非整段视频）
========================================================================

设计：
  - 客户端逐帧发送 JPEG 字节流（二进制），服务端解码 → submit_frame 进多进程流水线
    → get_result 取回已完成结果 → JPEG 编码 → 逐帧发回客户端。
  - 全程只有 FastAPI/Starlette 强制要求的 `await ws.receive/send`，不写自定义
    asyncio 编排（无 asyncio.Queue / to_thread / call_soon_threadsafe）。
  - 多进程流水线 TrafficAnalyzerMP 在服务启动时创建一次，模型只加载一次。

运行：
    uvicorn server:app --host 0.0.0.0 --port 8000

注意：
  - 本 demo 为单客户端会话（全局共享一个流水线）。多客户端并发需引入会话调度 +
    多条流水线（见 docs/线程与进程复盘.md 的扩展方案）。
  - submit_frame 在流水线满时会背压阻塞（约 13ms/帧 @ 74FPS），阻塞发生在事件
    循环里：单客户端可接受（相当于流控）；如需不阻塞事件循环，再包一层线程即可。
"""
import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from process_mp import TrafficAnalyzerMP, DetectionConfig

app = FastAPI(title="YOLO11 无人机交通流分析 - 流式服务")

# 全局唯一的多进程流水线（模型加载一次）。单客户端使用。
analyzer = TrafficAnalyzerMP(
    DetectionConfig(enable_bike_person=True, enable_stop_slow=True),
    window=8,
)


@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_on": analyzer._pipeline_on}


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    """逐帧流式：收 JPEG 字节 → 分析 → 回 JPEG 字节。

    协议：
      client -> server: 二进制 JPEG 帧（可连续发送）
      server -> client: 二进制 JPEG 结果帧（按处理完成顺序，逐帧）
    """
    await ws.accept()
    analyzer.start_pipeline()
    print("流式会话开始")
    try:
        while True:
            data = await ws.receive_bytes()
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                await ws.send_json({"error": "无法解码该帧"})
                continue

            # 提交进多进程流水线（满时背压阻塞 = 流控）
            analyzer.submit_frame(frame)

            # 把当前所有已完成的结果推回客户端
            while True:
                seq, res = analyzer.get_result(timeout=0)
                if res is None:
                    break
                ok, buf = cv2.imencode(".jpg", res.frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    # 可选：先 send_json({"seq": seq, "metrics": res.metrics})
                    # 再 send_bytes(buf)，客户端按顺序区分文本/二进制即可
                    await ws.send_bytes(buf.tobytes())
    except WebSocketDisconnect:
        print("客户端断开")
    except Exception as e:
        print(f"ws_stream 异常: {e!r}")
    finally:
        analyzer.close_pipeline()
        print("流式会话结束")
