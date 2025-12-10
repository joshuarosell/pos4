from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .camera import CameraNotReadyError, USBCameraStream
from .detector import YOLOOnnxDetector, GestureDetector, POSession
import cv2
import yaml
import json

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"
STATIC_DIR = WEB_DIR / "static"

app = FastAPI(title="Jetson USB Camera Stream", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

camera = USBCameraStream()
prices = yaml.safe_load((BASE_DIR / "prices.yaml").read_text(encoding="utf-8"))
detector = YOLOOnnxDetector(BASE_DIR / "best.onnx", BASE_DIR / "data.yaml")
gestures = GestureDetector()
session = POSession(prices)
ws_clients: set[WebSocket] = set()


@app.on_event("startup")
async def startup_event() -> None:
    camera.start()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    camera.stop()


import asyncio


async def _frame_generator():
    while True:
        try:
            # Get raw for inference and draw, then encode
            frame_bgr = camera.get_frame_bgr()
        except CameraNotReadyError:
            await asyncio.sleep(0.1)
            continue

        # Inference
        detections = detector.infer(frame_bgr)

        # Gesture detection
        gest = gestures.detect(frame_bgr)
        event: dict | None = None
        # Draw gesture/state overlay for debugging
        overlay = f"gesture: {gest or 'none'} | session: {'active' if session.active else 'inactive'}"
        cv2.putText(frame_bgr, overlay, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        if gest == "open_hand" and not session.active:
            session.start()
            event = {"type": "session_start"}
        elif gest == "closed_hand" and session.active:
            total = session.end()
            event = {"type": "session_end", "total": total, "items": session.items}

        # Add item if present
        if detections:
            top = max(detections, key=lambda d: d.conf)
            added = session.maybe_add_item(top.cls_name, top.conf)
            if added:
                name, price = added
                event = {"type": "item", "name": name, "price": price}

        # Draw boxes
        for det in detections:
            x1, y1, x2, y2 = det.xyxy
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det.cls_name} {det.conf:.2f}"
            cv2.putText(frame_bgr, label, (x1, max(y1-6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 240, 10), 1, cv2.LINE_AA)

        ok, buffer = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            await asyncio.sleep(0.01)
            continue

        # Broadcast event if any
        if event:
            data = json.dumps(event)
            for ws in list(ws_clients):
                try:
                    await ws.send_text(data)
                except Exception:
                    ws_clients.discard(ws)

        frame = buffer.tobytes()
        boundary = b"--frame\r\n"
        headers = b"Content-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
        yield boundary + headers + frame + b"\r\n"
        await asyncio.sleep(0.01)


@app.get("/", response_class=FileResponse)
async def read_index() -> FileResponse:
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html missing")
    return FileResponse(index_path)


@app.get("/video-stream")
async def video_stream() -> StreamingResponse:
    return StreamingResponse(_frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/health")
async def health() -> dict[str, str]:
    try:
        camera.get_frame()
    except CameraNotReadyError:
        return {"status": "warming_up"}
    return {"status": "ok"}


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    # On connect, send current session state
    await ws.send_json({"type": "session", "active": session.active, "items": session.items})
    try:
        while True:
            # Client may send pings or commands; we ignore for now
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_clients.discard(ws)
