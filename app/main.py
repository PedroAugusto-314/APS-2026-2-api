import os
from contextlib import asynccontextmanager
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

MODEL_PATH = os.getenv("MODEL_PATH", "best_ppe_yolo11s.pt")
CONFIDENCE = float(os.getenv("CONFIDENCE", "0.35"))
PREPROCESS = os.getenv("PREPROCESS", "1") == "1"
DENOISE = os.getenv("DENOISE", "1") == "1"
ADAPTIVE_GAMMA = os.getenv("ADAPTIVE_GAMMA", "1") == "1"
TARGET_BRIGHTNESS = float(os.getenv("TARGET_BRIGHTNESS", "110"))

model: YOLO | None = None
_gamma_cache: dict[int, np.ndarray] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model
    model = YOLO(MODEL_PATH)
    yield
    model = None


app = FastAPI(title="API de Reconhecimento de Pessoas e EPI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _gamma_lut(gamma: float) -> np.ndarray:
    key = int(round(gamma * 100))
    if key not in _gamma_cache:
        table = ((np.arange(256) / 255.0) ** gamma) * 255.0
        _gamma_cache[key] = np.clip(table, 0, 255).astype(np.uint8)
    return _gamma_cache[key]


def preprocess_image(image: np.ndarray) -> np.ndarray:
    if not PREPROCESS:
        return image

    if DENOISE:
        image = cv2.bilateralFilter(image, d=5, sigmaColor=40, sigmaSpace=40)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)

    if ADAPTIVE_GAMMA:
        mean_l = float(lightness.mean())
        if mean_l < TARGET_BRIGHTNESS - 15:
            gamma = np.clip(
                np.log(TARGET_BRIGHTNESS / 255.0) / np.log(max(mean_l, 1.0) / 255.0),
                0.5,
                1.0,
            )
            lightness = cv2.LUT(lightness, _gamma_lut(float(gamma)))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)

    enhanced = cv2.cvtColor(cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR)

    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    return cv2.addWeighted(enhanced, 1.25, blurred, -0.25, 0)


def detect_image(image: np.ndarray) -> list[dict[str, Any]]:
    if model is None:
        raise RuntimeError("Modelo ainda não foi carregado")

    processed_image = preprocess_image(image)
    results = model.predict(source=processed_image, conf=CONFIDENCE, verbose=False)
    detections: list[dict[str, Any]] = []

    for result in results:
        names = result.names
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls[0])
            x1, y1, x2, y2 = [round(float(value), 2) for value in box.xyxy[0]]
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": names[class_id],
                    "confidence": round(float(box.conf[0]), 4),
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }
            )

    return detections


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/detect")
async def detect(file: UploadFile = File(...)) -> dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo ainda não foi carregado")

    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Envie uma imagem válida")

    return {"filename": file.filename, "detections": detect_image(image)}


@app.websocket("/ws/detect")
async def detect_websocket(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        while True:
            frame = await websocket.receive_bytes()
            image = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)

            if image is None:
                await websocket.send_json({"error": "Envie um frame JPEG válido"})
                continue

            try:
                detections = detect_image(image)
            except RuntimeError as error:
                await websocket.send_json({"error": str(error)})
                continue

            await websocket.send_json({"detections": detections})
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)