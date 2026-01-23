from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from emotion_ai.predictor import EmotionPredictor
from emotion_ai.utils import bgr_from_bytes, bgr_to_base64_jpeg
from emotion_ai.version import __version__

logger = logging.getLogger("emotion_ai.api")


def create_app(config_path: str = "config/config.yaml", predictor: Optional[EmotionPredictor] = None) -> FastAPI:
    app = FastAPI(
        title="Emotion Intelligence API",
        description="Production-Ready Real-Time Emotion Intelligence System (DDAMFN)",
        version=__version__,
    )

    app.state.predictor = predictor

    @app.on_event("startup")
    def _startup() -> None:
        if app.state.predictor is None:
            logger.info("Initializing EmotionPredictor...")
            app.state.predictor = EmotionPredictor(config_path=config_path)

    @app.get("/health")
    def health_check() -> Dict[str, Any]:
        pred = app.state.predictor
        device = str(pred.device) if pred else "unknown"
        return {"status": "ok", "version": __version__, "device": device}

    @app.post("/predict")
    async def predict_emotion(file: UploadFile = File(...), annotate: bool = False) -> JSONResponse:
        """Upload an image -> JSON detections.

        Query params:
            annotate=true -> include base64 JPEG overlay image in response
        """
        pred: EmotionPredictor = app.state.predictor
        t0 = time.perf_counter()

        try:
            content = await file.read()
            img_bgr = bgr_from_bytes(content)

            detections = pred.predict(img_bgr)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            payload: Dict[str, Any] = {
                "filename": file.filename,
                "count": len(detections),
                "detections": detections,
                "processing_ms": round(float(elapsed_ms), 3),
            }

            if annotate:
                annotated = pred.annotate_image(img_bgr, detections)
                payload["annotated_image_base64_jpeg"] = bgr_to_base64_jpeg(annotated)

            return JSONResponse(content=payload)

        except Exception as e:
            logger.exception("Prediction error")
            return JSONResponse(content={"error": str(e)}, status_code=500)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
