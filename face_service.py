import os
import tempfile
import asyncio
from functools import lru_cache

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException

_MODEL_NAME = os.getenv("FACE_MODEL", "buffalo_sc")
_MAX_IMAGE_SIDE = int(os.getenv("FACE_MAX_IMAGE_SIDE", "640"))
_MODELS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def _rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    return rgb[:, :, ::-1].copy()


def _load_bgr(image_path: str) -> np.ndarray:
    try:
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
    except Exception as exc:
        raise ValueError("Invalid image file") from exc

    if rgb.size == 0:
        raise ValueError("Invalid image file")

    return _rgb_to_bgr(rgb)


def _resize_bgr(bgr: np.ndarray) -> np.ndarray:
    height, width = bgr.shape[:2]
    longest = max(height, width)
    if longest <= _MAX_IMAGE_SIDE:
        return bgr

    scale = _MAX_IMAGE_SIDE / longest
    new_size = (int(width * scale), int(height * scale))
    rgb = Image.fromarray(bgr[:, :, ::-1])
    rgb = rgb.resize(new_size, Image.Resampling.LANCZOS)
    return _rgb_to_bgr(np.asarray(rgb))


@lru_cache(maxsize=1)
def _face_analyzer():
    from insightface.app import FaceAnalysis

    os.makedirs(_MODELS_ROOT, exist_ok=True)
    analyzer = FaceAnalysis(
        name=_MODEL_NAME,
        root=_MODELS_ROOT,
        providers=["CPUExecutionProvider"],
    )
    analyzer.prepare(ctx_id=-1, det_size=(320, 320))
    return analyzer


def _extract_embedding(image_path: str) -> list[float]:
    bgr = _resize_bgr(_load_bgr(image_path))
    faces = _face_analyzer().get(bgr)
    if not faces:
        raise ValueError("No face detected in the image. Use a clear front-facing photo.")

    embedding = faces[0].normed_embedding
    if embedding is None:
        raise ValueError("Could not generate face embedding")

    return embedding.tolist()


app = FastAPI(title="HRMS Face Embedding Service")


@app.get("/")
def root():
    return {"status": True, "service": "face-embedding", "model": _MODEL_NAME}


@app.get("/health")
def health():
    return {"status": True, "service": "face-embedding", "model": _MODEL_NAME}


@app.post("/embedding")
async def embedding(image: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        temp.write(await image.read())
        temp_path = temp.name

    try:
        vector = await asyncio.to_thread(_extract_embedding, temp_path)
        return {"status": True, "embedding": vector, "dimensions": len(vector)}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
