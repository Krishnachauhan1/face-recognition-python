import os
import tempfile
import asyncio
from functools import lru_cache

# Keep TensorFlow CPU-only and quiet before any TF import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

from fastapi import FastAPI, File, UploadFile, HTTPException

app = FastAPI(title="HRMS Face Embedding Service")

_MODEL_NAME = "Facenet512"
_DETECTOR = "opencv"


@lru_cache(maxsize=1)
def _deepface():
    from deepface import DeepFace

    DeepFace.build_model(_MODEL_NAME)
    return DeepFace


def _extract_embedding(image_path: str) -> list[float]:
    deepface = _deepface()
    result = deepface.represent(
        img_path=image_path,
        model_name=_MODEL_NAME,
        detector_backend=_DETECTOR,
        enforce_detection=False,
    )

    if not result or "embedding" not in result[0]:
        raise ValueError("No face embedding could be generated")

    return result[0]["embedding"]


@app.get("/")
def root():
    return {"status": True, "service": "face-embedding"}


@app.get("/health")
def health():
    return {"status": True, "service": "face-embedding"}


@app.post("/embedding")
async def embedding(image: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        temp.write(await image.read())
        temp_path = temp.name

    try:
        vector = await asyncio.to_thread(_extract_embedding, temp_path)
        return {"status": True, "embedding": vector}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
