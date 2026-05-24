import os
import tempfile
import asyncio
from contextlib import asynccontextmanager
from functools import lru_cache

import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException

_MODEL_NAME = os.getenv("FACE_MODEL", "buffalo_sc")
_MAX_IMAGE_SIDE = int(os.getenv("FACE_MAX_IMAGE_SIDE", "640"))


@lru_cache(maxsize=1)
def _face_analyzer():
    from insightface.app import FaceAnalysis

    analyzer = FaceAnalysis(
        name=_MODEL_NAME,
        providers=["CPUExecutionProvider"],
    )
    analyzer.prepare(ctx_id=-1, det_size=(320, 320))
    return analyzer


def _prepare_image(image_path: str) -> str:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Invalid image file")

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest > _MAX_IMAGE_SIDE:
        scale = _MAX_IMAGE_SIDE / longest
        image = cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    resized_path = f"{image_path}.resized.jpg"
    cv2.imwrite(resized_path, image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return resized_path


def _extract_embedding(image_path: str) -> list[float]:
    resized_path = _prepare_image(image_path)
    try:
        image = cv2.imread(resized_path)
        faces = _face_analyzer().get(image)
        if not faces:
            raise ValueError("No face detected in the image. Use a clear front-facing photo.")

        embedding = faces[0].normed_embedding
        if embedding is None:
            raise ValueError("Could not generate face embedding")

        return embedding.tolist()
    finally:
        if os.path.exists(resized_path):
            os.remove(resized_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up model at startup so first /embedding request is fast.
    await asyncio.to_thread(_face_analyzer)
    yield


app = FastAPI(title="HRMS Face Embedding Service", lifespan=lifespan)


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
