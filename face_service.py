from fastapi import FastAPI, File, UploadFile, HTTPException
from deepface import DeepFace
import tempfile
import os

app = FastAPI(title="HRMS Face Embedding Service")


@app.get("/health")
def health():
    return {"status": True, "service": "face-embedding"}


@app.post("/embedding")
async def embedding(image: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        temp.write(await image.read())
        temp_path = temp.name

    try:
        result = DeepFace.represent(
            img_path=temp_path,
            model_name="Facenet512",
            enforce_detection=False,
        )

        if not result or "embedding" not in result[0]:
            raise HTTPException(status_code=422, detail="No face embedding could be generated")

        return {
            "status": True,
            "embedding": result[0]["embedding"],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
