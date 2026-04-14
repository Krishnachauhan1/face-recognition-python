from fastapi import FastAPI, File, UploadFile
from deepface import DeepFace
import tempfile
import os

app = FastAPI()

@app.post("/embedding")
async def embedding(image: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        temp.write(await image.read())
        temp_path = temp.name

    try:
        result = DeepFace.represent(
            img_path=temp_path,
            model_name="Facenet512",
            enforce_detection=False
        )

        return {
            "status": True,
            "embedding": result[0]["embedding"]
        }

    except Exception as e:
        return {
            "status": False,
            "error": str(e)
        }

    finally:
        os.remove(temp_path)

        