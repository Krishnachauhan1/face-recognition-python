"""Download InsightFace weights into ./models (Render + Vercel build)."""
import os

os.environ.setdefault("FACE_MODEL", "buffalo_sc")

from insightface.app import FaceAnalysis

models_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(models_root, exist_ok=True)

analyzer = FaceAnalysis(
    name="buffalo_sc",
    root=models_root,
    providers=["CPUExecutionProvider"],
)
analyzer.prepare(ctx_id=-1, det_size=(320, 320))
print(f"buffalo_sc ready at {models_root}")
