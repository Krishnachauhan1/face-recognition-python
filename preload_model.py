"""Download InsightFace weights during Render build."""
import os

os.environ.setdefault("FACE_MODEL", "buffalo_sc")

from insightface.app import FaceAnalysis

analyzer = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
analyzer.prepare(ctx_id=-1, det_size=(320, 320))
print("buffalo_sc ready")
