"""Download Facenet512 weights during Render build (not at request time)."""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from deepface import DeepFace

DeepFace.build_model("Facenet512")
print("Facenet512 ready")
