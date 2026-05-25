"""Vercel default FastAPI entrypoint (re-exports face_service.app)."""
from face_service import app

__all__ = ["app"]
