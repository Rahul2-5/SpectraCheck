"""Run: python -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000."""

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from aidetect.io_utils import load_config
from app.backend.inference import InferenceService, inspect_upload

ROOT = Path(__file__).resolve().parents[2]


def create_app(root: str | Path = ROOT, *, service=None, settings: dict | None = None) -> FastAPI:
    root = Path(root).resolve()
    settings = settings or load_config(Path(__file__).with_name("settings.yaml"))
    service = service or InferenceService(root, settings)
    api = FastAPI(title="SpectraCheck local API", docs_url=None, redoc_url=None)
    api.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"])

    @api.get("/api/health")
    def health():
        return service.health()

    @api.post("/api/analyze")
    async def analyze(request: Request):
        # Raw binary stream, not UploadFile: uploaded bytes never spill to disk.
        payload = bytearray()
        async for chunk in request.stream():
            if len(payload) + len(chunk) > settings["max_upload_bytes"]:
                raise HTTPException(413, "This image exceeds the upload size limit.")
            payload.extend(chunk)
        encoded = bytes(payload)
        try:
            inspect_upload(encoded, settings)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if not service.health()["ready"]:
            raise HTTPException(503, "The research model is not available yet. Check the local model bundle.")
        try:
            return await run_in_threadpool(service.analyze, encoded)
        except ValueError as exc:
            # Preprocessing exceptions are clear; internal artifact paths stay off the UI.
            if "smaller than target" in str(exc).lower() or "decode image" in str(exc).lower():
                raise HTTPException(422, str(exc)) from exc
            raise HTTPException(503, "The model bundle could not be verified. Check the local setup.") from exc
        except (OSError, KeyError, TypeError, ImportError) as exc:
            raise HTTPException(503, "The model could not be loaded. Check the local setup and library versions.") from exc

    dist = root / settings["frontend_dist"]
    if dist.is_dir():
        api.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return api


app = create_app()
