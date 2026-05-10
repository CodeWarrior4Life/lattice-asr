from __future__ import annotations

import asyncio
import base64
import os
from dataclasses import asdict

from fastapi import FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel

from lattice_asr import Transcriber

_API_VERSION = "v1"
_transcriber_singleton: Transcriber | None = None
_transcriber_singleton_lock = asyncio.Lock()


async def _get_transcriber() -> Transcriber:
    """Lazy-init the singleton Transcriber, lock-guarded against concurrent first-callers."""
    global _transcriber_singleton
    async with _transcriber_singleton_lock:
        if _transcriber_singleton is None:
            _transcriber_singleton = Transcriber()
    return _transcriber_singleton


class TranscribeRequest(BaseModel):
    audio_b64: str
    sample_rate: int = 16000
    language: str | None = None
    diarize: bool = False
    tenant_id: str | None = None


def create_app() -> FastAPI:
    app = FastAPI(title="lattice-asr-server", version=_API_VERSION)

    @app.middleware("http")
    async def add_version_header(request: Request, call_next):
        resp: Response = await call_next(request)
        resp.headers["X-Lattice-Asr-Api-Version"] = _API_VERSION
        return resp

    def _check_auth(authorization: str | None) -> None:
        expected = os.environ.get("LATTICE_ASR_API_KEY")
        if expected is None:
            return
        if authorization != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.get("/v1/health")
    async def health():
        return {"status": "ok", "version": _API_VERSION}

    @app.post("/v1/transcribe")
    async def transcribe(
        req: TranscribeRequest,
        authorization: str | None = Header(default=None),
    ):
        _check_auth(authorization)
        try:
            audio = base64.b64decode(req.audio_b64)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid audio_b64: {exc}") from exc
        t = await _get_transcriber()
        try:
            result = await t.transcribe(
                audio,
                sample_rate=req.sample_rate,
                language=req.language,
                diarize=req.diarize,
                tenant_id=req.tenant_id,
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"transcription failed: {exc}") from exc
        return {
            "text": result.text,
            "language": result.language,
            "confidence": result.confidence,
            "engine_name": result.engine_name,
            "segments": [asdict(s) for s in result.segments],
            "speaker_segments": [asdict(s) for s in result.speaker_segments],
            "audio_duration_ms": result.audio_duration_ms,
            "duration_ms": result.duration_ms,
        }

    return app
