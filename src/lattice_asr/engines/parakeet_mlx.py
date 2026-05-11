"""ParakeetMlxEngine — Apple Silicon MLX EN-only. Implemented in W3.1. Spec §6.1."""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
import tempfile
import time
import wave
from collections.abc import AsyncIterator
from typing import Any

from lattice_asr.engines.base import TranscriptionEngine
from lattice_asr.types import EngineCapabilities, Segment, TranscriptionResult


class ParakeetMlxEngine(TranscriptionEngine):
    """Adapter for senstella/parakeet-mlx (Apple Silicon MLX). EN-only; lazy-loaded."""

    def __init__(self, *, model: str = "mlx-community/parakeet-tdt-0.6b-v3"):
        self._model_name = model
        self._model: Any = None  # lazy-loaded via _ensure_model
        self.capabilities = EngineCapabilities(
            name="parakeet-mlx",
            languages=frozenset({"en"}),
            streaming=True,
            requires_gpu=False,
            requires_apple_silicon=True,
            typical_rtfx=15.0,
        )

    async def warmup(self) -> None:
        await asyncio.to_thread(self._ensure_model)

    def _ensure_model(self) -> Any:
        if self._model is None:
            from parakeet_mlx import from_pretrained  # type: ignore[import-untyped]

            self._model = from_pretrained(self._model_name)
        return self._model

    @staticmethod
    def _write_wav_tempfile(audio_pcm: bytes, sample_rate: int) -> str:
        """Write PCM (or pass-through WAV) bytes to a tempfile path. Caller cleans up."""
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            if audio_pcm[:4] == b"RIFF":
                with open(path, "wb") as f:
                    f.write(audio_pcm)
            else:
                with wave.open(path, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(sample_rate)
                    w.writeframes(audio_pcm)
            return path
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(path)
            raise

    @staticmethod
    def _audio_duration_ms(audio_pcm: bytes, sample_rate: int) -> int:
        """Derive audio duration in ms from PCM or WAV bytes (16-bit mono assumed for raw PCM)."""
        if audio_pcm[:4] == b"RIFF":
            with wave.open(io.BytesIO(audio_pcm), "rb") as w:
                frames = w.getnframes()
                rate = w.getframerate()
                return int(frames / rate * 1000)
        # raw PCM, 16-bit mono assumption
        return int(len(audio_pcm) / 2 / sample_rate * 1000)

    async def transcribe(
        self,
        audio_pcm: bytes,
        sample_rate: int,
        language: str | None,
    ) -> TranscriptionResult:
        """Transcribe raw PCM or WAV bytes. Requires sample_rate=16000."""
        if sample_rate != 16000:
            raise ValueError(f"ParakeetMlxEngine requires sample_rate=16000, got {sample_rate}")
        t0 = time.monotonic()
        model = await asyncio.to_thread(self._ensure_model)
        path = await asyncio.to_thread(self._write_wav_tempfile, audio_pcm, sample_rate)
        try:
            result = await asyncio.to_thread(model.transcribe, path)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(path)

        text = getattr(result, "text", "") or ""
        # parakeet-mlx may expose sentence/token timestamps via .sentences or .tokens;
        # for v0.1 minimum we return empty segments. If sentences exist with timing,
        # populate Segment tuples (best-effort, parakeet-mlx >= 0.5 surface).
        segs: tuple[Segment, ...] = ()
        sentences = getattr(result, "sentences", None)
        if sentences:
            try:
                segs = tuple(
                    Segment(
                        text=getattr(s, "text", "").strip(),
                        start_ms=int(getattr(s, "start", 0.0) * 1000),
                        end_ms=int(getattr(s, "end", 0.0) * 1000),
                        confidence=1.0,
                    )
                    for s in sentences
                )
            except (AttributeError, TypeError, ValueError):
                segs = ()

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        audio_ms = self._audio_duration_ms(audio_pcm, sample_rate)
        return TranscriptionResult(
            text=text.strip(),
            language="en",
            confidence=1.0,
            engine_name="parakeet-mlx",
            segments=segs,
            speaker_segments=(),
            audio_duration_ms=audio_ms,
            duration_ms=elapsed_ms,
        )

    async def transcribe_streaming(
        self,
        audio_chunks: AsyncIterator[bytes],
        sample_rate: int,
        language: str | None,
    ) -> AsyncIterator[TranscriptionResult]:
        """Stream PCM chunks; one result per ~1s buffered window. Requires sample_rate=16000."""
        if sample_rate != 16000:
            raise ValueError(f"ParakeetMlxEngine requires sample_rate=16000, got {sample_rate}")
        buf = bytearray()
        async for chunk in audio_chunks:
            buf.extend(chunk)
            if len(buf) >= sample_rate * 2:  # ~1s of 16-bit mono audio
                yield await self.transcribe(bytes(buf), sample_rate, language)
                buf.clear()
        if buf:
            yield await self.transcribe(bytes(buf), sample_rate, language)
