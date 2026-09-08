"""ParakeetMlxEngine — Apple Silicon MLX EN-only. Implemented in W3.1. Spec §6.1.

THREADING LAW (2026-09-08, measured on trinity, mlx 0.32.0 / parakeet-mlx 0.5.2):
**every MLX operation for this engine runs on ONE dedicated thread.**

MLX streams are thread-local. `parakeet_mlx.from_pretrained` leaves the dtype
casts of every weight LAZY (`v.astype(dtype)`), each recorded against the
loading thread's default stream. The first thread that forces one of those
pending casts must be the thread that owns that stream, or MLX raises

    RuntimeError('There is no Stream(cpu, 0) in current thread.')

`asyncio.to_thread` hands work to a shared pool where any idle worker may pick
up the job, so load-on-thread-A / transcribe-on-thread-B was a lottery the
operator lost on the FIRST dictation after every daemon start (7 identical
log signatures in lattice-dictate, 5-for-5 on the pattern warm -> first press
fails -> second press works). Reproduced in isolation: load on A, transcribe
on fresh B/C/D and on main -> all four fail; load + full `mx.eval` on A, or
load + transcribe pinned to one executor thread -> all pass.

Two layers, both kept on purpose:
1. `_materialize` forces every parameter to a concrete array right after load,
   so nothing lazy survives on the loader's stream.
2. A single-worker executor (`_mlx`) owns load AND inference, so even state the
   model creates lazily later (caches, positional tables) is born and used on
   the same thread. This also serializes inference on the model, which is not
   reentrant anyway.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
import tempfile
import time
import wave
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from lattice_asr.engines.base import TranscriptionEngine
from lattice_asr.types import EngineCapabilities, Segment, TranscriptionResult


class ParakeetMlxEngine(TranscriptionEngine):
    """Adapter for senstella/parakeet-mlx (Apple Silicon MLX). EN-only; lazy-loaded."""

    required_packages = ("parakeet_mlx",)

    def __init__(self, *, model: str = "mlx-community/parakeet-tdt-0.6b-v3"):
        self._model_name = model
        self._model: Any = None  # lazy-loaded via _ensure_model
        # The one thread that ever touches MLX for this engine. See module doc.
        self._mlx = ThreadPoolExecutor(max_workers=1, thread_name_prefix="parakeet-mlx")
        self.capabilities = EngineCapabilities(
            name="parakeet-mlx",
            languages=frozenset({"en"}),
            streaming=True,
            requires_gpu=False,
            requires_apple_silicon=True,
            typical_rtfx=15.0,
        )

    async def warmup(self) -> None:
        await self._on_mlx_thread(self._ensure_model)

    async def _on_mlx_thread(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run `fn` on the engine's dedicated MLX thread (never the shared pool)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._mlx, fn, *args)

    @staticmethod
    def _materialize(model: Any) -> None:
        """Force every parameter to a concrete array on the CURRENT thread.

        Best-effort: a model without `parameters()` (test fakes) is left alone,
        and an MLX import failure here is not this method's problem to report.
        """
        params = getattr(model, "parameters", None)
        if params is None:
            return
        try:
            import mlx.core as mx  # type: ignore[import-untyped]
            from mlx.utils import tree_flatten  # type: ignore[import-untyped]

            mx.eval([v for _, v in tree_flatten(params())])
        except ImportError:
            return

    def _ensure_model(self) -> Any:
        if self._model is None:
            from parakeet_mlx import from_pretrained  # type: ignore[import-untyped]

            model = from_pretrained(self._model_name)
            self._materialize(model)
            self._model = model
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
        model = await self._on_mlx_thread(self._ensure_model)
        path = await asyncio.to_thread(self._write_wav_tempfile, audio_pcm, sample_rate)
        try:
            result = await self._on_mlx_thread(model.transcribe, path)
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
