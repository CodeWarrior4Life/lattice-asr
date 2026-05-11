"""ParakeetTdtEngine — NVIDIA NeMo Parakeet-TDT EN-only. Implemented in W3.2. Spec §6.2."""

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
from lattice_asr.types import EngineCapabilities, TranscriptionResult


class ParakeetTdtEngine(TranscriptionEngine):
    """Adapter for NVIDIA NeMo Parakeet-TDT (Linux + CUDA). EN-only; lazy-loaded.

    Default checkpoint `nvidia/parakeet-tdt-0.6b-v3` is verified-runnable per S41
    (measured RTF 64.13× on Cypher RTX 2070 Turing sm_75 via direct nemo-toolkit API).
    """

    def __init__(self, *, model: str = "nvidia/parakeet-tdt-0.6b-v3"):
        self._model_name = model
        self._model: Any = None  # lazy-loaded via _ensure_model
        self.capabilities = EngineCapabilities(
            name="parakeet-tdt",
            languages=frozenset({"en"}),
            streaming=True,
            requires_gpu=True,
            requires_apple_silicon=False,
            typical_rtfx=50.0,
        )

    async def warmup(self) -> None:
        """Load the model and pay the one-time CUDA-kernel compile cost.

        NeMo's first transcribe() call after model load triggers CUDA kernel JIT compile
        (~0.5s on RTX 2070); without absorbing that here the first user-facing transcribe()
        runs slow. Synthesize ~0.5s of silence (RIFF 16 kHz mono) and run a throwaway
        transcribe to prime the GPU.
        """
        await asyncio.to_thread(self._ensure_model)
        # 0.5s of silence: 8000 samples × 2 bytes (16-bit PCM mono @ 16 kHz).
        # Warmup is best-effort; a failed dummy inference must not block engine use.
        silence = b"\x00" * (8000 * 2)
        with contextlib.suppress(Exception):
            await self.transcribe(silence, sample_rate=16000, language="en")

    def _ensure_model(self) -> Any:
        if self._model is None:
            import nemo.collections.asr as nemo_asr  # type: ignore[import-untyped]

            self._model = nemo_asr.models.ASRModel.from_pretrained(self._model_name)
        return self._model

    @staticmethod
    def _write_wav_tempfile(audio_pcm: bytes, sample_rate: int) -> str:
        """Write PCM (or pass-through WAV) bytes to a tempfile path. Caller cleans up."""
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)  # noqa: SIM115
        try:
            if audio_pcm[:4] == b"RIFF":
                tmp.write(audio_pcm)
                tmp.flush()
                tmp.close()
                return tmp.name
            tmp.close()
            with wave.open(tmp.name, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                w.writeframes(audio_pcm)
            return tmp.name
        except Exception:
            with contextlib.suppress(Exception):
                tmp.close()
            with contextlib.suppress(OSError):
                os.unlink(tmp.name)
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
            raise ValueError(f"ParakeetTdtEngine requires sample_rate=16000, got {sample_rate}")
        t0 = time.monotonic()
        model = await asyncio.to_thread(self._ensure_model)
        path = await asyncio.to_thread(self._write_wav_tempfile, audio_pcm, sample_rate)
        try:
            output = await asyncio.to_thread(model.transcribe, [path])
        finally:
            with contextlib.suppress(OSError):
                os.unlink(path)

        # NeMo transcribe([path]) returns a list of result objects (one per input path).
        # Per S41 measurement: output[0].text is the transcript string. Older NeMo versions
        # returned a list[str] directly — handle both for defensive compatibility.
        first = output[0] if output else None
        if first is None:
            text = ""
        elif isinstance(first, str):
            text = first
        else:
            text = getattr(first, "text", "") or ""

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        audio_ms = self._audio_duration_ms(audio_pcm, sample_rate)
        return TranscriptionResult(
            text=text.strip(),
            language="en",
            confidence=1.0,
            engine_name="parakeet-tdt",
            segments=(),
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
        """Stream PCM chunks; yield one result per ~1s buffered window. sample_rate=16000."""
        if sample_rate != 16000:
            raise ValueError(f"ParakeetTdtEngine requires sample_rate=16000, got {sample_rate}")
        buf = bytearray()
        async for chunk in audio_chunks:
            buf.extend(chunk)
            if len(buf) >= sample_rate * 2:  # ~1s of 16-bit mono audio
                yield await self.transcribe(bytes(buf), sample_rate, language)
                buf.clear()
        if buf:
            yield await self.transcribe(bytes(buf), sample_rate, language)
