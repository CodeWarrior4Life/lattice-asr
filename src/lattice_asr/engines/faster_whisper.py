"""FasterWhisperEngine — universal CPU/CUDA Whisper. Spec §6.3."""

from __future__ import annotations

import asyncio
import io
import math
import time
import wave
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from lattice_asr.engines.base import TranscriptionEngine
from lattice_asr.types import EngineCapabilities, Segment, TranscriptionResult

if TYPE_CHECKING:
    from faster_whisper import WhisperModel  # type: ignore[import-untyped]

# Spec §6.3: capabilities.languages = frozenset(WHISPER_SUPPORTED_LANGUAGES) (99 languages).
# Dynamic import so the frozenset reflects faster-whisper's actual tokenizer list rather than
# a hand-curated subset (the static fallback below is a conservative approximation only).
# NOTE: importing faster_whisper.tokenizer eagerly loads ctranslate2 (~238ms cold).
# WhisperModel itself remains lazy (loaded inside _ensure_model on first call).
try:
    from faster_whisper.tokenizer import (
        _LANGUAGE_CODES as _FW_LANG_CODES,  # type: ignore[import-untyped]
    )

    _WHISPER_LANGS: frozenset[str] = frozenset(_FW_LANG_CODES)
except (ImportError, AttributeError):
    # Fallback: faster-whisper not installed; capabilities still inspectable.
    # This list is a curated subset — verify against _LANGUAGE_CODES when the package is present.
    _WHISPER_LANGS = frozenset(
        {
            "af",
            "am",
            "ar",
            "az",
            "be",
            "bg",
            "bn",
            "br",
            "bs",
            "ca",
            "cs",
            "cy",
            "da",
            "de",
            "el",
            "en",
            "es",
            "et",
            "eu",
            "fa",
            "fi",
            "fo",
            "fr",
            "gl",
            "gu",
            "he",
            "hi",
            "hr",
            "hu",
            "hy",
            "id",
            "is",
            "it",
            "ja",
            "ka",
            "km",
            "kn",
            "ko",
            "lb",
            "lo",
            "lt",
            "lv",
            "mi",
            "mk",
            "ml",
            "mr",
            "ms",
            "mt",
            "my",
            "ne",
            "nl",
            "no",
            "pa",
            "pl",
            "pt",
            "ro",
            "ru",
            "si",
            "sk",
            "sl",
            "sq",
            "sr",
            "sv",
            "sw",
            "ta",
            "te",
            "tg",
            "th",
            "tk",
            "tl",
            "tr",
            "uk",
            "ur",
            "uz",
            "vi",
            "yi",
            "yue",
            "zh",
        }
    )


class FasterWhisperEngine(TranscriptionEngine):
    """Adapter for SYSTRAN faster-whisper. Multilingual; CPU + CUDA."""

    def __init__(
        self,
        *,
        model: str = "distil-large-v3",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
    ):
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._model: WhisperModel | None = None  # lazy-loaded
        self.capabilities = EngineCapabilities(
            name="faster-whisper",
            languages=_WHISPER_LANGS,
            streaming=True,
            requires_gpu=device == "cuda",
            requires_apple_silicon=False,
            typical_rtfx=2.0 if device == "cpu" else 30.0,
        )

    async def warmup(self) -> None:
        await asyncio.to_thread(self._ensure_model)

    def _ensure_model(self) -> WhisperModel:
        if self._model is None:
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]

            self._model = WhisperModel(
                self._model_name,
                device=self._device,
                compute_type=self._compute_type,
            )
        return self._model

    @staticmethod
    def _pcm_bytes_to_wav(audio_pcm: bytes, sample_rate: int) -> io.BytesIO:
        if audio_pcm[:4] == b"RIFF":
            return io.BytesIO(audio_pcm)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(audio_pcm)
        buf.seek(0)
        return buf

    async def transcribe(
        self,
        audio_pcm: bytes,
        sample_rate: int,
        language: str | None,
    ) -> TranscriptionResult:
        """Transcribe raw PCM or WAV bytes. Requires sample_rate=16000."""
        if sample_rate != 16000:
            raise ValueError(f"FasterWhisperEngine requires sample_rate=16000, got {sample_rate}")
        t0 = time.monotonic()
        model = await asyncio.to_thread(self._ensure_model)
        wav = self._pcm_bytes_to_wav(audio_pcm, sample_rate)

        def _run():
            segments, info = model.transcribe(
                wav,
                language=language,
                beam_size=self._beam_size,
            )
            return list(segments), info

        seg_list, info = await asyncio.to_thread(_run)
        text = " ".join(s.text.strip() for s in seg_list).strip()
        segs = tuple(
            Segment(
                text=s.text.strip(),
                start_ms=int(s.start * 1000),
                end_ms=int(s.end * 1000),
                confidence=float(math.exp(s.avg_logprob)),
            )
            for s in seg_list
        )
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        return TranscriptionResult(
            text=text,
            language=info.language,
            confidence=float(info.language_probability),
            engine_name="faster-whisper",
            segments=segs,
            speaker_segments=(),
            audio_duration_ms=int(info.duration * 1000),
            duration_ms=elapsed_ms,
        )

    async def transcribe_streaming(
        self,
        audio_chunks: AsyncIterator[bytes],
        sample_rate: int,
        language: str | None,
    ) -> AsyncIterator[TranscriptionResult]:
        """Transcribe a streaming audio source.

        Requires sample_rate=16000. Yields one result per buffered window.
        """
        if sample_rate != 16000:
            raise ValueError(f"FasterWhisperEngine requires sample_rate=16000, got {sample_rate}")
        buf = bytearray()
        async for chunk in audio_chunks:
            buf.extend(chunk)
            if len(buf) >= sample_rate * 2:  # ~1s of 16-bit mono audio
                yield await self.transcribe(bytes(buf), sample_rate, language)
                buf.clear()
        if buf:
            yield await self.transcribe(bytes(buf), sample_rate, language)
