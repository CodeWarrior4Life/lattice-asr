"""FasterWhisperEngine — universal CPU/CUDA Whisper. Spec §6.3."""

from __future__ import annotations

import asyncio
import io
import time
import wave
from collections.abc import AsyncIterator

from lattice_asr.engines.base import TranscriptionEngine
from lattice_asr.types import EngineCapabilities, Segment, TranscriptionResult


_WHISPER_LANGS = frozenset(
    {
        "en",
        "es",
        "fr",
        "de",
        "it",
        "pt",
        "nl",
        "pl",
        "ru",
        "ja",
        "ko",
        "zh",
        "ar",
        "hi",
        "tr",
        "sv",
        "da",
        "no",
        "fi",
        "el",
        "he",
        "th",
        "vi",
        "id",
        "ms",
        "uk",
        "cs",
        "hu",
        "ro",
        "bg",
        "hr",
        "sk",
        "sl",
        "et",
        "lv",
        "lt",
        "fa",
        "ur",
        "bn",
        "ta",
        "te",
        "ml",
        "mr",
        "gu",
        "kn",
        "pa",
        "or",
        "si",
        "ne",
        "my",
        "km",
        "lo",
        "ka",
        "am",
        "sw",
        "zu",
        "xh",
        "af",
        "is",
        "ga",
        "cy",
        "mt",
        "eu",
        "ca",
        "gl",
        "lb",
        "fo",
        "yi",
        "yue",
        "wuu",
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
        self._model = None  # lazy-loaded
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

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]

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
                confidence=float(s.avg_logprob),
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
        buf = bytearray()
        async for chunk in audio_chunks:
            buf.extend(chunk)
            if len(buf) >= sample_rate * 2:  # ~1s of 16-bit mono audio
                yield await self.transcribe(bytes(buf), sample_rate, language)
                buf.clear()
        if buf:
            yield await self.transcribe(bytes(buf), sample_rate, language)
