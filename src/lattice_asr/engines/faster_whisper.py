"""FasterWhisperEngine — universal CPU/CUDA Whisper. Spec §6.3."""

from __future__ import annotations

import asyncio
import io
import logging
import math
import time
import wave
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from lattice_asr.engines.base import LANGUAGE_DETECT_SECONDS, TranscriptionEngine
from lattice_asr.lid import LidResult
from lattice_asr.types import EngineCapabilities, Segment, TranscriptionResult

if TYPE_CHECKING:
    from faster_whisper import WhisperModel  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# The tokenizer's language list is the CEILING of what a Whisper checkpoint can emit -- it is NOT
# what any given checkpoint was trained to do. Reading it as capability is the bug this comment
# replaces: every route defaulted to `distil-large-v3`, an ENGLISH-ONLY distilled model, while
# declaring all 99 tokenizer languages. MEASURED 2026-09-01 on 30s of Spanish (link, CUDA):
#
#     distil-large-v3 -> detect en@0.9463, and TRANSLATES: "family and have the capacity
#                        financial for that... A gringo here that I've been in Spanish"
#     large-v3        -> detect es@0.8779, transcribes Spanish correctly
#     medium          -> detect es@0.9468, transcribes Spanish correctly
#
# So the multilingual route was served by a model that confidently reports English for everything
# and translates rather than transcribes -- the exact defect, one layer down, and it survived the
# first pass of this fix. See ENGLISH_ONLY_MODEL_MARKERS below.
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


# A checkpoint is English-only if its NAME says so. Both families are English-only by
# construction, not by configuration:
#   ".en" suffix  -- OpenAI's tiny.en / base.en / small.en / medium.en
#   "distil-"     -- Distil-Whisper's distilled checkpoints (distil-large-v3, distil-small.en, ...)
# Name-based because the checkpoint does not advertise its training coverage anywhere readable, and
# guessing WIDE is the dangerous direction: an over-declared engine gets handed audio it will
# silently mistranslate. Pass `languages=` explicitly for a checkpoint this misjudges.
ENGLISH_ONLY_MODEL_MARKERS = ("distil-",)


def _languages_for_model(model: str) -> frozenset[str]:
    """Declared language coverage for a checkpoint name. Narrow when unsure."""
    name = model.rsplit("/", 1)[-1].lower()
    if name.endswith(".en") or any(m in name for m in ENGLISH_ONLY_MODEL_MARKERS):
        return frozenset({"en"})
    return _WHISPER_LANGS


class FasterWhisperEngine(TranscriptionEngine):
    """Adapter for SYSTRAN faster-whisper. Multilingual; CPU + CUDA."""

    required_packages = ("faster_whisper",)

    def __init__(
        self,
        *,
        model: str = "distil-large-v3",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
        languages: frozenset[str] | None = None,
        cpu_fallback_model: str | None = None,
    ):
        self._cpu_fallback_model = cpu_fallback_model
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._model: WhisperModel | None = None  # lazy-loaded
        self.capabilities = EngineCapabilities(
            name="faster-whisper",
            languages=languages if languages is not None else _languages_for_model(model),
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

            try:
                model = WhisperModel(
                    self._model_name,
                    device=self._device,
                    compute_type=self._compute_type,
                )
                if self._device == "cuda" and self._cpu_fallback_model is not None:
                    # CTranslate2 loads cuDNN / cuBLAS kernels at the FIRST
                    # inference, not at load: an unsupported card, a missing
                    # cuDNN sub-DLL or an OOM only shows up here. Probe one second
                    # of silence inside the same guard so the fallback sees it.
                    import numpy as np

                    segments, _info = model.transcribe(
                        np.zeros(16000, dtype=np.float32), language="en", beam_size=1
                    )
                    list(segments)
                self._model = model
            except Exception as exc:
                if self._device != "cuda" or self._cpu_fallback_model is None:
                    raise
                # A GPU that is present but unusable (runtime DLLs missing,
                # driver too old, too little VRAM for the model) must degrade to
                # the CPU tier, loudly, rather than leave the host with no ASR.
                logger.warning(
                    "faster-whisper: CUDA load of %r FAILED (%s: %s); falling back to CPU %r int8",
                    self._model_name,
                    type(exc).__name__,
                    exc,
                    self._cpu_fallback_model,
                )
                self._model_name = self._cpu_fallback_model
                self._device = "cpu"
                self._compute_type = "int8"
                self.capabilities = replace(
                    self.capabilities,
                    languages=_languages_for_model(self._model_name),
                    requires_gpu=False,
                    typical_rtfx=2.0,
                )
                self._model = WhisperModel(self._model_name, device="cpu", compute_type="int8")
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

    @staticmethod
    def _to_float_mono(audio_pcm: bytes) -> Any:
        """PCM or WAV bytes -> the 1D float32 array `detect_language()` requires.

        `detect_language()` does NOT accept the file-like object `transcribe()`
        takes -- it indexes its argument directly, so a BytesIO raises
        `TypeError: '_io.BytesIO' object is not subscriptable`. Converting the
        int16 PCM here rather than via `faster_whisper.audio.decode_audio` keeps
        this off the av/ffmpeg path for audio we already know the format of.
        """
        import numpy as np

        if audio_pcm[:4] == b"RIFF":
            with wave.open(io.BytesIO(audio_pcm), "rb") as w:
                pcm = w.readframes(w.getnframes())
        else:
            pcm = audio_pcm
        if len(pcm) % 2:  # a truncated final sample would misalign the whole array
            pcm = pcm[:-1]
        return np.frombuffer(pcm, dtype=np.int16).astype("float32") / 32768.0

    @staticmethod
    def _leading_slice(audio_pcm: bytes, sample_rate: int, seconds: float) -> bytes:
        """Return at most `seconds` of leading audio, preserving PCM-vs-WAV form."""
        max_bytes = int(sample_rate * 2 * seconds)
        if audio_pcm[:4] != b"RIFF":
            return audio_pcm[:max_bytes]
        with wave.open(io.BytesIO(audio_pcm), "rb") as w:
            frames = w.readframes(min(w.getnframes(), int(w.getframerate() * seconds)))
            rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as out:
            out.setnchannels(channels)
            out.setsampwidth(width)
            out.setframerate(rate)
            out.writeframes(frames)
        return buf.getvalue()

    async def detect_language(self, audio_pcm: bytes, sample_rate: int) -> LidResult | None:
        """Identify the language using faster-whisper's OWN detector — no Silero.

        This is what replaced ``SileroLid``. Whisper's encoder already produces a
        language distribution, so detection costs one forward pass over a bounded
        leading slice and needs no second model and no ``torch.hub`` download.

        Two code paths because ``WhisperModel.detect_language()`` is only present
        in newer faster-whisper releases; the ``transcribe()`` fallback reads the
        same ``info.language`` off an API that has always existed. Both are
        bounded to ``LANGUAGE_DETECT_SECONDS``.
        """
        if sample_rate != 16000:
            raise ValueError(f"FasterWhisperEngine requires sample_rate=16000, got {sample_rate}")
        model = await asyncio.to_thread(self._ensure_model)
        head = self._leading_slice(audio_pcm, sample_rate, LANGUAGE_DETECT_SECONDS)

        def _run() -> LidResult:
            native = getattr(model, "detect_language", None)
            if native is not None:
                lang, prob, *_ = native(self._to_float_mono(head))
                return LidResult(language=str(lang), confidence=float(prob))
            # Older faster-whisper: no detect_language(). Read the language off a
            # normal auto-detect pass instead -- same encoder decision, just with
            # decoding we discard. This path DOES take a file-like object.
            _segments, info = model.transcribe(
                self._pcm_bytes_to_wav(head, sample_rate), language=None, beam_size=1
            )
            return LidResult(
                language=str(info.language), confidence=float(info.language_probability)
            )

        return await asyncio.to_thread(_run)

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
