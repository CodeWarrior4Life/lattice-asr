# src/lattice_asr/lid.py
"""Silero LID — language detection on first 1.5s. Spec §8."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

# Spec §8 latency contract: Silero LID is bounded to the first 1.5s of audio
# so detect() latency is ≤50ms regardless of caller input length.
LID_AUDIO_SECONDS = 1.5


@dataclass(frozen=True)
class LidResult:
    language: str
    confidence: float


class SileroLid:
    """Wraps Silero language-id model. CPU; <50ms on 1.5s audio.

    Call ``warmup()`` once at startup; concurrent ``detect()`` calls before
    warmup completes will redundantly re-trigger ``torch.hub.load``.
    """

    def __init__(self) -> None:
        self._model: Optional[Any] = None

    async def warmup(self) -> None:
        await asyncio.to_thread(self._ensure_model)

    def _ensure_model(self) -> Any:
        if self._model is None:
            import torch  # type: ignore[import-not-found]

            self._model, _utils = torch.hub.load(  # type: ignore[no-untyped-call]
                "snakers4/silero-vad", "silero_lang_detector_95"
            )
        return self._model

    async def detect(self, audio_pcm: bytes, sample_rate: int) -> LidResult:
        if sample_rate != 16000:
            raise ValueError("SileroLid requires sample_rate=16000")

        def _run() -> LidResult:
            import io
            import wave

            import numpy as np
            import torch  # type: ignore[import-not-found]

            model = self._ensure_model()
            if audio_pcm[:4] == b"RIFF":
                with wave.open(io.BytesIO(audio_pcm)) as w:
                    pcm = w.readframes(w.getnframes())
            else:
                pcm = audio_pcm
            arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            # Spec §8: bound input to LID_AUDIO_SECONDS for latency target.
            max_samples = int(sample_rate * LID_AUDIO_SECONDS)
            arr = arr[:max_samples]
            tensor = torch.from_numpy(arr)
            languages, lang_probs = model(tensor, top_n=1)  # type: ignore[no-untyped-call]
            lang = str(languages[0]).split(",")[0][:2]
            conf = float(lang_probs[0])
            return LidResult(language=lang, confidence=conf)

        return await asyncio.to_thread(_run)
