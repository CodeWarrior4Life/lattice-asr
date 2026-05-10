"""Silero LID — language detection on first 1.5s. Spec §8."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class LidResult:
    language: str
    confidence: float


class SileroLid:
    """Wraps Silero language-id model. CPU; <50ms on 1.5s audio."""

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
            tensor = torch.from_numpy(arr)
            languages, lang_probs = model(tensor, top_n=1)  # type: ignore[no-untyped-call]
            lang = str(languages[0])[:2]
            conf = float(lang_probs[0])
            return LidResult(language=lang, confidence=conf)

        return await asyncio.to_thread(_run)
