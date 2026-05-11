"""PyAnnoteAdapter — pyannote.audio. Spec §7.1."""

from __future__ import annotations

import asyncio
import io
import os
import wave

from lattice_asr.diarize.base import Diarizer
from lattice_asr.types import SpeakerSegment


class PyAnnoteAdapter(Diarizer):
    def __init__(
        self,
        *,
        model: str = "pyannote/speaker-diarization-3.1",
        device: str = "auto",
        auth_token: str | None = None,
    ):
        self._model_name = model
        self._device = device
        self._auth_token = auth_token or os.environ.get("HF_TOKEN")
        self._pipeline = None

    async def warmup(self) -> None:
        await asyncio.to_thread(self._ensure_pipeline)

    def _ensure_pipeline(self):
        if self._pipeline is None:
            if not self._auth_token:
                raise ValueError(
                    "PyAnnoteAdapter requires HF_TOKEN env var or auth_token kwarg "
                    "(model is gated on HuggingFace)"
                )
            from pyannote.audio import Pipeline  # type: ignore[import-not-found]

            self._pipeline = Pipeline.from_pretrained(  # type: ignore[no-untyped-call]
                self._model_name, use_auth_token=self._auth_token
            )
        return self._pipeline

    async def diarize(self, audio_pcm: bytes, sample_rate: int) -> list[SpeakerSegment]:
        if not self._auth_token:
            raise ValueError("PyAnnoteAdapter requires HF_TOKEN env var or auth_token kwarg")

        def _run():
            pipeline = self._ensure_pipeline()
            if audio_pcm[:4] == b"RIFF":
                wav_bytes = audio_pcm
            else:
                buf = io.BytesIO()
                with wave.open(buf, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(sample_rate)
                    w.writeframes(audio_pcm)
                wav_bytes = buf.getvalue()
            import tempfile
            from pathlib import Path

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                Path(f.name).write_bytes(wav_bytes)
                annotation = pipeline(f.name)  # type: ignore[union-attr,operator]
            segments: list[SpeakerSegment] = []
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                segments.append(
                    SpeakerSegment(
                        label=str(speaker),
                        start_ms=int(turn.start * 1000),
                        end_ms=int(turn.end * 1000),
                        text="",
                    )
                )
            return segments

        return await asyncio.to_thread(_run)
