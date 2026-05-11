"""NvidiaSortformerAdapter — NVIDIA NeMo Sortformer. Spec §7.2."""

from __future__ import annotations

import asyncio

from lattice_asr.diarize.base import Diarizer
from lattice_asr.types import SpeakerSegment


class NvidiaSortformerAdapter(Diarizer):
    def __init__(self, *, model: str = "nvidia/sortformer-diarization-l24-30s"):
        self._model_name = model
        self._model = None

    async def warmup(self) -> None:
        await asyncio.to_thread(self._ensure_model)

    def _ensure_model(self):
        if self._model is None:
            import nemo.collections.asr as nemo_asr  # type: ignore[import-not-found]

            self._model = nemo_asr.models.SortformerEncLabelModel.from_pretrained(  # type: ignore
                self._model_name
            )
        return self._model

    async def diarize(self, audio_pcm: bytes, sample_rate: int) -> list[SpeakerSegment]:
        def _run():
            import tempfile
            import wave
            from pathlib import Path

            model = self._ensure_model()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = Path(f.name)
            try:
                if audio_pcm[:4] == b"RIFF":
                    wav_path.write_bytes(audio_pcm)
                else:
                    with wave.open(str(wav_path), "wb") as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)
                        w.setframerate(sample_rate)
                        w.writeframes(audio_pcm)
                preds = model.diarize(audio=[str(wav_path)])  # type: ignore[union-attr]
                segs: list[SpeakerSegment] = []
                for entry in preds[0]:
                    start_s, end_s, label = entry[0], entry[1], entry[2]
                    segs.append(
                        SpeakerSegment(
                            label=str(label),
                            start_ms=int(start_s * 1000),
                            end_ms=int(end_s * 1000),
                            text="",
                        )
                    )
                return segs
            finally:
                wav_path.unlink(missing_ok=True)

        return await asyncio.to_thread(_run)
