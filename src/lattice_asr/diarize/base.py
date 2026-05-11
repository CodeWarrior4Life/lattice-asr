"""Diarizer ABC + segment-merge helper. Spec §7."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from lattice_asr.types import Segment, SpeakerSegment


class Diarizer(ABC):
    @abstractmethod
    async def diarize(self, audio_pcm: bytes, sample_rate: int) -> list[SpeakerSegment]: ...


def merge_segments_with_text(
    speaker_segments: Sequence[SpeakerSegment], transcription_segments: Sequence[Segment]
) -> list[SpeakerSegment]:
    """Spec §7.3: merge transcription text into speaker timeline by midpoint.

    For each speaker segment, gather transcription segments whose midpoint falls
    within [start_ms, end_ms]; concatenate text.
    """
    out: list[SpeakerSegment] = []
    for sp in speaker_segments:
        joined = []
        for tx in transcription_segments:
            mid = (tx.start_ms + tx.end_ms) // 2
            if sp.start_ms <= mid <= sp.end_ms:
                joined.append(tx.text.strip())
        text = " ".join(t for t in joined if t).strip()
        out.append(
            SpeakerSegment(
                label=sp.label,
                start_ms=sp.start_ms,
                end_ms=sp.end_ms,
                text=text,
                voice_print_id=sp.voice_print_id,
            )
        )
    return out
