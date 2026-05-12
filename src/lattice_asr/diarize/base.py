"""Diarizer ABC + segment-merge helper. Spec §7."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from lattice_asr.types import Segment, SpeakerSegment


class Diarizer(ABC):
    @abstractmethod
    async def diarize(self, audio_pcm: bytes, sample_rate: int) -> list[SpeakerSegment]: ...


def merge_segments_with_text(
    speaker_segments: Sequence[SpeakerSegment], transcription_segments: Sequence[Segment]
) -> list[SpeakerSegment]:
    """Spec §7.3: merge transcription text into speaker timeline by midpoint.

    Each transcription segment is assigned to the FIRST speaker (in input order)
    whose [start_ms, end_ms] range covers the transcription midpoint. The earlier
    speaker wins ties on shared boundaries (so adjacent A=[0,3000], B=[3000,6000]
    with mid=3000 goes to A only); this also prevents double-counting when
    diarizer outputs overlap.
    """
    assigned = [False] * len(transcription_segments)
    out: list[SpeakerSegment] = []
    for sp in speaker_segments:
        joined = []
        for i, tx in enumerate(transcription_segments):
            if assigned[i]:
                continue
            mid = (tx.start_ms + tx.end_ms) // 2
            if sp.start_ms <= mid <= sp.end_ms:
                joined.append(tx.text.strip())
                assigned[i] = True
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
