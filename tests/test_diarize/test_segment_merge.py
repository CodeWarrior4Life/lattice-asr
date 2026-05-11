import pytest

from lattice_asr.diarize import merge_segments_with_text
from lattice_asr.types import Segment, SpeakerSegment


@pytest.mark.r_tier
def test_single_speaker_takes_all_segments():
    speakers = [SpeakerSegment(label="S1", start_ms=0, end_ms=10_000, text="")]
    txs = [Segment(text="hello world", start_ms=100, end_ms=2000, confidence=0.9)]
    out = merge_segments_with_text(speakers, txs)
    assert out[0].text == "hello world"


@pytest.mark.r_tier
def test_two_speakers_split_by_midpoint():
    speakers = [
        SpeakerSegment(label="A", start_ms=0, end_ms=5000, text=""),
        SpeakerSegment(label="B", start_ms=5000, end_ms=10000, text=""),
    ]
    txs = [
        Segment(text="hi", start_ms=1000, end_ms=2000, confidence=0.9),
        Segment(text="bye", start_ms=7000, end_ms=8000, confidence=0.9),
    ]
    out = merge_segments_with_text(speakers, txs)
    assert out[0].text == "hi"
    assert out[1].text == "bye"


@pytest.mark.r_tier
def test_speaker_with_no_overlap_emits_empty_text():
    speakers = [SpeakerSegment(label="A", start_ms=0, end_ms=1000, text="")]
    txs = [Segment(text="far", start_ms=5000, end_ms=6000, confidence=0.9)]
    out = merge_segments_with_text(speakers, txs)
    assert out[0].text == ""


@pytest.mark.r_tier
def test_overlapping_transcription_assigned_by_midpoint():
    speakers = [
        SpeakerSegment(label="A", start_ms=0, end_ms=3000, text=""),
        SpeakerSegment(label="B", start_ms=3000, end_ms=6000, text=""),
    ]
    txs = [Segment(text="straddle", start_ms=2000, end_ms=4000, confidence=0.9)]
    out = merge_segments_with_text(speakers, txs)
    assert out[0].text == "straddle"
    assert out[1].text == ""


@pytest.mark.r_tier
def test_three_speakers_concatenate_multiple_texts():
    speakers = [
        SpeakerSegment(label="S1", start_ms=0, end_ms=5000, text=""),
    ]
    txs = [
        Segment(text="one", start_ms=500, end_ms=1500, confidence=0.9),
        Segment(text="two", start_ms=2000, end_ms=3000, confidence=0.9),
    ]
    out = merge_segments_with_text(speakers, txs)
    assert out[0].text == "one two"


@pytest.mark.r_tier
def test_empty_inputs_return_empty():
    assert merge_segments_with_text([], []) == []
    assert merge_segments_with_text([], [Segment("a", 0, 1, 0.9)]) == []
