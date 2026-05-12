import pytest

from lattice_asr import (
    EngineCapabilities,
    Segment,
    SpeakerSegment,
    TranscriptionResult,
)
from lattice_asr.engines import TranscriptionEngine


@pytest.mark.r_tier
def test_segment_is_frozen():
    s = Segment(text="hi", start_ms=0, end_ms=100, confidence=0.9)
    with pytest.raises((AttributeError, TypeError)):
        s.text = "bye"  # type: ignore[misc]


@pytest.mark.r_tier
def test_speaker_segment_voice_print_id_optional():
    s = SpeakerSegment(label="Speaker 1", start_ms=0, end_ms=1000, text="hello")
    assert s.voice_print_id is None


@pytest.mark.r_tier
def test_transcription_result_default_segments_empty():
    r = TranscriptionResult(
        text="hi",
        language="en",
        confidence=0.95,
        engine_name="test",
        segments=(),
        speaker_segments=(),
        audio_duration_ms=1000,
        duration_ms=50,
    )
    assert r.segments == ()
    assert r.speaker_segments == ()


@pytest.mark.r_tier
def test_engine_capabilities_immutable():
    c = EngineCapabilities(
        name="test",
        languages=frozenset({"en"}),
        streaming=True,
        requires_gpu=False,
        requires_apple_silicon=False,
        typical_rtfx=10.0,
    )
    with pytest.raises((AttributeError, TypeError)):
        c.name = "other"  # type: ignore[misc]


@pytest.mark.r_tier
def test_transcription_engine_is_abstract():
    with pytest.raises(TypeError):
        TranscriptionEngine()  # type: ignore[abstract]
