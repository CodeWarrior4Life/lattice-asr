from unittest.mock import patch, AsyncMock
import pytest
import lattice_asr
from lattice_asr.types import TranscriptionResult


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_module_transcribe_uses_singleton():
    fake = TranscriptionResult(
        text="hi",
        language="en",
        confidence=0.9,
        engine_name="faster-whisper",
        segments=(),
        speaker_segments=(),
        audio_duration_ms=100,
        duration_ms=20,
    )
    lattice_asr._reset_singleton()
    with patch("lattice_asr.Transcriber.transcribe", new=AsyncMock(return_value=fake)):
        r = await lattice_asr.transcribe(b"\x00", language="en")
    assert r.text == "hi"


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_module_transcribe_lazy_inits_once():
    lattice_asr._reset_singleton()
    fake = TranscriptionResult(
        text="hi",
        language="en",
        confidence=0.9,
        engine_name="faster-whisper",
        segments=(),
        speaker_segments=(),
        audio_duration_ms=100,
        duration_ms=20,
    )
    with patch("lattice_asr.Transcriber.transcribe", new=AsyncMock(return_value=fake)):
        await lattice_asr.transcribe(b"\x00", language="en")
        first = lattice_asr._singleton
        await lattice_asr.transcribe(b"\x00", language="en")
        second = lattice_asr._singleton
    assert first is second
