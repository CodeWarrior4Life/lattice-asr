from unittest.mock import AsyncMock, patch

import httpx
import pytest

from lattice_asr.engines.remote import RemoteEngine
from lattice_asr.types import TranscriptionResult


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_remote_engine_against_in_process_server():
    from lattice_asr_server.app import create_app

    app = create_app()

    fake = TranscriptionResult(
        text="round-trip",
        language="en",
        confidence=0.9,
        engine_name="remote:test",
        segments=(),
        speaker_segments=(),
        audio_duration_ms=100,
        duration_ms=20,
    )

    transport = httpx.ASGITransport(app=app)
    with patch("lattice_asr_server.app._get_transcriber") as mock_t:
        t = mock_t.return_value
        t.transcribe = AsyncMock(return_value=fake)
        eng = RemoteEngine(url="http://server", _transport=transport)
        result = await eng.transcribe(b"\x00\x01" * 100, sample_rate=16000, language="en")

    assert result.text == "round-trip"
    assert result.engine_name == "remote:test"
