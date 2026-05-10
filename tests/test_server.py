import base64
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from lattice_asr.types import TranscriptionResult


@pytest.fixture
def client():
    from lattice_asr_server.app import create_app

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
    app = create_app()
    with patch("lattice_asr_server.app._transcriber_singleton") as mock_t:
        mock_t.transcribe = AsyncMock(return_value=fake)
        yield TestClient(app)


@pytest.mark.r_tier
def test_health_endpoint(client):
    r = client.get("/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.r_tier
def test_transcribe_endpoint_returns_result(client):
    body = {
        "audio_b64": base64.b64encode(b"\x00\x01" * 1000).decode("ascii"),
        "sample_rate": 16000,
        "language": "en",
        "diarize": False,
    }
    r = client.post("/v1/transcribe", json=body)
    assert r.status_code == 200
    assert r.headers["X-Lattice-Asr-Api-Version"] == "v1"
    assert r.json()["text"] == "hi"


@pytest.mark.r_tier
def test_transcribe_rejects_missing_audio(client):
    r = client.post("/v1/transcribe", json={"sample_rate": 16000})
    assert r.status_code == 422


@pytest.mark.r_tier
def test_auth_required_when_api_key_set(monkeypatch):
    monkeypatch.setenv("LATTICE_ASR_API_KEY", "sek")
    from lattice_asr_server.app import create_app

    app = create_app()
    c = TestClient(app)
    body = {"audio_b64": base64.b64encode(b"\x00").decode(), "sample_rate": 16000}
    r = c.post("/v1/transcribe", json=body)
    assert r.status_code == 401
    r = c.post("/v1/transcribe", json=body, headers={"Authorization": "Bearer sek"})
    # 200 (if patched) or 500 (if real Transcriber init fails on this host) — either is non-401
    assert r.status_code != 401
