import base64
import json
import pytest
import httpx
from lattice_asr.engines.remote import RemoteEngine, RemoteEngineError


def _ok_handler(req: httpx.Request) -> httpx.Response:
    body = json.loads(req.content)
    audio_len = len(base64.b64decode(body["audio_b64"]))
    return httpx.Response(
        200,
        json={
            "text": "hello",
            "language": body.get("language") or "en",
            "confidence": 0.95,
            "engine_name": "remote:morpheus",
            "segments": [],
            "speaker_segments": [],
            "audio_duration_ms": audio_len // 32,
            "duration_ms": 100,
        },
        headers={"X-Lattice-Asr-Api-Version": "v1"},
    )


def _500_then_ok_handler(req: httpx.Request) -> httpx.Response:
    if not hasattr(_500_then_ok_handler, "n"):
        _500_then_ok_handler.n = 0  # type: ignore[attr-defined]
    _500_then_ok_handler.n += 1  # type: ignore[attr-defined]
    if _500_then_ok_handler.n == 1:  # type: ignore[attr-defined]
        return httpx.Response(503, text="upstream busy")
    return _ok_handler(req)


def _version_mismatch_handler(req):
    return httpx.Response(
        200,
        json={
            "text": "x",
            "language": "en",
            "confidence": 0.5,
            "engine_name": "remote:m",
            "segments": [],
            "speaker_segments": [],
            "audio_duration_ms": 0,
            "duration_ms": 0,
        },
        headers={"X-Lattice-Asr-Api-Version": "v2"},
    )


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_capabilities():
    eng = RemoteEngine(url="http://x")
    assert eng.capabilities.name == "remote"
    assert eng.capabilities.requires_gpu is False


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_round_trip_serialization():
    eng = RemoteEngine(url="http://m", _transport=httpx.MockTransport(_ok_handler))
    r = await eng.transcribe(b"\x00\x01" * 16000, sample_rate=16000, language="en")
    assert r.text == "hello"
    assert r.engine_name == "remote:morpheus"


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_503_retries_then_succeeds():
    _500_then_ok_handler.n = 0  # type: ignore[attr-defined]
    eng = RemoteEngine(url="http://m", _transport=httpx.MockTransport(_500_then_ok_handler))
    r = await eng.transcribe(b"\x00", sample_rate=16000, language="en")
    assert r.text == "hello"
    assert _500_then_ok_handler.n == 2  # type: ignore[attr-defined]


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_4xx_raises():
    def h(req):
        return httpx.Response(400, text="bad")

    eng = RemoteEngine(url="http://m", _transport=httpx.MockTransport(h))
    with pytest.raises(RemoteEngineError):
        await eng.transcribe(b"\x00", sample_rate=16000, language="en")


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_version_mismatch_raises():
    eng = RemoteEngine(url="http://m", _transport=httpx.MockTransport(_version_mismatch_handler))
    with pytest.raises(RemoteEngineError, match="version"):
        await eng.transcribe(b"\x00", sample_rate=16000, language="en")


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_api_key_in_authorization_header():
    captured = {}

    def h(req):
        captured["auth"] = req.headers.get("Authorization")
        return _ok_handler(req)

    eng = RemoteEngine(url="http://m", api_key="sek", _transport=httpx.MockTransport(h))
    await eng.transcribe(b"\x00", sample_rate=16000, language="en")
    assert captured["auth"] == "Bearer sek"


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_exhausted_retries_raises():
    """All attempts return 5xx → RemoteEngineError with 'exhausted' message."""
    call_count = {"n": 0}

    def h(req):
        call_count["n"] += 1
        return httpx.Response(503, text="upstream busy")

    eng = RemoteEngine(url="http://m", _transport=httpx.MockTransport(h))
    with pytest.raises(RemoteEngineError, match="exhausted"):
        await eng.transcribe(b"\x00", sample_rate=16000, language="en")
    assert call_count["n"] == 3  # exactly _RETRIES attempts before giving up


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_connection_error_retries_then_raises():
    """httpx.ConnectError on every attempt → retried then RemoteEngineError."""
    call_count = {"n": 0}

    def h(req):
        call_count["n"] += 1
        raise httpx.ConnectError("connection refused")

    eng = RemoteEngine(url="http://m", _transport=httpx.MockTransport(h))
    with pytest.raises(RemoteEngineError, match="exhausted"):
        await eng.transcribe(b"\x00", sample_rate=16000, language="en")
    assert call_count["n"] == 3  # exactly _RETRIES attempts


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_transcribe_streaming_buffers_to_one_second_then_flushes_partial():
    """Streaming buffers to 1s of int16 PCM, yields once at threshold, flushes final partial.

    Three 0.5s chunks → buffer hits 1s after chunk 2 (yield #1) →
    chunk 3 sits in buffer → stream end flushes final partial (yield #2).
    """
    call_count = {"n": 0}

    def h(req):
        call_count["n"] += 1
        return _ok_handler(req)

    eng = RemoteEngine(url="http://m", _transport=httpx.MockTransport(h))

    async def chunks():
        # 16000 bytes = 8000 int16 frames = 0.5s at 16kHz
        yield b"\x00\x01" * 8000
        yield b"\x00\x02" * 8000
        yield b"\x00\x03" * 8000

    results = [
        r async for r in eng.transcribe_streaming(chunks(), sample_rate=16000, language="en")
    ]
    assert len(results) == 2  # one full-second yield + one final-partial yield
    assert all(r.text == "hello" for r in results)
    assert call_count["n"] == 2  # one POST per yield
