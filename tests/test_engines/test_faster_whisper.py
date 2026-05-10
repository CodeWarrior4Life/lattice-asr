import pytest
from lattice_asr.engines.faster_whisper import FasterWhisperEngine


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_capabilities():
    eng = FasterWhisperEngine(model="distil-large-v3", device="cpu", compute_type="int8")
    assert eng.capabilities.name == "faster-whisper"
    assert "en" in eng.capabilities.languages
    assert "es" in eng.capabilities.languages
    assert eng.capabilities.streaming is True


@pytest.mark.s_tier
@pytest.mark.asyncio
async def test_transcribe_english(hello_en_2s_wav):
    eng = FasterWhisperEngine(model="distil-large-v3", device="cpu", compute_type="int8")
    audio = hello_en_2s_wav.read_bytes()
    result = await eng.transcribe(audio, sample_rate=16000, language="en")
    assert result.engine_name == "faster-whisper"
    assert result.language == "en"
    assert result.text.strip() != ""
    assert result.audio_duration_ms > 0
    assert result.duration_ms > 0


@pytest.mark.s_tier
@pytest.mark.asyncio
async def test_transcribe_auto_language(hello_en_2s_wav):
    eng = FasterWhisperEngine(model="distil-large-v3", device="cpu", compute_type="int8")
    audio = hello_en_2s_wav.read_bytes()
    result = await eng.transcribe(audio, sample_rate=16000, language=None)
    assert result.language in ("en",)


@pytest.mark.s_tier  # AUTHORIZED DEVIATION — warmup() loads ~600 MB model, violates r_tier FAST contract
@pytest.mark.asyncio
async def test_warmup_does_not_raise():
    eng = FasterWhisperEngine(model="distil-large-v3", device="cpu", compute_type="int8")
    await eng.warmup()


@pytest.mark.s_tier
@pytest.mark.asyncio
async def test_transcribe_emits_segments(hello_en_2s_wav):
    eng = FasterWhisperEngine(model="distil-large-v3", device="cpu", compute_type="int8")
    audio = hello_en_2s_wav.read_bytes()
    result = await eng.transcribe(audio, sample_rate=16000, language="en")
    assert isinstance(result.segments, tuple)
    if result.segments:
        s = result.segments[0]
        assert s.start_ms >= 0
        assert s.end_ms > s.start_ms


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_invalid_sample_rate_raises():
    eng = FasterWhisperEngine(model="distil-large-v3", device="cpu", compute_type="int8")
    with pytest.raises(ValueError):
        await eng.transcribe(b"\x00\x00", sample_rate=8000, language="en")
