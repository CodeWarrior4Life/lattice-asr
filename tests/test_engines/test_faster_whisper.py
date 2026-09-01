import pytest

from lattice_asr.engines.faster_whisper import FasterWhisperEngine


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_capabilities():
    """Coverage follows the CHECKPOINT, not faster-whisper's tokenizer.

    This test asserted `"es" in distil-large-v3`'s languages until 2026-09-01.
    That claim was false and load-bearing: it is why the multilingual route was
    served by an English-only model which MEASURED en@0.9463 on Spanish audio and
    translated it. A distilled checkpoint declares English only.
    """
    distil = FasterWhisperEngine(model="distil-large-v3", device="cpu", compute_type="int8")
    assert distil.capabilities.name == "faster-whisper"
    assert distil.capabilities.languages == frozenset({"en"})
    assert distil.capabilities.streaming is True

    multilingual = FasterWhisperEngine(model="medium", device="cpu", compute_type="int8")
    assert "en" in multilingual.capabilities.languages
    assert "es" in multilingual.capabilities.languages
    assert multilingual.capabilities.streaming is True


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
    assert result.language == "en"


# AUTHORIZED DEVIATION — warmup() loads ~600 MB model, violates r_tier FAST contract
@pytest.mark.s_tier
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
