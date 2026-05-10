import pytest

from lattice_asr.lid import LidResult, SileroLid


@pytest.mark.r_tier
def test_lid_result_frozen():
    r = LidResult(language="en", confidence=0.95)
    with pytest.raises((AttributeError, TypeError)):
        r.language = "es"  # type: ignore[misc]


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_lid_constructor_does_not_load_model():
    lid = SileroLid()
    assert lid._model is None  # lazy


@pytest.mark.s_tier
@pytest.mark.asyncio
async def test_lid_detects_english(hello_en_2s_wav):
    lid = SileroLid()
    # Pass the full WAV; SileroLid trims to LID_AUDIO_SECONDS (1.5s) internally
    # per spec §8 latency contract.
    audio = hello_en_2s_wav.read_bytes()
    result = await lid.detect(audio, sample_rate=16000)
    assert result.language == "en"
    assert result.confidence > 0.5
