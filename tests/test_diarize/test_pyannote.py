import pytest

from lattice_asr.diarize import Diarizer
from lattice_asr.diarize.pyannote import PyAnnoteAdapter


@pytest.mark.r_tier
def test_pyannote_adapter_constructs_without_loading_model():
    adapter = PyAnnoteAdapter(auth_token="fake")
    assert adapter._pipeline is None  # lazy


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_pyannote_raises_clearly_when_no_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    adapter = PyAnnoteAdapter()  # no auth_token
    with pytest.raises(ValueError, match="HF_TOKEN"):
        await adapter.diarize(b"\x00", sample_rate=16000)


@pytest.mark.r_tier
def test_pyannote_inherits_from_diarizer():
    assert issubclass(PyAnnoteAdapter, Diarizer)
