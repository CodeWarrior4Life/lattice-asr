import pytest

from lattice_asr.diarize.sortformer import NvidiaSortformerAdapter


@pytest.mark.r_tier
def test_sortformer_lazy_init():
    adapter = NvidiaSortformerAdapter()
    assert adapter._model is None


@pytest.mark.s_tier
@pytest.mark.nvidia_cuda
@pytest.mark.asyncio
async def test_sortformer_diarizes(hello_en_2s_wav):
    adapter = NvidiaSortformerAdapter()
    audio = hello_en_2s_wav.read_bytes()
    segs = await adapter.diarize(audio, sample_rate=16000)
    assert len(segs) >= 1
