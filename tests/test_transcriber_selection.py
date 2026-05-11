"""Engine selection logic for Transcriber. Spec §5."""

import pytest

from lattice_asr.engines.faster_whisper import FasterWhisperEngine
from lattice_asr.hardware import HardwareProfile
from lattice_asr.transcriber import _build_engine_registry


def _hw(
    *,
    apple_silicon=False,
    nvidia_cuda=False,
    cuda_cap=None,
    os_="linux",
    arch="x86_64",
):
    return HardwareProfile(
        os=os_,
        cpu_arch=arch,
        apple_silicon=apple_silicon,
        nvidia_cuda=nvidia_cuda,
        cuda_capability=cuda_cap,
        total_ram_gb=16.0,
        cpu_cores=8,
    )


@pytest.mark.r_tier
def test_cpu_only_uses_faster_whisper_for_both_routes():
    reg = _build_engine_registry(_hw(), force=None)
    assert isinstance(reg["en"], FasterWhisperEngine)
    assert reg["en"] is reg["multi"]  # single engine instance covers both


@pytest.mark.r_tier
def test_force_engine_loads_only_named_engine():
    reg = _build_engine_registry(_hw(), force="faster-whisper")
    assert isinstance(reg["en"], FasterWhisperEngine)
    assert isinstance(reg["multi"], FasterWhisperEngine)


@pytest.mark.r_tier
def test_apple_silicon_uses_parakeet_mlx_and_whisper_cpp():
    # ParakeetMlx / WhisperCpp lazy-imported in registry; should not raise on construction
    reg = _build_engine_registry(
        _hw(apple_silicon=True, os_="darwin", arch="arm64"),
        force=None,
    )
    assert reg["en"].capabilities.name == "parakeet-mlx"
    assert reg["multi"].capabilities.name == "whisper.cpp"


@pytest.mark.r_tier
def test_nvidia_cuda_uses_parakeet_tdt_and_faster_whisper_cuda():
    reg = _build_engine_registry(_hw(nvidia_cuda=True, cuda_cap=(8, 9)), force=None)
    assert reg["en"].capabilities.name == "parakeet-tdt"
    assert reg["multi"].capabilities.name == "faster-whisper"


@pytest.mark.r_tier
def test_cuda_below_7_falls_back_to_faster_whisper():
    reg = _build_engine_registry(_hw(nvidia_cuda=True, cuda_cap=(6, 1)), force=None)
    assert isinstance(reg["en"], FasterWhisperEngine)


@pytest.mark.r_tier
def test_force_remote_constructs_remote_engine():
    reg = _build_engine_registry(_hw(), force="remote:http://morpheus:5556")
    assert reg["en"].capabilities.name == "remote"


@pytest.mark.r_tier
def test_unknown_force_engine_raises():
    with pytest.raises(ValueError, match="unknown force_engine"):
        _build_engine_registry(_hw(), force="nonexistent-engine")


@pytest.mark.r_tier
def test_force_remote_empty_url_raises():
    with pytest.raises(ValueError, match="requires a URL"):
        _build_engine_registry(_hw(), force="remote:")
