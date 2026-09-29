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
def test_cpu_only_uses_faster_whisper_for_both_routes_with_distinct_models():
    """Both routes are faster-whisper, but they must NOT share one instance.

    This asserted `reg["en"] is reg["multi"]` until 2026-09-01. That shared
    instance was `distil-large-v3` -- an ENGLISH-ONLY model -- so the
    multilingual route was served by a model that MEASURED en@0.9463 on Spanish
    and translated it instead of transcribing. The routes are the same engine
    CLASS with different checkpoints, and that distinction is the fix.
    """
    reg = _build_engine_registry(_hw(), force=None)
    assert isinstance(reg["en"], FasterWhisperEngine)
    assert isinstance(reg["multi"], FasterWhisperEngine)
    assert reg["en"] is not reg["multi"]
    assert reg["en"].capabilities.languages == frozenset({"en"})
    assert "es" in reg["multi"].capabilities.languages


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


# --- 2026-09-28: torch-less CUDA (frozen Windows bundle) + model overrides ---


@pytest.mark.r_tier
def test_cuda_with_unknown_capability_routes_whisper_to_gpu():
    """CUDA seen only via CTranslate2 (no torch, so capability None): Whisper on
    the GPU for every language. Before this it fell through to the CPU tier."""
    reg = _build_engine_registry(_hw(nvidia_cuda=True, cuda_cap=None), force=None)
    assert reg["en"] is reg["multi"]
    assert reg["en"].capabilities.name == "faster-whisper"
    assert reg["en"]._device == "cuda"
    assert reg["en"]._compute_type == "default"


@pytest.mark.r_tier
def test_old_gpu_with_known_capability_still_routes_to_cpu():
    """Negative twin: a KNOWN capability below 7.0 keeps its pre-existing CPU
    route; the new GPU branch is only for the unknown-capability case."""
    reg = _build_engine_registry(_hw(nvidia_cuda=True, cuda_cap=(6, 1)), force=None)
    assert reg["en"]._device == "cpu"


@pytest.mark.r_tier
def test_cpu_model_override_serves_every_language_from_one_model():
    reg = _build_engine_registry(_hw(), force=None, cpu_model="small")
    assert reg["en"] is reg["multi"]
    assert reg["en"]._model_name == "small"
    assert "es" in reg["en"].capabilities.languages
    assert reg["en"]._device == "cpu"


@pytest.mark.r_tier
def test_no_cpu_model_keeps_the_two_tested_defaults():
    reg = _build_engine_registry(_hw(), force=None, cpu_model=None)
    assert reg["en"] is not reg["multi"]


@pytest.mark.r_tier
def test_cuda_model_override_applies_to_torchless_gpu_route():
    reg = _build_engine_registry(_hw(nvidia_cuda=True), force=None, cuda_model="medium")
    assert "es" in reg["en"].capabilities.languages
    assert reg["en"]._device == "cuda"


def test_has_cuda_falls_back_to_ctranslate2_when_torch_is_absent(monkeypatch):
    import builtins
    import sys
    import types

    from lattice_asr import hardware

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *a, **k)

    fake_ct2 = types.SimpleNamespace(get_cuda_device_count=lambda: 1)
    monkeypatch.setitem(sys.modules, "ctranslate2", fake_ct2)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert hardware._has_cuda() is True
    fake_ct2.get_cuda_device_count = lambda: 0
    assert hardware._has_cuda() is False
