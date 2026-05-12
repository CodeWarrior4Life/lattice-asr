from pathlib import Path

import pytest

from lattice_asr.config import default_model_cache_dir, load_config


@pytest.mark.r_tier
def test_load_config_missing_file_returns_defaults(tmp_path):
    cfg = load_config(tmp_path / "nope.yaml")
    assert cfg.default_language == "en"
    assert cfg.sample_rate == 16000
    assert cfg.hardware_force is None
    assert cfg.lid.confidence_threshold == 0.85


@pytest.mark.r_tier
def test_load_config_partial_yaml_fills_defaults(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("default_language: es\nlid:\n  confidence_threshold: 0.7\n")
    cfg = load_config(p)
    assert cfg.default_language == "es"
    assert cfg.lid.confidence_threshold == 0.7
    assert cfg.sample_rate == 16000  # default preserved


@pytest.mark.r_tier
def test_load_config_full_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "default_language: fr\n"
        "sample_rate: 22050\n"
        "hardware_force: faster-whisper\n"
        "remote:\n"
        "  url: http://morpheus:5556\n"
        "  api_key_env: LATTICE_ASR_API_KEY\n"
        "  timeout_seconds: 20\n"
        "diarization:\n"
        "  enabled: true\n"
        "  adapter: pyannote\n"
        "lid:\n"
        "  enabled: false\n"
        "  confidence_threshold: 0.9\n"
    )
    cfg = load_config(p)
    assert cfg.hardware_force == "faster-whisper"
    assert cfg.remote.url == "http://morpheus:5556"
    assert cfg.diarization.enabled is True
    assert cfg.lid.enabled is False


@pytest.mark.r_tier
def test_default_model_cache_dir_unix(monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr("pathlib.Path.home", lambda: Path("/home/user"))
    p = default_model_cache_dir()
    assert p == Path("/home/user/.cache/lattice-asr/models")


@pytest.mark.r_tier
def test_default_model_cache_dir_windows(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    p = default_model_cache_dir()
    assert p == tmp_path / "lattice-asr" / "models"


@pytest.mark.r_tier
def test_load_config_invalid_yaml_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("default_language: [\n")  # malformed
    with pytest.raises(Exception):  # noqa: B017 - YAML loader exception type varies by version
        load_config(p)
