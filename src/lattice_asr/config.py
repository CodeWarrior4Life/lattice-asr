"""Config loader — YAML, OS-aware paths. Spec §10."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RemoteConfig:
    url: str | None = None
    api_key_env: str = "LATTICE_ASR_API_KEY"
    timeout_seconds: float = 10.0


@dataclass
class PyAnnoteConfig:
    model: str = "pyannote/speaker-diarization-3.1"
    auth_token_env: str = "HF_TOKEN"


@dataclass
class SortformerConfig:
    model: str = "nvidia/sortformer-diarization-l24-30s"


@dataclass
class DiarizationConfig:
    enabled: bool = False
    adapter: str = "pyannote"
    pyannote: PyAnnoteConfig = field(default_factory=PyAnnoteConfig)
    sortformer: SortformerConfig = field(default_factory=SortformerConfig)


@dataclass
class LidConfig:
    enabled: bool = True
    confidence_threshold: float = 0.85


@dataclass
class LatticeAsrConfig:
    default_language: str = "en"
    sample_rate: int = 16000
    hardware_force: str | None = None
    remote: RemoteConfig = field(default_factory=RemoteConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    lid: LidConfig = field(default_factory=LidConfig)
    model_cache_dir: Path | None = None


def default_model_cache_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "lattice-asr" / "models"
    return Path.home() / ".cache" / "lattice-asr" / "models"


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path | str | None = None) -> LatticeAsrConfig:
    """Load config from YAML, fall back to defaults if file missing."""
    if path is None or not Path(path).exists():
        cfg = LatticeAsrConfig()
        if cfg.model_cache_dir is None:
            cfg.model_cache_dir = default_model_cache_dir()
        return cfg

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    remote = RemoteConfig(
        **_merge(
            {"url": None, "api_key_env": "LATTICE_ASR_API_KEY", "timeout_seconds": 10.0},
            raw.get("remote", {}) or {},
        )
    )
    diar_raw = raw.get("diarization", {}) or {}
    diar = DiarizationConfig(
        enabled=diar_raw.get("enabled", False),
        adapter=diar_raw.get("adapter", "pyannote"),
        pyannote=PyAnnoteConfig(
            **_merge(
                {"model": "pyannote/speaker-diarization-3.1", "auth_token_env": "HF_TOKEN"},
                diar_raw.get("pyannote", {}) or {},
            )
        ),
        sortformer=SortformerConfig(
            **_merge(
                {"model": "nvidia/sortformer-diarization-l24-30s"},
                diar_raw.get("sortformer", {}) or {},
            )
        ),
    )
    lid = LidConfig(
        **_merge(
            {"enabled": True, "confidence_threshold": 0.85},
            raw.get("lid", {}) or {},
        )
    )
    cfg = LatticeAsrConfig(
        default_language=raw.get("default_language", "en"),
        sample_rate=int(raw.get("sample_rate", 16000)),
        hardware_force=raw.get("hardware_force"),
        remote=remote,
        diarization=diar,
        lid=lid,
        model_cache_dir=Path(raw["model_cache_dir"])
        if raw.get("model_cache_dir")
        else default_model_cache_dir(),
    )
    return cfg
