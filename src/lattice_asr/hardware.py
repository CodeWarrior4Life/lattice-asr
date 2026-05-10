"""Hardware probe -- runs once at Transcriber init.

Spec §3, §5: Probe identifies platform, CPU arch, Apple Silicon Y/N,
NVIDIA CUDA Y/N + capability, RAM, cores. Used to select engine registry.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class HardwareProfile:
    os: Literal["darwin", "linux", "win32"]
    cpu_arch: Literal["arm64", "x86_64"]
    apple_silicon: bool
    nvidia_cuda: bool
    cuda_capability: tuple[int, int] | None
    total_ram_gb: float
    cpu_cores: int


_OS_MAP = {"Darwin": "darwin", "Linux": "linux", "Windows": "win32"}
_ARCH_MAP = {"arm64": "arm64", "aarch64": "arm64", "AMD64": "x86_64", "x86_64": "x86_64"}


def _has_cuda() -> bool:
    try:
        import torch  # type: ignore[import-not-found]

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _cuda_capability() -> tuple[int, int] | None:
    try:
        import torch  # type: ignore[import-not-found]

        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(0)
        return (int(major), int(minor))
    except Exception:
        return None


def _total_ram_gb() -> float:
    try:
        import psutil  # type: ignore[import-not-found]

        return round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return round(pages * page_size / (1024**3), 2)
        except (ValueError, AttributeError, OSError):
            return 0.0


def _cpu_cores() -> int:
    return os.cpu_count() or 1


def detect_hardware() -> HardwareProfile:
    """Probe host hardware once. Idempotent (no caching here; caller may cache)."""
    raw_os = platform.system()
    raw_arch = platform.machine()
    os_norm = _OS_MAP.get(raw_os, "linux")
    arch_norm = _ARCH_MAP.get(raw_arch, "x86_64")
    apple_silicon = os_norm == "darwin" and arch_norm == "arm64"
    has_cuda = _has_cuda()
    cap = _cuda_capability() if has_cuda else None
    return HardwareProfile(
        os=os_norm,  # type: ignore[arg-type]
        cpu_arch=arch_norm,  # type: ignore[arg-type]
        apple_silicon=apple_silicon,
        nvidia_cuda=has_cuda,
        cuda_capability=cap,
        total_ram_gb=_total_ram_gb(),
        cpu_cores=_cpu_cores(),
    )
