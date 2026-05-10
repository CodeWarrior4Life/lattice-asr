from unittest.mock import patch
import pytest
from lattice_asr.hardware import detect_hardware, HardwareProfile


@pytest.mark.r_tier
def test_apple_silicon_detection():
    with (
        patch("platform.system", return_value="Darwin"),
        patch("platform.machine", return_value="arm64"),
        patch("lattice_asr.hardware._has_cuda", return_value=False),
        patch("lattice_asr.hardware._total_ram_gb", return_value=16.0),
        patch("lattice_asr.hardware._cpu_cores", return_value=10),
    ):
        hw = detect_hardware()
    assert hw.os == "darwin"
    assert hw.cpu_arch == "arm64"
    assert hw.apple_silicon is True
    assert hw.nvidia_cuda is False


@pytest.mark.r_tier
def test_nvidia_cuda_detection_with_capability():
    with (
        patch("platform.system", return_value="Linux"),
        patch("platform.machine", return_value="x86_64"),
        patch("lattice_asr.hardware._has_cuda", return_value=True),
        patch("lattice_asr.hardware._cuda_capability", return_value=(8, 9)),
        patch("lattice_asr.hardware._total_ram_gb", return_value=64.0),
        patch("lattice_asr.hardware._cpu_cores", return_value=16),
    ):
        hw = detect_hardware()
    assert hw.nvidia_cuda is True
    assert hw.cuda_capability == (8, 9)


@pytest.mark.r_tier
def test_cpu_only_linux():
    with (
        patch("platform.system", return_value="Linux"),
        patch("platform.machine", return_value="x86_64"),
        patch("lattice_asr.hardware._has_cuda", return_value=False),
        patch("lattice_asr.hardware._total_ram_gb", return_value=8.0),
        patch("lattice_asr.hardware._cpu_cores", return_value=8),
    ):
        hw = detect_hardware()
    assert hw.os == "linux"
    assert hw.apple_silicon is False
    assert hw.nvidia_cuda is False


@pytest.mark.r_tier
def test_windows_cpu_only():
    with (
        patch("platform.system", return_value="Windows"),
        patch("platform.machine", return_value="AMD64"),
        patch("lattice_asr.hardware._has_cuda", return_value=False),
        patch("lattice_asr.hardware._total_ram_gb", return_value=32.0),
        patch("lattice_asr.hardware._cpu_cores", return_value=12),
    ):
        hw = detect_hardware()
    assert hw.os == "win32"
    assert hw.cpu_arch == "x86_64"


@pytest.mark.r_tier
def test_windows_nvidia():
    with (
        patch("platform.system", return_value="Windows"),
        patch("platform.machine", return_value="AMD64"),
        patch("lattice_asr.hardware._has_cuda", return_value=True),
        patch("lattice_asr.hardware._cuda_capability", return_value=(7, 5)),
        patch("lattice_asr.hardware._total_ram_gb", return_value=32.0),
        patch("lattice_asr.hardware._cpu_cores", return_value=12),
    ):
        hw = detect_hardware()
    assert hw.nvidia_cuda is True


@pytest.mark.r_tier
def test_hardware_profile_is_frozen():
    with (
        patch("platform.system", return_value="Linux"),
        patch("platform.machine", return_value="x86_64"),
        patch("lattice_asr.hardware._has_cuda", return_value=False),
        patch("lattice_asr.hardware._total_ram_gb", return_value=8.0),
        patch("lattice_asr.hardware._cpu_cores", return_value=4),
    ):
        hw = detect_hardware()
    with pytest.raises((AttributeError, TypeError)):
        hw.os = "darwin"  # type: ignore[misc]
