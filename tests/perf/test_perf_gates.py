"""Performance gates per spec §12.3.

C1 — Apple Silicon parakeet.cpp >10× RTF.
C2 — NVIDIA parakeet-tdt >50× RTF.
C3 — CPU faster-whisper distil-large-v3 int8 >2× RTF.

Hard-fail gates: C1, C2, C3. Run via `LATTICE_ASR_PERF_RUN=1 pytest tests/perf/ -m perf`.
"""

import os
import time
import pytest

PERF_RUN = os.getenv("LATTICE_ASR_PERF_RUN") == "1"
perfonly = pytest.mark.skipif(not PERF_RUN, reason="set LATTICE_ASR_PERF_RUN=1")


@perfonly
@pytest.mark.perf
@pytest.mark.apple_silicon
@pytest.mark.asyncio
async def test_c1_apple_silicon_parakeet_cpp_10x_rtf(hello_en_30s_wav):
    from lattice_asr.engines.parakeet_cpp import ParakeetCppEngine

    eng = ParakeetCppEngine()
    await eng.warmup()
    audio = hello_en_30s_wav.read_bytes()
    t0 = time.monotonic()
    result = await eng.transcribe(audio, sample_rate=16000, language="en")
    elapsed = time.monotonic() - t0
    rtf = (result.audio_duration_ms / 1000) / elapsed
    assert rtf > 10.0, f"C1 fail: rtf={rtf:.1f}× expected >10×"


@perfonly
@pytest.mark.perf
@pytest.mark.nvidia_cuda
@pytest.mark.asyncio
async def test_c2_nvidia_parakeet_tdt_50x_rtf(hello_en_30s_wav):
    from lattice_asr.engines.parakeet_tdt import ParakeetTdtEngine

    eng = ParakeetTdtEngine()
    await eng.warmup()
    audio = hello_en_30s_wav.read_bytes()
    t0 = time.monotonic()
    result = await eng.transcribe(audio, sample_rate=16000, language="en")
    elapsed = time.monotonic() - t0
    rtf = (result.audio_duration_ms / 1000) / elapsed
    assert rtf > 50.0, f"C2 fail: rtf={rtf:.1f}× expected >50×"


@perfonly
@pytest.mark.perf
@pytest.mark.asyncio
async def test_c3_cpu_distil_2x_rtf(hello_en_30s_wav):
    from lattice_asr.engines.faster_whisper import FasterWhisperEngine

    eng = FasterWhisperEngine(model="distil-large-v3", device="cpu", compute_type="int8")
    await eng.warmup()
    audio = hello_en_30s_wav.read_bytes()
    t0 = time.monotonic()
    result = await eng.transcribe(audio, sample_rate=16000, language="en")
    elapsed = time.monotonic() - t0
    rtf = (result.audio_duration_ms / 1000) / elapsed
    assert rtf > 2.0, f"C3 fail: rtf={rtf:.1f}× expected >2×"
