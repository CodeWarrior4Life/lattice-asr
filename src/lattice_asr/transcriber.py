"""Transcriber — engine selection + transcribe routing. Spec §4, §5."""

from __future__ import annotations

from lattice_asr.engines.base import TranscriptionEngine
from lattice_asr.engines.faster_whisper import FasterWhisperEngine
from lattice_asr.hardware import HardwareProfile


def _build_engine_registry(
    hw: HardwareProfile, force: str | None
) -> dict[str, TranscriptionEngine]:
    """Build {language_route: engine}. Spec §5.

    Routes: "en" (English-optimized) and "multi" (multilingual fallback).
    Lazy-imports per-platform engines to avoid pulling heavy deps unless needed.
    """
    if force:
        if force == "faster-whisper":
            engine = FasterWhisperEngine(
                model="distil-large-v3",
                device="cuda" if hw.nvidia_cuda else "cpu",
                compute_type="float16" if hw.nvidia_cuda else "int8",
            )
            return {"en": engine, "multi": engine}
        if force.startswith("remote:"):
            from lattice_asr.engines.remote import RemoteEngine

            url = force.split(":", 1)[1]
            if not url:
                raise ValueError("force_engine='remote:' requires a URL, got empty")
            engine = RemoteEngine(url=url)
            return {"en": engine, "multi": engine}
        if force == "parakeet.cpp":
            from lattice_asr.engines.parakeet_cpp import ParakeetCppEngine

            engine = ParakeetCppEngine()
            return {"en": engine, "multi": engine}
        if force == "parakeet-tdt":
            from lattice_asr.engines.parakeet_tdt import ParakeetTdtEngine

            engine = ParakeetTdtEngine()
            return {"en": engine, "multi": engine}
        if force == "whisper.cpp":
            from lattice_asr.engines.whisper_cpp import WhisperCppEngine

            engine = WhisperCppEngine()
            return {"en": engine, "multi": engine}
        raise ValueError(f"unknown force_engine: {force}")

    if hw.apple_silicon:
        from lattice_asr.engines.parakeet_cpp import ParakeetCppEngine
        from lattice_asr.engines.whisper_cpp import WhisperCppEngine

        return {
            "en": ParakeetCppEngine(),
            "multi": WhisperCppEngine(),
        }

    if hw.nvidia_cuda and hw.cuda_capability is not None and hw.cuda_capability >= (7, 0):
        from lattice_asr.engines.parakeet_tdt import ParakeetTdtEngine

        return {
            "en": ParakeetTdtEngine(),
            "multi": FasterWhisperEngine(
                model="distil-large-v3", device="cuda", compute_type="float16"
            ),
        }

    cpu_engine = FasterWhisperEngine(model="distil-large-v3", device="cpu", compute_type="int8")
    return {"en": cpu_engine, "multi": cpu_engine}
