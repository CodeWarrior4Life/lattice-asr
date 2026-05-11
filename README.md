# lattice-asr

Hardware-adaptive multilingual ASR library for the Lattice family.

[![PyPI](https://img.shields.io/pypi/v/lattice-asr.svg)](https://pypi.org/project/lattice-asr/) [![License](https://img.shields.io/pypi/l/lattice-asr.svg)](https://github.com/CodeWarrior4Life/lattice-asr/blob/main/LICENSE)

```bash
pip install lattice-asr==0.1.0
```

> **Status:** v0.1 implementation in progress. W1 (foundation), W2 (Whisper + LID + Transcriber), W3.1 (`ParakeetMlxEngine`), W3.2 (`ParakeetTdtEngine`), W4 (RemoteEngine + lattice-asr-server), W5 (diarization), and W6.1 (perf-gate skeleton) are landed; W3-future (`WhisperCppEngine` for Apple Silicon multilingual) is deferred past v0.1; W6.2 (CI workflow), W6.3 (this rewrite), and v0.1.0 (tag + PyPI) are pending. All three v0.1 perf gates clear via the wrappers on canonical hosts: **C1 RTF 45.43×** on Switch (Apple M4 Pro, parakeet-mlx), **C2 RTF 115.12×** on Cypher (RTX 2070 Turing, parakeet-tdt), **C3 RTF 3.35×** on Switch (faster-whisper distil-large-v3 int8). 81 r_tier tests passing. See [CHANGELOG.md](CHANGELOG.md) for the unreleased detail and [docs/performance-baseline.md](docs/performance-baseline.md) for the canonical baselines.

## What it is

A Lattice-layer library that lifts speech-to-text from per-consumer ad-hoc plumbing into a single pluggable surface. Consumers (`lattice-meetbot` for meeting transcription, `lattice-dictate` for push-to-talk dictation, WhatsApp voice-message decryption, future ambient capture) all transcribe through the same `Transcriber` interface.

## What it solves

- **Hardware adaptivity** — picks Parakeet-TDT (NeMo) on NVIDIA, parakeet-mlx on Apple Silicon, Distil-Whisper via faster-whisper on CPU-only (arm64 or x86_64 — non-GPU fallback path). Single API across all paths.
- **Multilingual** — dual-engine load: English route uses Parakeet (fastest, English-only); non-English route uses Whisper-large-v3-turbo or Distil-Whisper. Routing via Silero LID on the first 1.5 s of audio.
- **Optional diarization** — `transcribe(..., diarize=True)` returns segments with speaker labels (pyannote.audio CPU/GPU, NVIDIA Sortformer GPU).
- **Streaming** — VAD-segment-bounded partial results for ambient/meeting use.
- **Remote engine** — `RemoteEngine` adapter forwards transcription to a network endpoint speaking the lattice-asr wire protocol; lets a CPU-only laptop offload to a GPU host.
- **Telemetry-injected** — every call records duration, engine, language, audio length to a consumer-supplied sink. No hidden coupling.

## Non-goals (v0.1)

Speaker identification (deferred to a future `lattice-voiceprint`), on-device fine-tuning, sub-100 ms streaming partials, audio enhancement, translation, browser/mobile bindings.

## Status

| Phase | State |
| --- | --- |
| Spec | Locked — vault `02_Projects/Lattice/lattice-asr/Specifications/2026-04-27 lattice-asr v1 - Design Spec.md` |
| Plan | Ratified and in execution — vault `02_Projects/Lattice/lattice-asr/Plans/2026-05-08 lattice-asr v0.1 - Implementation Plan.md` |
| W1 Foundation | Landed (hardware probe, types, telemetry, config) |
| W2 Whisper + LID + Transcriber MVP | Landed |
| W3.1 ParakeetMlxEngine (Apple Silicon EN, MLX runtime) | Landed — verified Switch RTF 45.43× via wrapper |
| W3.2 ParakeetTdtEngine (NVIDIA EN, NeMo runtime) | Landed — verified Cypher RTX 2070 RTF 115.12× via wrapper |
| W3-future WhisperCppEngine (Apple Silicon multilingual) | Deferred past v0.1 |
| W4 RemoteEngine + lattice-asr-server | Landed |
| W5 Diarization | Landed (pyannote + sortformer adapters + Transcriber wire-in; real-model exercise gated on `HF_TOKEN` / NeMo) |
| W6 Ship (perf, CI, release) | Partial — W6.1 perf-gate skeleton landed and all three C1/C2/C3 gates pass via wrappers on canonical hosts; W6.2 CI workflow + W6.3 README rewrite + v0.1.0 tag pending. Canonical hosts: C1+C3 on Switch (Apple M4 Pro Mac mini), C2 on Cypher (Linux + RTX 2070). |
| First consumer | `lattice-dictate` (planned) |
| Second consumer | `lattice-meetbot` (refactored transcription path) |

## License

Apache-2.0 (Lattice default). See [LICENSE](LICENSE).

## Family

Part of the [Lattice family](https://github.com/CodeWarrior4Life?tab=repositories&q=lattice). Sibling libraries: `lattice-meetbot`, `lattice-meeting`, `lattice-watch`, `lattice-dictate` (planned), `lattice-recall` (planned).
