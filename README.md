# lattice-asr

Hardware-adaptive multilingual ASR library for the Lattice family.

> **Status:** v0.1 implementation in progress. W1 (foundation), W2 (Whisper + LID + Transcriber), and W4 (RemoteEngine + lattice-asr-server) are landed; W3 (Apple Silicon engines) and W5 (diarization) are pending. 69 r_tier tests passing. See [CHANGELOG.md](CHANGELOG.md) for the unreleased detail.

## What it is

A Lattice-layer library that lifts speech-to-text from per-consumer ad-hoc plumbing into a single pluggable surface. Consumers (`lattice-meetbot` for meeting transcription, `lattice-dictate` for push-to-talk dictation, WhatsApp voice-message decryption, future ambient capture) all transcribe through the same `Transcriber` interface.

## What it solves

- **Hardware adaptivity** — picks Parakeet TDT on NVIDIA, parakeet.cpp on Apple Silicon, Distil-Whisper via faster-whisper on CPU-only. Single API across all paths.
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
| W3 Apple Silicon engines | Pending (needs Mac runner) |
| W4 RemoteEngine + lattice-asr-server | Landed |
| W5 Diarization | Pending (needs `HF_TOKEN`) |
| W6 Ship (perf, CI, release) | Pending |
| First consumer | `lattice-dictate` (planned) |
| Second consumer | `lattice-meetbot` (refactored transcription path) |

## License

Apache-2.0 (Lattice default). See [LICENSE](LICENSE).

## Family

Part of the [Lattice family](https://github.com/CodeWarrior4Life?tab=repositories&q=lattice). Sibling libraries: `lattice-meetbot`, `lattice-meeting`, `lattice-watch`, `lattice-dictate` (planned), `lattice-recall` (planned).
