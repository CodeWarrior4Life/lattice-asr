# lattice-asr

Hardware-adaptive multilingual ASR library for the Lattice family.

> **Status:** v0.1 scaffold, no implementation yet. Spec is locked. Implementation lands in subsequent sessions.

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
| Plan | Pending |
| Implementation | Pending — repo scaffolded S13 (2026-05-08) |
| First consumer | `lattice-dictate` (planned) |
| Second consumer | `lattice-meetbot` (refactored transcription path) |

## License

Apache-2.0 (Lattice default). See [LICENSE](LICENSE).

## Family

Part of the [Lattice family](https://github.com/CodeWarrior4Life?tab=repositories&q=lattice). Sibling libraries: `lattice-meetbot`, `lattice-meeting`, `lattice-watch`, `lattice-dictate` (planned), `lattice-recall` (planned).
