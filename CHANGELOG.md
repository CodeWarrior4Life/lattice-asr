# Changelog

All notable changes to `lattice-asr` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Repo scaffold: pyproject, src/, tests/, CI workflow, .gitignore, README.
- Spec lives in vault: `02_Projects/Lattice/lattice-asr/Specifications/2026-04-27 lattice-asr v1 - Design Spec.md`.

### Pending (v0.1)
- `Transcriber` core class with hardware-probed engine selection.
- `TranscriptionEngine` Protocol + WhisperEngine, ParakeetEngine, RemoteEngine adapters.
- VAD-segment-bounded streaming via `transcribe_streaming()`.
- Optional diarization via `diarize=True` (pyannote.audio + Sortformer adapters).
- Telemetry sink injection.
- R-tier (≥65% coverage gate) + S-tier (real engines, nightly) + E-tier (real hardware, opt-in).

[Unreleased]: https://github.com/CodeWarrior4Life/lattice-asr/commits/main
