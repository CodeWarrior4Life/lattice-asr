# Changelog

All notable changes to `lattice-asr` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### W1 — Foundation (spec §3, §4, §9, §10)
- Hardware probe (`hardware.py`): detects OS, architecture, CUDA, RAM, cores with warn-on-unknown-OS/arch + missing-RAM-probe fallback.
- Public API dataclasses (`types.py`) + `TranscriptionEngine` ABC (`engines/base.py`).
- Telemetry sinks: `NullTelemetrySink` + `ListTelemetrySink` (`telemetry.py`).
- Config YAML loader with OS-aware defaults (`config.py`).

#### W2 — Whisper + LID + Transcriber MVP (spec §4, §5, §6.3, §8)
- `FasterWhisperEngine` — multilingual CPU/CUDA engine.
- `SileroLid` — language detection on first 1.5 s of audio (warmup contract + latency bound).
- Engine registry + lazy-import; stubs for not-yet-implemented engines (W3).
- `Transcriber` class — LID routing + diarization scaffold + telemetry; guards empty `force_remote`; covers unknown-force in tests.
- Module-level `transcribe()` singleton convenience for one-shot use.

#### W4 — RemoteEngine + lattice-asr-server (spec §6.5)
- `RemoteEngine` — HTTP forwarding with retries + version check; covers exhausted-retries, connection-error, and streaming paths.
- `lattice-asr-server` (`src/lattice_asr_server/`) — FastAPI wrapper exposing `/v1/health` + `/v1/transcribe`, with:
  - Optional bearer auth via `LATTICE_ASR_API_KEY` env.
  - `X-Lattice-Asr-Api-Version: v1` response header (middleware).
  - Lazy lock-guarded `Transcriber` singleton (asyncio.Lock against concurrent first-callers).
  - Env-driven host/port (`LATTICE_ASR_HOST` / `LATTICE_ASR_PORT`).
  - `lattice-asr-server` console script entry point.
- In-process server↔client round-trip integration test (`tests/test_remote_integration.py`) using `httpx.ASGITransport`.

### Test surface
- 69 r_tier tests passing across hardware probe, types, telemetry, config, engines (faster_whisper, remote), LID, transcriber, module convenience, and server.
- 5 tests skipped under `apple_silicon` / `nvidia_cuda` markers — gated on hardware not present on the current runner.

### Pending (v0.1)
- W3: `ParakeetCppEngine` (Apple Silicon), `ParakeetTdtEngine` (NVIDIA), `WhisperCppEngine` (Apple Silicon Metal).
- W5: `PyAnnoteAdapter`, `NvidiaSortformerAdapter`, diarizer wire-in to `Transcriber`, segment-merge helper.
- W6: performance baseline gates (C1/C2/C3 from spec §12.3), CI workflow, README rewrite, `v0.1.0` tag + PyPI publish.

[Unreleased]: https://github.com/CodeWarrior4Life/lattice-asr/commits/main
