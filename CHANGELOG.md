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

#### W5 — Diarization (spec §7)
- `Diarizer` ABC + `merge_segments_with_text` helper (`src/lattice_asr/diarize/base.py`). Merge is first-match-wins on shared boundaries (the earlier speaker in input order claims a transcription whose midpoint falls on a shared boundary; prevents double-counting when intervals overlap).
- `PyAnnoteAdapter` (`src/lattice_asr/diarize/pyannote.py`) — lazy pyannote.audio import; raises clear `ValueError` if `HF_TOKEN` env / `auth_token` kwarg absent.
- `NvidiaSortformerAdapter` (`src/lattice_asr/diarize/sortformer.py`) — lazy NeMo import; runs against `nvidia/sortformer-diarization-l24-30s`.
- `Transcriber` diarizer wiring (`src/lattice_asr/transcriber.py`) — `enable_diarization=True` now loads `PyAnnoteAdapter` (default) or `NvidiaSortformerAdapter` (when `config.diarization.adapter='sortformer'`) per `config.diarization.{adapter,pyannote,sortformer}`; unknown adapter raises `ValueError`. Real model loads stay lazy (HF_TOKEN / NeMo deps not required for construction).
- W5 r_tier unit tests across pyannote scaffold, sortformer scaffold, 6 segment-merge boundary cases, and 3 Transcriber-wire-in cases (pyannote default, sortformer override, unknown-adapter ValueError).

#### W6 — Ship (partial, spec §12.3)
- **W6.1 perf-gate skeleton** (`tests/perf/test_perf_gates.py`) — C1 (Apple Silicon parakeet.cpp >10× RTF), C2 (NVIDIA parakeet-tdt >50× RTF), C3 (CPU faster-whisper distil-large-v3 int8 >2× RTF). Opt-in behind `LATTICE_ASR_PERF_RUN=1`; r_tier suite unaffected.
- `hello_en_30s_wav` session-scope fixture added to `tests/conftest.py` (mirrors `hello_en_2s_wav` pattern; HF dataset `CodeWarrior4Life/lattice-asr-fixtures` currently 401-unauth, local fallback at `tests/fixtures/audio/hello-en-30s.wav` documented).
- `docs/performance-baseline.md` — gate table, how-to-run per gate, fixture-status note, baseline log placeholders for Morpheus / Trinity / Cypher.

#### Tooling + config
- Registered remaining plan-spec pytest markers (`apple_silicon`, `nvidia_cuda`, `perf`) in `pyproject.toml`.

### Test surface
- 81 r_tier tests passing across hardware probe, types, telemetry, config, engines (faster_whisper, remote), LID, transcriber, module convenience, server, and diarize (pyannote / sortformer / segment-merge / Transcriber wire-in).
- 9 tests skipped under `apple_silicon` / `nvidia_cuda` / `s_tier` / `perf` markers — gated on hardware/scope/env not present on the current runner (perf gates require `LATTICE_ASR_PERF_RUN=1`).
- Coverage: 79.15%+ (v0.1 65% gate cleared).

### Pending (v0.1)
- W3: `ParakeetCppEngine` (Apple Silicon), `ParakeetTdtEngine` (NVIDIA), `WhisperCppEngine` (Apple Silicon Metal) — needs Mac and NVIDIA runners respectively.
- W6.2: `.github/workflows/ci.yml` (needs `workflow`-scoped gh token).
- W6.3: README quickstart rewrite + adoption-tracker doc.
- W6.4 / v0.1.0: tag + PyPI publish (owner-gated).

[Unreleased]: https://github.com/CodeWarrior4Life/lattice-asr/commits/main
