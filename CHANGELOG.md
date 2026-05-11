# Changelog

All notable changes to `lattice-asr` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-11

**Published to PyPI:** https://pypi.org/project/lattice-asr/0.1.0/ — install with `pip install lattice-asr==0.1.0`.

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

#### W3 — Parakeet engines (spec §6.1, §6.2)
- `ParakeetMlxEngine` (`src/lattice_asr/engines/parakeet_mlx.py`) — Apple Silicon MLX runtime via `parakeet-mlx` (PyPI 0.5.1). Lazy `from_pretrained` (default `mlx-community/parakeet-tdt-0.6b-v3`); PCM-or-WAV bytes via tempfile; `asyncio.to_thread` wrapping. **Renamed from `ParakeetCppEngine` / `parakeet_cpp.py`** — the prior name referenced the phantom `parakeet-cpp-py` dependency that never shipped to PyPI (see `a4e5922`). Force-engine string `"parakeet.cpp"` -> `"parakeet-mlx"`; capability name updated; transcriber registry and r_tier selection test renamed in lockstep. C1 perf gate verified on Switch (Apple M4 Pro) at **RTF 45.43× via wrapper** (>10× target, 3× the S41 direct-API 15.13× baseline — model cached). Commit `e4d6a66`.
- `ParakeetTdtEngine` (`src/lattice_asr/engines/parakeet_tdt.py`) — NVIDIA via `nemo-toolkit[asr]`. Lazy `nemo_asr.models.ASRModel.from_pretrained` (default `nvidia/parakeet-tdt-0.6b-v3`, replacing the un-verified `parakeet-tdt-1.1b-v3` reference); `asyncio.to_thread` wrapping. Warmup absorbs CUDA-kernel JIT compile via a 0.5 s silence dummy inference (RTX 2070 Turing pays this on first transcribe; without it the first timed run measured 32–43× cold/loaded — failing the 50× gate even with a fast direct-API number). C2 perf gate verified on Cypher (RTX 2070 sm_75) at **RTF 115.12× via wrapper** (>50× target, ~2× the S41 direct-API 64.13× baseline — warmup absorbing kernel compile is the delta). Commits `8289072` + `8b17997`.

#### W5 — Diarization (spec §7)
- `Diarizer` ABC + `merge_segments_with_text` helper (`src/lattice_asr/diarize/base.py`). Merge is first-match-wins on shared boundaries (the earlier speaker in input order claims a transcription whose midpoint falls on a shared boundary; prevents double-counting when intervals overlap).
- `PyAnnoteAdapter` (`src/lattice_asr/diarize/pyannote.py`) — lazy pyannote.audio import; raises clear `ValueError` if `HF_TOKEN` env / `auth_token` kwarg absent.
- `NvidiaSortformerAdapter` (`src/lattice_asr/diarize/sortformer.py`) — lazy NeMo import; runs against `nvidia/sortformer-diarization-l24-30s`.
- `Transcriber` diarizer wiring (`src/lattice_asr/transcriber.py`) — `enable_diarization=True` now loads `PyAnnoteAdapter` (default) or `NvidiaSortformerAdapter` (when `config.diarization.adapter='sortformer'`) per `config.diarization.{adapter,pyannote,sortformer}`; unknown adapter raises `ValueError`. Real model loads stay lazy (HF_TOKEN / NeMo deps not required for construction).
- W5 r_tier unit tests across pyannote scaffold, sortformer scaffold, 6 segment-merge boundary cases, and 3 Transcriber-wire-in cases (pyannote default, sortformer override, unknown-adapter ValueError).

#### W6 — Ship (partial, spec §12.3)
- **W6.1 perf-gate skeleton** (`tests/perf/test_perf_gates.py`) — C1 (Apple Silicon parakeet-mlx >10× RTF), C2 (NVIDIA parakeet-tdt >50× RTF), C3 (CPU faster-whisper distil-large-v3 int8 >2× RTF). Opt-in behind `LATTICE_ASR_PERF_RUN=1`; r_tier suite unaffected. As of S42 all three gates clear via wrappers on canonical hosts (Switch for C1+C3, Cypher for C2).
- `hello_en_30s_wav` session-scope fixture added to `tests/conftest.py` (mirrors `hello_en_2s_wav` pattern; HF dataset `CodeWarrior4Life/lattice-asr-fixtures` currently 401-unauth, local fallback at `tests/fixtures/audio/hello-en-30s.wav` documented).
- `docs/performance-baseline.md` — gate table, how-to-run per gate, fixture-status note, baseline log placeholders for Morpheus / Trinity / Cypher.

#### Tooling + config
- Registered remaining plan-spec pytest markers (`apple_silicon`, `nvidia_cuda`, `perf`) in `pyproject.toml`.

### Test surface
- 81 r_tier tests passing across hardware probe, types, telemetry, config, engines (faster_whisper, remote), LID, transcriber, module convenience, server, and diarize (pyannote / sortformer / segment-merge / Transcriber wire-in).
- 9 tests skipped under `apple_silicon` / `nvidia_cuda` / `s_tier` / `perf` markers — gated on hardware/scope/env not present on the current runner (perf gates require `LATTICE_ASR_PERF_RUN=1`).
- Coverage: 79.15%+ (v0.1 65% gate cleared).

### Pending (v0.1)
- W3 future: `WhisperCppEngine` (Apple Silicon Metal multilingual) — deferred past v0.1.
- W6.2: `.github/workflows/ci.yml` with arm64 macOS runner (C1+C3 on Switch) AND NVIDIA-capable runner (C2 on Cypher); self-hosted runner registration on both; uses `workflow`-scoped `nexus-durable-master` PAT.
- W6.3: README quickstart rewrite reflecting Switch canonical C1+C3 + parakeet-mlx (not parakeet.cpp) + broadened C3 hardware-class label.
- W6.4 / v0.1.0: tag + PyPI publish (owner-gated).
- Plan + spec phantom-dep correction sweep: replace inline `parakeet-cpp-py` / `parakeet.cpp` body references in `Plans/2026-05-08…Implementation Plan.md` + `Specifications/2026-04-27 lattice-asr v1 - Design Spec.md` (correction notices already at top per S41; deeper body rewrite deferred to pre-v0.1.0 polish if churn-worthy).

[Unreleased]: https://github.com/CodeWarrior4Life/lattice-asr/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/CodeWarrior4Life/lattice-asr/releases/tag/v0.1.0
