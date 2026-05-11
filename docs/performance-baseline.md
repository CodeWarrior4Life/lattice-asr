# Performance Baseline — lattice-asr v0.1

Hard-fail performance gates per [spec §12.3](../README.md) and plan task W6.1
(`02_Projects/Lattice/lattice-asr/Plans/2026-05-08 lattice-asr v0.1 - Implementation Plan.md`).
These gates protect the engine throughput contract that consumers (lattice-dictation,
lattice-meeting, MindPractice) depend on.

## Gates

| Gate | Host class | Engine | Audio | Target RTF | Hard-fail | Measured |
| ---- | ---------- | ------ | ----- | ---------- | --------- | -------- |
| C1   | Apple Silicon (M-series) | `parakeet.cpp` | hello-en-30s.wav | > 10× | yes | TBD (Trinity / Cypher) |
| C2   | NVIDIA CUDA GPU | `parakeet-tdt` (NeMo) | hello-en-30s.wav | > 50× | yes | TBD (Morpheus RTX 5080) |
| C3   | Modern x86_64 CPU (8-core) | `faster-whisper` distil-large-v3 int8 | hello-en-30s.wav | > 2× | yes | TBD (Morpheus CPU) |

RTF = `audio_duration_seconds / wall_clock_seconds`. A 30-second clip transcribed in
3 seconds is 10× RTF. The 30s clip is the standardized benchmark fixture; the 2s
clip is for smoke/correctness, not throughput.

## How to run

Perf gates are opt-in via `LATTICE_ASR_PERF_RUN=1`. The `perf` marker is independent
of `s_tier` and `e_tier`; perf tests do not run on a default `pytest` invocation.

### C1 — Apple Silicon parakeet.cpp

```bash
LATTICE_ASR_PERF_RUN=1 pytest tests/perf/ -m perf -k c1 -v
```

Requires `parakeet-cpp-py` (the `parakeet` extra). Skip elsewhere via the
`apple_silicon` marker. Run on Trinity (M5 Max) or Cypher.

### C2 — NVIDIA parakeet-tdt

```bash
LATTICE_ASR_PERF_RUN=1 pytest tests/perf/ -m perf -k c2 -v
```

Requires `nemo-toolkit[asr]` (the `nvidia` extra, Linux-only). Skip elsewhere via
the `nvidia_cuda` marker. Run on Morpheus (RTX 5080).

### C3 — CPU distil-large-v3 int8

```bash
LATTICE_ASR_PERF_RUN=1 pytest tests/perf/ -m perf -k c3 -v
```

Requires `faster-whisper` (the `whisper` extra). Portable; runs on any modern
8-core x86_64. Reference host: Morpheus CPU.

### All three (single command)

```bash
LATTICE_ASR_PERF_RUN=1 pytest tests/perf/ -m perf -v
```

Tests for engines unavailable on the current host raise `ImportError` /
`OSError` at import time; the marker-host-class pairing is informational, not
enforced by the test harness.

## Fixture status

The 30-second benchmark clip lives at:

```
https://huggingface.co/datasets/CodeWarrior4Life/lattice-asr-fixtures/resolve/main/hello-en-30s.wav
```

As of 2026-05-11 this dataset returns HTTP 401 (private or not yet populated).
Two paths exist while the dataset is being prepared:

- **Local fallback (preferred until HF is public)**: drop the file at
  `tests/fixtures/audio/hello-en-30s.wav` before running perf tests. The
  `hello_en_30s_wav` fixture is lazy and only attempts a network download when
  the file is missing; if it exists locally, no HF traffic happens.
- **Public HF**: once the dataset is public the fixture downloads on first run
  and caches under `tests/fixtures/audio/`.

The same arrangement applies to `hello-en-2s.wav` (the smoke fixture).

## Baseline log

Pre-seeded with placeholder rows. Append a row per run; never overwrite. The most
recent passing measurement is the baseline of record for that host.

| Date | Host | Hardware | Gate | Measured RTF | Pass/Fail | Notes |
| ---- | ---- | -------- | ---- | ------------ | --------- | ----- |
|      | Morpheus | NVIDIA RTX 5080 (16 GB), CUDA 12.x | C2 | TBD | TBD |       |
|      | Morpheus | x86_64 8-core CPU | C3 | TBD | TBD |       |
|      | Trinity | Apple M5 Max (unified 128 GB) | C1 | TBD | TBD |       |
|      | Cypher | Apple Silicon | C1 | TBD | TBD |       |

## Notes

- CI integration is task **W6.2** (deferred). It requires a `workflow`-scoped
  GitHub token for the perf workflow to read CUDA / Apple Silicon runner state.
  Until W6.2 lands, perf gates run **manually** + on **opt-in** only.
- A gate regression (RTF below target) is a release-blocker. File a PVD issue
  against the engine adapter, not the harness.
- Sub-30s clips are not valid for these gates; warmup overhead pollutes the
  measurement for clips shorter than ~10s. Use `hello-en-30s.wav` exclusively.
- Warmup is always performed (`await eng.warmup()`) before the timed window so
  model load + first-batch JIT compilation are excluded from RTF.
