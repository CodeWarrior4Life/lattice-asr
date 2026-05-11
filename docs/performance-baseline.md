# Performance Baseline — lattice-asr v0.1

Hard-fail performance gates per [spec §12.3](../README.md) and plan task W6.1
(`02_Projects/Lattice/lattice-asr/Plans/2026-05-08 lattice-asr v0.1 - Implementation Plan.md`).
These gates protect the engine throughput contract that consumers (lattice-dictation,
lattice-meeting, MindPractice) depend on.

## Gates

| Gate | Host class | Engine | Audio | Target RTF | Hard-fail | Canonical host |
| ---- | ---------- | ------ | ----- | ---------- | --------- | -------------- |
| C1   | Apple Silicon (M-series) | `parakeet.cpp` | hello-en-30s.wav | > 10× | yes | Trinity (Apple M5 Max) |
| C2   | NVIDIA CUDA GPU (Linux) | `parakeet-tdt` (NeMo) | hello-en-30s.wav | > 50× | yes | Cypher (RTX 2070, Ubuntu 24.04) |
| C3   | x86_64 CPU (Linux server) | `faster-whisper` distil-large-v3 int8 | hello-en-30s.wav | > 2× | yes | Cypher (x86_64 Ubuntu 24.04) |

**Canonical hosts are production servers.** Dev workstations (e.g., Morpheus
running Windows) are useful for plumbing verification but are explicitly NOT
baselines of record — workstation noise (browsers, IDEs, sync daemons) makes
their RTF numbers untrustworthy for release decisions.

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
`apple_silicon` marker. Run on Trinity (Apple M5 Max).

### C2 — NVIDIA parakeet-tdt

```bash
LATTICE_ASR_PERF_RUN=1 pytest tests/perf/ -m perf -k c2 -v
```

Requires `nemo-toolkit[asr]` (the `nvidia` extra, Linux-only). Skip elsewhere via
the `nvidia_cuda` marker. Run on Cypher (RTX 2070, Ubuntu 24.04).

### C3 — CPU distil-large-v3 int8

```bash
LATTICE_ASR_PERF_RUN=1 pytest tests/perf/ -m perf -k c3 -v
```

Requires `faster-whisper` (the `whisper` extra). Run on Cypher (x86_64 Ubuntu
24.04, the canonical Linux server). Workstation runs (Morpheus / dev boxes)
are plumbing verification only, not baselines of record.

### All three (single command)

```bash
LATTICE_ASR_PERF_RUN=1 pytest tests/perf/ -m perf -v
```

Tests for engines unavailable on the current host raise `ImportError` /
`OSError` at import time; the marker-host-class pairing is informational, not
enforced by the test harness.

## Fixture status

The benchmark clips are committed in-repo at:

```
tests/fixtures/audio/hello-en-2s.wav    # 1.70s, 54 KB
tests/fixtures/audio/hello-en-30s.wav   # 30.00s, 960 KB
```

Both files are kept under version control (the broad `tests/fixtures/*.wav`
ignore in `.gitignore` doesn't match the `audio/` subdir). The `hello_en_2s_wav`
and `hello_en_30s_wav` fixtures in `tests/conftest.py` are lazy: if the file
already exists locally they skip the network entirely; otherwise they fall
back to the HF dataset URL.

The clips are synthesized from Windows SAPI (`System.Speech.Synthesis`) using
a deterministic English paragraph at 16 kHz mono 16-bit PCM. Determinism
matters because the perf gates pin against a known audio duration — any
re-generation must preserve the exact frame count (480 000 frames for the
30 s clip, sample rate 16 000 Hz).

**Original HF plan (not adopted):** the dataset
`CodeWarrior4Life/lattice-asr-fixtures` was originally planned as the network
source. Probed on 2026-05-11 with a valid READ-scope HF token and returned
HTTP 404 (repo not found, never created). Committing the WAVs in-repo
sidesteps the HF write-token rotation entirely; revisit if fixtures grow past
the ~10 MB hand-wave threshold.

## Baseline log

Pre-seeded with placeholder rows. Append a row per run; never overwrite. The most
recent passing measurement is the baseline of record for that host.

| Date | Host | Hardware | Gate | Measured RTF | Pass/Fail | Notes |
| ---- | ---- | -------- | ---- | ------------ | --------- | ----- |
| 2026-05-11 | Cypher | x86_64 Ubuntu 24.04, AMD Ryzen 9 3900X 12C/24T (Zen 2, no AVX-512), 64 GB RAM | C3 | 1.57× warm / 1.59× cold | **FAIL** | distil-large-v3 int8 underperforms the 2.0× target on Zen 2. Warm RTF < cold RTF (1.57 vs 1.59) confirms model load is not the bottleneck — steady-state CPU throughput is. Faster-whisper int8 leans heavily on AVX-512 (Zen 4+ / Intel Sapphire Rapids+). Owner decision: relax target, re-designate canonical C3 host, or accept that some hosts fail this gate. **Do not silently massage the target.** |
| 2026-05-11 | Cypher | NVIDIA RTX 2070 (8 GB, CUDA 7.5, Turing), driver 595.58.03 | C2 | — | **BLOCKED** | `ParakeetTdtEngine.transcribe` raises `NotImplementedError("ParakeetTdtEngine implemented in W3.2")` — the engine is a stub. Install validated end-to-end (nemo-toolkit 2.7.3, torch 2.11.0 + nvidia-cuda-* 13.x, faster-whisper 1.2.1); C2 baseline runs the moment W3.2 lands. RTX 2070 has no FP8 tensor cores, so the 50× target may still be tight on Turing once unblocked. |
|      | Trinity | Apple M5 Max (unified 128 GB) | C1 | TBD | TBD | Canonical C1 host. |

**C3 finding (2026-05-11):** Cypher's measured RTF 1.57× sits 22 % below the 2.0×
target, and Cypher is the canonical C3 host. The warm/cold delta (1.57 vs 1.59) is
within noise — model-download overhead is not masking real perf. Root cause is
the AMD Ryzen 9 3900X's lack of AVX-512: faster-whisper int8 on CPU is
AVX-512-bound on the inner GEMM kernels. Hosts with AVX-512 (Zen 4+, Intel
Sapphire Rapids+, Apple silicon via Accelerate) should clear 2.0× comfortably.
Decision surface (open to owner): (a) re-calibrate C3 target downward, (b)
re-designate canonical C3 host to an AVX-512 box, (c) keep current target and
accept that some hosts fail this gate. **Do not silently massage the target.**

**C2 finding (2026-05-11):** Engine implementation gap, not a hardware issue.
`ParakeetTdtEngine` is a W3.2 stub that raises `NotImplementedError`. NeMo
toolkit + torch + CUDA install successfully on Cypher (verified by import); the
C2 gate runs the moment W3.2 (NVIDIA Parakeet-TDT NeMo engine) lands. RTX 2070
(Turing, no FP8 tensor cores) may also struggle vs the 50× target once
unblocked — re-verify at that point.

**Plumbing verification (not a baseline of record):** 2026-05-11 on Morpheus
(Windows workstation), C3 cold run measured RTF 1.998 (failed by 0.002 — concurrent
model download); warm run passed in ~20 s wall clock. Validated fixture-load,
marker-filter, gate-evaluation, and engine-warmup path end-to-end. Morpheus does
not appear in canonical baselines. Note: Morpheus's 1.998 cold edges Cypher's
1.59 cold despite the workstation-vs-server inversion — the Cypher 3900X is a
2019 Zen 2 part and is likely the slower CPU here. Reinforces the canonical-C3
host re-think.

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
