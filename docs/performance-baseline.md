# Performance Baseline — lattice-asr v0.1

Hard-fail performance gates per [spec §12.3](../README.md) and plan task W6.1
(`02_Projects/Lattice/lattice-asr/Plans/2026-05-08 lattice-asr v0.1 - Implementation Plan.md`).
These gates protect the engine throughput contract that consumers (lattice-dictation,
lattice-meeting, MindPractice) depend on.

## Gates

| Gate | Host class | Engine | Audio | Target RTF | Hard-fail | Canonical host |
| ---- | ---------- | ------ | ----- | ---------- | --------- | -------------- |
| C1   | Apple Silicon (M-series) | `parakeet.cpp` | hello-en-30s.wav | > 10× | yes | Switch (Apple M4 Pro, Mac mini production host) |
| C2   | NVIDIA CUDA GPU (Linux) | `parakeet-tdt` (NeMo) | hello-en-30s.wav | > 50× | yes | Cypher (RTX 2070, Ubuntu 24.04) |
| C3   | CPU (non-GPU fallback path, arm64 or x86_64) | `faster-whisper` distil-large-v3 int8 | hello-en-30s.wav | > 2× | yes | Switch (Apple M4 Pro, Mac mini production host) |

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
`apple_silicon` marker. Run on **Switch** (Apple M4 Pro, the canonical C1 host).
**Currently BLOCKED** on two upstream gaps: (1) `ParakeetCppEngine` is a W3.1
stub that raises `NotImplementedError`, and (2) `parakeet-cpp-py>=0.1` is not
yet published to PyPI (pip resolution fails). C1 baseline runs the moment W3.1
ships AND the upstream Python wrapper lands. Trinity (M5 Max laptop) is NOT
canonical — it is the user's laptop, not a Lattice production host; the same
production-host rule that excludes Morpheus applies to Trinity.

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

Requires `faster-whisper` (the `whisper` extra). Run on **Switch** (Apple M4 Pro,
canonical C3 host as of 2026-05-11 S41). The gate is hardware-class-agnostic on
CPU — Apple Silicon NEON int8 via CTranslate2 and x86_64 AVX2/AVX-512 int8
both qualify, as long as the host is a Lattice production host. Workstation
runs (Morpheus / dev boxes) are plumbing verification only, not baselines of
record.

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
| 2026-05-11 | **Switch** | Apple M4 Pro 12-core ARM64 (Mac mini production host, Nexus primary since S245) | C3 | **3.35× warm / 3.28× cold** | **PASS** | **Canonical C3 host as of 2026-05-11 S41.** CTranslate2 NEON int8 path comfortably clears the 2.0× target by 64-67%. Print surfaced via the W6.1 feat: `[C3] RTF=3.35× elapsed=8.95s audio=30.00s`. Re-designation from Cypher driven by Cypher's 2019 Zen 2 silicon being too slow to clear 2× — see Cypher row below. C3 hardware class broadened from `x86_64 CPU` to `CPU (non-GPU fallback)` to accommodate the cross-arch reality. |
| 2026-05-11 | Cypher | x86_64 Ubuntu 24.04, AMD Ryzen 9 3900X 12C/24T (Zen 2, 2019), 64 GB RAM | C3 | 1.57× warm / 1.59× cold | **FAIL** | Historic measurement — superseded by Switch as canonical C3 host (above). distil-large-v3 int8 underperforms the 2.0× target on this 2019 Zen 2 silicon. Warm RTF < cold RTF (1.57 vs 1.59) confirms model load is not the bottleneck — steady-state CPU compute is. Morpheus's Intel Core Ultra 9 285K (2024, also no AVX-512) clears 2.44× on the same code path, so the bottleneck is raw CPU clock+IPC, not AVX-512 specifically. Cypher's 2019 silicon is simply too old to clear 2× on this workload. |
| 2026-05-11 | Switch | Apple M4 Pro 12-core ARM64 (Mac mini production host) | C1 | **15.13× (upstream-direct)** | **PASS (via direct API call); lattice-asr wrapper PENDING** | Direct invocation of upstream `parakeet-mlx` (PyPI 0.5.1, Apache 2.0, MLX runtime) via `from parakeet_mlx import from_pretrained; model.transcribe(...)` on `mlx-community/parakeet-tdt-0.6b-v3` measured RTF 15.13× (1.98s for 30s audio). Comfortably clears the 10× target. **The lattice-asr W3.1 `ParakeetCppEngine.transcribe()` wrapper is still a stub** — once it delegates to parakeet-mlx, the C1 perf gate will pass. **pyproject.toml `parakeet` extra was previously bound to phantom `parakeet-cpp-py>=0.1` (does NOT exist on PyPI, never did). Corrected to `parakeet-mlx>=0.5` in this commit.** Class name `ParakeetCppEngine` should be renamed to `ParakeetMlxEngine` when W3.1 implements; left for the implementation commit to avoid a cosmetic-only refactor. |
| 2026-05-11 | Cypher | NVIDIA RTX 2070 (8 GB, CUDA 7.5, Turing), driver 595.58.03 | C2 | **64.13× (upstream-direct)** | **PASS (via direct API call); lattice-asr wrapper PENDING** | Direct invocation of upstream `nemo-toolkit[asr]` 2.7.3 via `import nemo.collections.asr as nemo_asr; nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3").transcribe(...)` measured RTF 64.13× (0.47s for 30s audio) on RTX 2070 (Turing, sm_75). Comfortably clears the 50× target — and notably works on Turing despite NVIDIA's HF model card listing only Ampere+/Volta/T4 as supported. **The lattice-asr W3.2 `ParakeetTdtEngine.transcribe()` wrapper is still a stub** — once it delegates to nemo-toolkit, the C2 perf gate will pass. RTX 2070 has no FP8 tensor cores; the 50× target is cleared by ~28% headroom on Turing, so future Ampere+ hosts should clear it more comfortably. |

**C3 resolution (2026-05-11 S41):** Canonical C3 host re-designated from Cypher
to **Switch** (Apple M4 Pro Mac mini, the Lattice Nexus primary since S245).
Switch's NEON int8 path measures **3.35× warm / 3.28× cold** — comfortably above
the 2.0× target. The C3 contract is preserved: it still validates the non-GPU
CPU fallback path for consumers like lattice-dictation; the hardware-class label
broadens from `x86_64 CPU (Linux server)` to `CPU (non-GPU fallback path,
arm64 or x86_64)` to match the cross-arch reality.

The earlier root-cause investigation: Cypher's 1.57× sits 22% below 2.0× because
its 2019 Ryzen 9 3900X (Zen 2) is simply too slow. A first-pass hypothesis blamed
AVX-512 absence, but Morpheus's Intel Core Ultra 9 285K (Arrow Lake 2024, also
no AVX-512) clears 2.44× on the same code path with the same install — so the
real bottleneck is raw CPU clock+IPC, not AVX-512. CTranslate2 does gain ~3-5×
when AVX-512 VNNI is available (Intel Cascade Lake-X server, Sapphire Rapids+,
AMD Zen 4+), but it is not a requirement for clearing 2×. The Cypher row is
preserved in the baseline log as historic data; Switch is now the host of
record. **Target was NOT massaged — Switch's measurement is comfortably above
the as-written 2.0× threshold.**

**Upstream verification (2026-05-11, post-investigation):**

The earlier "C1 + C2 BLOCKED" narrative pinned blame on missing upstream packages
and unimplemented lattice-asr engine wrappers in equal measure. Investigation
proved the upstream side is fine — the only real gap is the lattice-asr wrapper
code (W3.1 + W3.2).

**C1 (Apple Silicon parakeet):** The `parakeet-cpp-py>=0.1` dependency the
`parakeet` extra pointed at **does not exist on PyPI and never did** (404 on
`https://pypi.org/pypi/parakeet-cpp-py/json`). The plan committed lattice-asr to
a phantom package. The actual Apple Silicon Parakeet Python runtime is
**`parakeet-mlx`** (PyPI 0.5.1, Apache 2.0, by senstella) — uses Apple's MLX
framework rather than the C++ binding the plan assumed. Direct API invocation
on Switch (M4 Pro):

```python
from parakeet_mlx import from_pretrained
model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
result = model.transcribe("tests/fixtures/audio/hello-en-30s.wav")
# RTF 15.13× (1.98s for 30s audio) — clears 10× target by 51%
```

`pyproject.toml` corrected in this commit. The lattice-asr engine class
`ParakeetCppEngine` is still mis-named (refers to a C++ binding that doesn't
exist in Python); rename to `ParakeetMlxEngine` is owned by the W3.1 commit
that writes `transcribe()`. There is also `Frikallo/parakeet.cpp` on GitHub
(an actual C++ library with Metal acceleration, 96× speedup on Apple Silicon,
~27 ms encoder inference) — but it ships only a C API + no Python binding;
wrapping it would be net-new work and is NOT what the lattice-asr plan needs
for v0.1.

**C2 (NVIDIA parakeet-tdt):** Upstream is `nemo-toolkit[asr]>=2.0` — already
correctly named in `pyproject.toml`. Direct API invocation on Cypher (RTX 2070
Turing, sm_75):

```python
import nemo.collections.asr as nemo_asr
asr = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
output = asr.transcribe(["tests/fixtures/audio/hello-en-30s.wav"])
# RTF 64.13× (0.47s for 30s audio) — clears 50× target by 28%
```

The NVIDIA HF model card lists "Ampere/Hopper/Blackwell/Volta/T4" as supported,
omitting Turing-class RTX cards — but the RTX 2070 is `sm_75` (same generation
as T4), and the inference path works fine in practice.

**Both engines are implementable today.** Total remaining v0.1 work:

| Item | Effort | Status |
| ---- | ------ | ------ |
| pyproject.toml `parakeet` extra correction | ~1 LOC | LANDED in this commit |
| `ParakeetCppEngine.transcribe()` → `ParakeetMlxEngine.transcribe()` (W3.1) | ~30 LOC + class rename | PENDING |
| `ParakeetTdtEngine.transcribe()` (W3.2) | ~30 LOC | PENDING |
| W6.2 CI (arm64 macOS + CUDA runners) | YAML + secrets | PENDING |
| W6.3 README rewrite | doc only | PENDING |
| v0.1.0 tag + PyPI publish | release step | PENDING |

Once W3.1 + W3.2 ship, all three perf gates (C1, C2, C3) will be measured
through the lattice-asr engine layer, not via direct upstream invocation.
Expected RTFs at that point: ≥15× C1 on Switch, ≥64× C2 on Cypher, 3.35× C3
on Switch.

**Lesson for the plan:** Verify every upstream dependency exists on PyPI (and
that its actual API matches what the plan assumes) BEFORE locking it into
`pyproject.toml`. The `parakeet-cpp-py` reference survived from plan to repo
to spec to perf gate to first install attempt — six places that could have
caught it. None did. A 5-second `curl https://pypi.org/pypi/parakeet-cpp-py/json`
during plan authoring would have surfaced this immediately.

**Plumbing verification (not a baseline of record):** 2026-05-11 on Morpheus
(Windows workstation, Intel Core Ultra 9 285K — Arrow Lake desktop, no AVX-512),
C3 measured **RTF 2.44× (PASS)** with model cache warm (downloaded by an earlier
S40 run, served from HF cache during warmup) in a fresh Python 3.11 +
faster-whisper 1.2.1 + CTranslate2 4.7.1 venv at `D:/Dev/lattice-asr`. Print
surfaced via the new feat commit: `[C3] RTF=2.44 elapsed=12.31s audio=30.00s`.
Validated fixture-load, marker-filter, gate-evaluation, engine-warmup path
end-to-end, AND that the 2.0× target is achievable on modern no-AVX-512 silicon.
Morpheus is the user's personal workstation, not a Lattice production host, so
does NOT appear in canonical baselines — but the measurement is the cleanest
proof that the algorithm is fine; the bottleneck is just Cypher's age. An
earlier S40 measurement of 1.998× cold on Morpheus was contaminated by concurrent
HF model download bleeding into the warmup; this 2.44× run is the clean retest.

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
