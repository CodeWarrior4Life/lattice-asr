"""lattice-asr — hardware-adaptive multilingual ASR library for the Lattice family.

v0.1 scaffold. Spec at vault `02_Projects/Lattice/lattice-asr/Specifications/
2026-04-27 lattice-asr v1 - Design Spec.md`.

Implementation lands in subsequent sessions. The public surface in v0.1.0
will expose:

- `Transcriber` — main entry point, hardware-probed engine selection.
- `TranscriptionEngine` — Protocol for engine plug-ins (Whisper, Parakeet, Remote).
- `TranscriptionResult` / `TranscriptSegment` — wire-format dataclasses.
- `transcribe()` / `transcribe_streaming()` — high-level helpers.
"""

from __future__ import annotations

__version__ = "0.1.0.dev0"
__all__ = ["__version__"]
