"""Language identification — result type and the shared "undetermined" contract.

**Silero LID was DELETED here on 2026-09-01.** What used to live in this module
was ``SileroLid``, which called::

    torch.hub.load("snakers4/silero-vad", "silero_lang_detector_95")

Upstream removed that callable. The hub load raised on every invocation from
2026-05-27 onward, and because ``Transcriber`` caught nothing and simply fell
back to ``default_language``, the failure was **silent** -- every
``language=None`` call quietly became English for three months.

It is deleted rather than repaired because nothing needed it: the engines can
identify language themselves (faster-whisper exposes native detection; Parakeet
TDT v3 auto-detects internally), so a separate model, a ``torch.hub`` download
and the 1.5s-slice latency contract were all pure cost.

The lesson that outlived the model, and the reason ``UNDETERMINED`` exists: a
detector that cannot tell you the language must SAY SO, not hand back a
plausible default. See ``Transcriber._resolve_language``.
"""

from __future__ import annotations

from dataclasses import dataclass

# ISO 639-2 "und". Returned when an engine genuinely cannot determine the
# language -- never substitute a default here, that is the bug this module
# is a monument to.
UNDETERMINED = "und"


@dataclass(frozen=True)
class LidResult:
    """A language identification outcome.

    ``confidence`` is the detector's own probability. A result of
    ``LidResult(UNDETERMINED, 0.0)`` is the honest answer for "I cannot tell",
    and callers must route it to an engine that can handle any language rather
    than guessing.
    """

    language: str
    confidence: float

    @property
    def is_determined(self) -> bool:
        return self.language != UNDETERMINED
