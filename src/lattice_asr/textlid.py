"""Language identification from TRANSCRIBED TEXT — the last-resort detector.

Some engines identify language and cannot tell you which they picked. Parakeet
TDT is the case that forced this module: it is an RNNT model that auto-detects
internally, transcribes Spanish correctly, and exposes no language field at all.
MEASURED 2026-09-01: its vocab carries ``<|predict_lang|>`` and ``<|es|>`` tokens,
but a decoded hypothesis contains NO language token -- ``y_sequence`` begins
directly at content. So the language genuinely cannot be read out of the decode.

The result was transcripts stamped ``language: und`` sitting on top of 480 words
of unmistakable Spanish. Correct, honest, and useless to anyone filtering
recordings by language.

So: identify the language of the OUTPUT when nothing could identify the INPUT.
This is weaker evidence than audio LID and is labelled as such --
``TranscriptionResult.language_source == "transcript-text"`` -- because it
describes what the engine WROTE, not what was SPOKEN. Those differ exactly when
the engine mistranslates, which is the failure mode that started all this.

Cheap enough to be unconditional: ~0.9ms on a 300-character transcript, pure
Python, no model download, no GPU, and nothing in front of the latency-critical
dictation path. Optional dependency (``lattice-asr[textlid]``); absent, this
returns None and the language stays ``und`` rather than becoming a guess.
"""

from __future__ import annotations

import logging

from lattice_asr.lid import LidResult

logger = logging.getLogger(__name__)

# Below this, text LID is not reliable enough to be worth stating. Short PTT
# dictations ("open the file") land here and correctly get no language rather
# than a coin-flip -- the whole point is to stop inventing language decisions.
MIN_CHARS_FOR_TEXT_LID = 40

_UNAVAILABLE_LOGGED = False


def detect_language_from_text(text: str) -> LidResult | None:
    """Identify the language of a transcript. None when unavailable or too short.

    None is a real answer here and callers must leave the language undetermined
    rather than substituting a default.
    """
    global _UNAVAILABLE_LOGGED

    if not text or len(text.strip()) < MIN_CHARS_FOR_TEXT_LID:
        return None

    try:
        from langdetect import DetectorFactory, detect_langs  # type: ignore[import-untyped]
    except ImportError:
        if not _UNAVAILABLE_LOGGED:
            logger.info(
                "lattice-asr: text language-id unavailable (pip install "
                "'lattice-asr[textlid]'); transcripts from engines that cannot "
                "report a language will stay 'und'"
            )
            _UNAVAILABLE_LOGGED = True
        return None

    # langdetect is randomised per-process unless seeded, which would make the
    # same transcript report different languages across runs.
    DetectorFactory.seed = 0

    try:
        ranked = detect_langs(text)
    except Exception as exc:  # noqa: BLE001 - never fail a transcription over metadata
        logger.warning("lattice-asr: text language-id failed (%s: %s)", type(exc).__name__, exc)
        return None

    if not ranked:
        return None
    best = ranked[0]
    return LidResult(language=str(best.lang), confidence=float(best.prob))
