"""Language routing, engine availability, and the loud-fallback contract.

Everything here is a regression guard for the 2026-05-27..2026-09-01 outage in
which lattice-dictate transcribed every non-English recording as English and
said nothing. See `docs/` and lattice_asr/lid.py for the incident.
"""

from unittest.mock import AsyncMock, patch

import pytest

from lattice_asr.engines.base import TranscriptionEngine
from lattice_asr.engines.parakeet_tdt import PARAKEET_TDT_V3_LANGUAGES, ParakeetTdtEngine
from lattice_asr.engines.whisper_cpp import WhisperCppEngine
from lattice_asr.lid import UNDETERMINED, LidResult
from lattice_asr.telemetry import ListTelemetrySink
from lattice_asr.transcriber import Transcriber
from lattice_asr.types import EngineCapabilities, TranscriptionResult


class _FakeEngine(TranscriptionEngine):
    """Minimal engine with a declarable language set and availability."""

    def __init__(self, name, languages, *, available=True, detects=None):
        self._available = available
        self._detects = detects
        self.capabilities = EngineCapabilities(
            name=name,
            languages=frozenset(languages),
            streaming=True,
            requires_gpu=False,
            requires_apple_silicon=False,
            typical_rtfx=1.0,
        )

    @classmethod
    def is_available(cls):
        return getattr(cls, "_AVAILABLE", True)

    async def transcribe(self, audio_pcm, sample_rate, language):
        return TranscriptionResult(
            text="x",
            language=language or UNDETERMINED,
            confidence=0.0,
            engine_name=self.capabilities.name,
        )

    async def transcribe_streaming(self, audio_chunks, sample_rate, language):
        yield await self.transcribe(b"", sample_rate, language)


def _fake(name, languages, *, available=True):
    """Fresh subclass per engine: `is_available()` is a CLASSmethod, so two
    instances of one class could not disagree about availability."""
    cls = type(f"_Fake_{name}", (_FakeEngine,), {"_AVAILABLE": available})
    return cls(name, languages, available=available)


def _transcriber(engines, **kw):
    """Build a Transcriber with a hand-made registry, bypassing hardware probing."""
    with (
        patch("lattice_asr.transcriber.detect_hardware"),
        patch("lattice_asr.transcriber._build_engine_registry", return_value=engines),
    ):
        return Transcriber(**kw)


# --------------------------------------------------------------------------
# capability-based routing (replaces the hard-coded "en" / "multi" strings)
# --------------------------------------------------------------------------


@pytest.mark.r_tier
def test_specific_language_picks_narrowest_capable_engine():
    """A specialist wins on its own language -- that is why the fast path exists."""
    en_only = _fake("parakeet", {"en"})
    broad = _fake("whisper", {"en", "es", "fr", "ja"})
    t = _transcriber({"en": en_only, "multi": broad})
    assert t._select_engine("en") is en_only
    assert t._select_engine("es") is broad


@pytest.mark.r_tier
def test_unknown_language_routes_to_broadest_not_narrowest():
    """`None` means "I don't know", and unknown must go to the CAPABLE engine.

    The old code did `route = "en" if language == "en" else "multi"` after
    coercing an unknown language to `default_language`, i.e. unknown audio went
    to the English-only engine. This is the inversion.
    """
    en_only = _fake("parakeet", {"en"})
    broad = _fake("whisper", {"en", "es", "fr", "ja"})
    t = _transcriber({"en": en_only, "multi": broad})
    assert t._select_engine(None) is broad
    assert t._select_engine(UNDETERMINED) is broad


@pytest.mark.r_tier
def test_unavailable_engine_is_not_selected():
    """The link failure mode: the multilingual route's package was never installed.

    Routing Spanish there would raise ModuleNotFoundError rather than transcribe.
    An engine that cannot run must lose to one that can.
    """
    installed = _fake("parakeet", {"en", "es"})
    not_installed = _fake("whisper", {"en", "es", "fr", "ja"}, available=False)
    t = _transcriber({"en": installed, "multi": not_installed})
    assert type(not_installed).is_available() is False
    assert type(installed).is_available() is True
    assert t._select_engine("es") is installed
    assert t._select_engine(None) is installed


@pytest.mark.r_tier
def test_language_no_engine_supports_falls_back_to_broadest_with_warning(caplog):
    narrow = _fake("parakeet", {"en"})
    mid = _fake("whisper-eu", {"en", "es"})
    t = _transcriber({"en": narrow, "multi": mid})
    with caplog.at_level("WARNING"):
        assert t._select_engine("ja") is mid
    assert any("no loaded engine declares support" in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# the loud-fallback contract  (brief §5: fail closed, never fail accurate)
# --------------------------------------------------------------------------


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_detector_exception_is_warned_not_swallowed(caplog):
    """The exact 2026-05-27 shape: the detector raises.

    Silero's `torch.hub` callable was removed upstream, so `detect()` raised on
    every call. The old code let that become `default_language` with no log line,
    which is why nobody noticed for three months.
    """
    broad = _fake("whisper", {"en", "es"})
    t = _transcriber({"en": broad, "multi": broad})
    t._detector = broad
    with (
        patch.object(
            broad,
            "detect_language",
            new=AsyncMock(side_effect=RuntimeError("hub callable removed")),
        ),
        caplog.at_level("WARNING"),
    ):
        language, detection, warning = await t._resolve_language(b"\x00" * 3200, 16000, None)

    assert language is None, "a failed detector must not yield a language"
    assert detection is None
    assert "hub callable removed" in warning
    assert any("language detection FAILED" in r.message for r in caplog.records)


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_no_detector_available_is_reported():
    """A host whose only engine self-detects has no LID -- that must be stated."""
    auto = _fake("parakeet", {"en", "es"})
    t = _transcriber({"en": auto, "multi": auto})
    t._detector = None
    language, detection, warning = await t._resolve_language(b"\x00" * 3200, 16000, None)
    assert language is None
    assert warning is not None and "no loaded engine can identify language" in warning


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_explicit_language_bypasses_detection_entirely():
    broad = _fake("whisper", {"en", "es"})
    t = _transcriber({"en": broad, "multi": broad})
    t._detector = broad
    detect = AsyncMock(return_value=LidResult("es", 0.99))
    with patch.object(broad, "detect_language", new=detect):
        language, detection, warning = await t._resolve_language(b"\x00" * 3200, 16000, "fr")
    assert (language, detection, warning) == ("fr", None, None)
    detect.assert_not_awaited()


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_undetermined_detection_yields_no_language():
    broad = _fake("whisper", {"en", "es"})
    t = _transcriber({"en": broad, "multi": broad})
    t._detector = broad
    with patch.object(
        broad, "detect_language", new=AsyncMock(return_value=LidResult(UNDETERMINED, 0.0))
    ):
        language, _detection, warning = await t._resolve_language(b"\x00" * 3200, 16000, None)
    assert language is None
    assert warning is not None


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_lid_disabled_does_not_silently_mean_english():
    """`lid.enabled=false` + `language=None` must auto-detect, not assume English."""
    broad = _fake("whisper", {"en", "es"})
    t = _transcriber({"en": broad, "multi": broad}, default_language="en")
    t._lid_enabled = False
    t._detector = None
    language, _d, warning = await t._resolve_language(b"\x00" * 3200, 16000, None)
    assert language is None
    assert warning is not None and "disabled" in warning


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_warning_reaches_both_transcript_and_telemetry():
    """Acceptance criterion: a wrong-language result must be LOUD in the record."""
    sink = ListTelemetrySink()
    broad = _fake("whisper", {"en", "es"})
    t = _transcriber({"en": broad, "multi": broad}, telemetry_sink=sink)
    t._detector = broad
    with patch.object(broad, "detect_language", new=AsyncMock(return_value=LidResult("es", 0.10))):
        result = await t.transcribe(b"\x00" * 3200, 16000, language=None)
    assert result.language_warning is not None
    assert sink.records[0].language_warning == result.language_warning
    assert sink.records[0].language_confidence == 0.10


# --------------------------------------------------------------------------
# engine honesty
# --------------------------------------------------------------------------


@pytest.mark.r_tier
def test_parakeet_declares_spanish_and_is_not_english_only():
    """It transcribed Spanish for months while declaring `languages={"en"}`."""
    caps = ParakeetTdtEngine().capabilities
    assert "es" in caps.languages
    assert caps.languages == PARAKEET_TDT_V3_LANGUAGES
    assert len(caps.languages) == 25


@pytest.mark.r_tier
def test_parakeet_language_set_is_not_scraped_from_the_tokenizer():
    """The vocab carries 183 ISO tokens; only 25 languages are actually trained.

    Guards against "the tokenizer has a token for it, so we support it" -- a
    vocab entry is a flag, not a state.
    """
    assert len(PARAKEET_TDT_V3_LANGUAGES) == 25
    for absent in ("ja", "zh", "ar", "ko", "he", "hi"):
        assert absent not in PARAKEET_TDT_V3_LANGUAGES


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_parakeet_does_not_fabricate_language_or_confidence():
    """It used to return `language="en", confidence=1.0` unconditionally."""
    e = ParakeetTdtEngine()

    class _Hyp:
        text = "hola que tal"

    class _Model:
        def transcribe(self, paths, **kw):
            return [_Hyp()]

    with patch.object(e, "_ensure_model", return_value=_Model()):
        r = await e.transcribe(b"\x00" * 3200, 16000, None)
    assert r.text == "hola que tal"
    assert r.language == UNDETERMINED, "must not claim English for auto-detected audio"
    assert r.confidence == 0.0, "must not fabricate certainty it does not have"


@pytest.mark.r_tier
def test_unimplemented_engine_reports_unavailable():
    """WhisperCppEngine.transcribe() raises NotImplementedError; never route to it."""
    assert WhisperCppEngine.is_available() is False


@pytest.mark.r_tier
def test_missing_packages_names_the_dependency():
    assert ParakeetTdtEngine.required_packages == ("nemo",)
    from lattice_asr.engines.faster_whisper import FasterWhisperEngine

    assert FasterWhisperEngine.required_packages == ("faster_whisper",)


@pytest.mark.r_tier
def test_base_detect_language_default_is_none_not_a_guess():
    """The default "I cannot detect" answer must be None, never a language."""
    assert TranscriptionEngine.detect_language.__doc__ is not None
    e = _fake("x", {"en"})
    import asyncio as _a

    assert _a.run(TranscriptionEngine.detect_language(e, b"", 16000)) is None
