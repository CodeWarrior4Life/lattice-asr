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
    # Two engines: a lone detecting engine skips the separate pass (see
    # test_single_detecting_engine_skips_the_separate_detection_pass).
    t = _transcriber({"en": _fake("parakeet", {"en"}), "multi": broad})
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
    # Two engines: a lone detecting engine skips the separate pass (see
    # test_single_detecting_engine_skips_the_separate_detection_pass).
    t = _transcriber({"en": _fake("parakeet", {"en"}), "multi": broad})
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
    # Two engines: a lone detecting engine skips the separate pass (see
    # test_single_detecting_engine_skips_the_separate_detection_pass).
    t = _transcriber({"en": _fake("parakeet", {"en"}), "multi": broad}, telemetry_sink=sink)
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


# --------------------------------------------------------------------------
# text LID -- labelling a transcript the engine could not label itself
# --------------------------------------------------------------------------


@pytest.mark.r_tier
def test_text_lid_identifies_the_recovered_spanish():
    """The real transcript the fix recovers must be identifiable as Spanish.

    Parakeet transcribes this correctly but cannot say in what language, which
    left 480 words of unmistakable Spanish stamped `und`.
    """
    from lattice_asr.textlid import detect_language_from_text

    es = (
        "su familia y tenga la capacidad financiera para eso habría que estas "
        "hablando conmigo un un gringo aquí que que no he hablado en español en casi "
        "tres meses explícame más suave por favor una cosa para los países que estás "
        "enfocando y como me estás diciendo que hay que estructurar todo"
    )
    r = detect_language_from_text(es)
    assert r is not None
    assert r.language == "es"
    assert r.confidence > 0.9


@pytest.mark.r_tier
def test_text_lid_declines_on_short_text_instead_of_guessing():
    """A three-word PTT dictation must get NO language, not a coin flip."""
    from lattice_asr.textlid import detect_language_from_text

    assert detect_language_from_text("open the file") is None
    assert detect_language_from_text("") is None
    assert detect_language_from_text("   ") is None


@pytest.mark.r_tier
def test_text_lid_is_deterministic():
    """langdetect randomises per-process unless seeded; the same transcript must
    not report different languages across runs."""
    from lattice_asr.textlid import detect_language_from_text

    text = (
        "mira generalmente lo que hago yo es leer el proyecto que me envía mi amigo "
        "lo principal es que ya tenga todos los permisos de construcción"
    )
    results = {detect_language_from_text(text).language for _ in range(5)}  # type: ignore[union-attr]
    assert results == {"es"}


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_undetermined_engine_result_is_labelled_from_the_transcript():
    """End of the chain: Parakeet's `und` becomes `es`, labelled as text-derived."""
    spanish = (
        "no podemos ofrecerselas a cualquier persona porque pues estamos perdiendo "
        "tiempo entonces no sé si ellos están ofreciendo algo más también más bajo"
    )

    class _AutoDetectEngine(_FakeEngine):
        async def transcribe(self, audio_pcm, sample_rate, language):
            # exactly what ParakeetTdtEngine now returns: text, but no language
            return TranscriptionResult(
                text=spanish,
                language=UNDETERMINED,
                confidence=0.0,
                engine_name="parakeet-tdt",
            )

    e = type("_Auto", (_AutoDetectEngine,), {"_AVAILABLE": True})("parakeet", {"en", "es"})
    sink = ListTelemetrySink()
    t = _transcriber({"en": e, "multi": e}, telemetry_sink=sink)
    t._detector = None

    result = await t.transcribe(b"\x00" * 3200, 16000, language=None)

    assert result.language == "es"
    assert result.confidence > 0.9
    assert result.language_source == "transcript-text", "provenance must be labelled"
    assert "inferred from the TRANSCRIPT TEXT" in result.language_warning
    assert sink.records[0].language_source == "transcript-text"
    assert sink.records[0].language_detected == "es"


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_pinned_language_is_labelled_requested_and_never_text_guessed():
    """An explicit request is authoritative; nothing may second-guess it."""
    e = _fake("whisper", {"en", "es"})
    t = _transcriber({"en": e, "multi": e})
    result = await t.transcribe(b"\x00" * 3200, 16000, language="en")
    assert result.language == "en"
    assert result.language_source == "requested"


@pytest.mark.r_tier
def test_text_lid_absent_yields_none_not_a_guess(monkeypatch):
    """Without the optional extra the language stays undetermined."""
    import builtins

    import lattice_asr.textlid as tl

    real_import = builtins.__import__

    def _no_langdetect(name, *a, **kw):
        if name == "langdetect":
            raise ImportError("not installed")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _no_langdetect)
    long_text = "esto es una frase suficientemente larga para superar el umbral de caracteres"
    assert tl.detect_language_from_text(long_text) is None


# --------------------------------------------------------------------------
# a checkpoint's NAME is capability; the tokenizer's language list is not
# --------------------------------------------------------------------------


@pytest.mark.r_tier
def test_distil_models_declare_english_only():
    """distil-large-v3 was the default on EVERY route while declaring 99 languages.

    MEASURED 2026-09-01 on 30s of Spanish: it reports en@0.9463 and translates
    rather than transcribes. Declaring the tokenizer's list as capability is what
    let the multilingual route be served by an English-only model.
    """
    from lattice_asr.engines.faster_whisper import FasterWhisperEngine

    for name in ("distil-large-v3", "distil-small.en", "distil-whisper/distil-large-v3"):
        caps = FasterWhisperEngine(model=name).capabilities
        assert caps.languages == frozenset({"en"}), name


@pytest.mark.r_tier
def test_dot_en_models_declare_english_only():
    from lattice_asr.engines.faster_whisper import FasterWhisperEngine

    for name in ("medium.en", "base.en", "tiny.en"):
        assert FasterWhisperEngine(model=name).capabilities.languages == frozenset({"en"}), name


@pytest.mark.r_tier
def test_real_whisper_models_declare_multilingual():
    from lattice_asr.engines.faster_whisper import FasterWhisperEngine

    for name in ("large-v3", "medium", "small"):
        langs = FasterWhisperEngine(model=name).capabilities.languages
        assert "es" in langs and "ja" in langs and len(langs) > 50, name


@pytest.mark.r_tier
def test_explicit_languages_override_the_name_heuristic():
    """A checkpoint the name heuristic misjudges must be declarable."""
    from lattice_asr.engines.faster_whisper import FasterWhisperEngine

    e = FasterWhisperEngine(
        model="distil-some-multilingual-fork", languages=frozenset({"en", "es"})
    )
    assert e.capabilities.languages == frozenset({"en", "es"})


@pytest.mark.r_tier
def test_multilingual_route_is_never_an_english_only_model():
    """THE regression guard for the second layer of this defect.

    Whatever hardware profile is detected, the engine that non-English audio lands
    on must actually declare non-English support.
    """
    from lattice_asr.hardware import HardwareProfile
    from lattice_asr.transcriber import _build_engine_registry

    profiles = [
        HardwareProfile(
            os="linux",
            cpu_arch="x86_64",
            apple_silicon=False,
            nvidia_cuda=True,
            cuda_capability=(8, 6),
            total_ram_gb=32.0,
            cpu_cores=16,
        ),
        HardwareProfile(
            os="linux",
            cpu_arch="x86_64",
            apple_silicon=False,
            nvidia_cuda=False,
            cuda_capability=None,
            total_ram_gb=16.0,
            cpu_cores=8,
        ),
    ]
    for hw in profiles:
        registry = _build_engine_registry(hw, None)
        multi = registry["multi"]
        assert multi.capabilities.languages != frozenset({"en"}), (
            f"multilingual route on {hw.os}/cuda={hw.nvidia_cuda} is an English-only "
            f"engine ({multi.capabilities.name}); non-English audio would be translated"
        )
        assert "es" in multi.capabilities.languages


@pytest.mark.r_tier
def test_forced_faster_whisper_is_multilingual():
    from lattice_asr.hardware import HardwareProfile
    from lattice_asr.transcriber import _build_engine_registry

    hw = HardwareProfile(
        os="linux",
        cpu_arch="x86_64",
        apple_silicon=False,
        nvidia_cuda=False,
        cuda_capability=None,
        total_ram_gb=16.0,
        cpu_cores=8,
    )
    registry = _build_engine_registry(hw, "faster-whisper")
    assert "es" in registry["multi"].capabilities.languages


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_audio_detected_language_is_credited_to_the_detector_not_the_engine():
    """A detected language passed to an engine comes back echoed; do not relabel it.

    MEASURED regression: a real Spanish dictation reported
    `language=es, language_source=engine` when the AUDIO DETECTOR had found it and
    the engine merely echoed the argument. That laundered an audio-LID finding as
    the engine's own and discarded the detector's confidence with it.
    """

    class _Echo(_FakeEngine):
        async def transcribe(self, audio_pcm, sample_rate, language):
            return TranscriptionResult(
                text="hola que tal amigo",
                language=language or UNDETERMINED,  # engines echo the argument
                confidence=0.0,
                engine_name="parakeet-tdt",
            )

    e = type("_E", (_Echo,), {"_AVAILABLE": True})("parakeet", {"en", "es"})
    sink = ListTelemetrySink()
    t = _transcriber({"en": _fake("parakeet", {"en"}), "multi": e}, telemetry_sink=sink)
    t._detector = e
    with patch.object(e, "detect_language", new=AsyncMock(return_value=LidResult("es", 0.9468))):
        result = await t.transcribe(b"\x00" * 3200, 16000, language=None)

    assert result.language == "es"
    assert result.language_source == "audio-lid", (
        "an audio detection must not be credited to the engine"
    )
    assert result.confidence == pytest.approx(0.9468)
    assert sink.records[0].language_confidence == pytest.approx(0.9468)


# --- 2026-09-28: a lone detecting engine must not run the encoder twice ---


class _CountingDetector(_FakeEngine):
    detect_calls = 0

    async def detect_language(self, audio_pcm, sample_rate):
        type(self).detect_calls += 1
        return LidResult("es", 0.99)


@pytest.mark.asyncio
async def test_single_detecting_engine_skips_the_separate_detection_pass():
    cls = type("_Solo", (_CountingDetector,), {"detect_calls": 0})
    solo = cls("whisper", {"en", "es"})
    t = _transcriber({"en": solo, "multi": solo})
    assert t._detector is solo
    result = await t.transcribe(b"\x00" * 3200, 16000)
    assert cls.detect_calls == 0
    assert result.language_source != "audio-lid"


@pytest.mark.asyncio
async def test_detection_still_runs_when_a_second_engine_could_be_chosen():
    """Negative twin: with a specialist to route to, detection decides the route."""
    cls = type("_Duo", (_CountingDetector,), {"detect_calls": 0})
    broad = cls("whisper", {"en", "es"})
    t = _transcriber({"en": _fake("parakeet", {"en"}), "multi": broad})
    result = await t.transcribe(b"\x00" * 3200, 16000)
    assert cls.detect_calls == 1
    assert result.language == "es" and result.language_source == "audio-lid"
