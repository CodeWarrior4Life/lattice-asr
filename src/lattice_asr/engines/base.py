"""TranscriptionEngine ABC — adapter base. Spec §4, §6."""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from lattice_asr.lid import LidResult
from lattice_asr.types import EngineCapabilities, TranscriptionResult

# How much leading audio a language pre-pass may look at. Detection quality
# plateaus well before this; the bound exists so detect_language() cost stays
# independent of how long the caller's recording is.
LANGUAGE_DETECT_SECONDS = 30.0


class EngineUnavailableError(RuntimeError):
    """No usable engine exists for a request, with the reason and the remedy.

    Raised in place of the bare ``ModuleNotFoundError`` that used to surface from
    deep inside an engine's lazy ``_ensure_model()`` on the first real call --
    long after engine selection, with nothing naming which optional extra was
    missing.
    """


class TranscriptionEngine(ABC):
    """Adapter for one ASR engine."""

    capabilities: EngineCapabilities

    #: Import names this engine needs at runtime. Declared rather than probed so
    #: availability is answerable without importing (and paying for) heavy deps.
    required_packages: tuple[str, ...] = ()

    @abstractmethod
    async def transcribe(
        self,
        audio_pcm: bytes,
        sample_rate: int,
        language: str | None,
    ) -> TranscriptionResult: ...

    @abstractmethod
    async def transcribe_streaming(
        self,
        audio_chunks: AsyncIterator[bytes],
        sample_rate: int,
        language: str | None,
    ) -> AsyncIterator[TranscriptionResult]:
        if False:  # makes this an async generator (subtype of AsyncIterator)
            yield  # type: ignore[unreachable]  # pragma: no cover

    @classmethod
    def missing_packages(cls) -> tuple[str, ...]:
        """Which of `required_packages` are not importable in this interpreter."""
        return tuple(p for p in cls.required_packages if importlib.util.find_spec(p) is None)

    @classmethod
    def is_available(cls) -> bool:
        """True when this engine could actually run here.

        Engine selection MUST consult this. A registry entry is a declaration,
        not a working engine: on link, `FasterWhisperEngine` sat in the registry
        as the multilingual route for months while `faster_whisper` was not
        installed at all, so routing any non-English audio to it would have
        raised rather than transcribed. A flag is not a state.
        """
        return not cls.missing_packages()

    async def warmup(self) -> None:  # noqa: B027 - optional override; default no-op intentional
        pass

    async def detect_language(  # noqa: B027 - optional capability; default "cannot" intentional
        self, audio_pcm: bytes, sample_rate: int
    ) -> LidResult | None:
        """Identify the spoken language, or return ``None`` if this engine cannot.

        ``None`` means "no opinion" and is the correct answer for an engine with
        no detection capability -- it is NOT an error and NOT a vote for the
        default language. Callers must treat it as "route to something that can
        handle any language" (see ``Transcriber._resolve_language``); the
        three-month silent-English outage this interface exists to prevent was
        caused precisely by turning "I don't know" into "probably English".

        Implementations should bound their work to ``LANGUAGE_DETECT_SECONDS``.
        """
        return None
