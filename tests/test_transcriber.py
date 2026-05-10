"""Transcriber class — LID routing + diarization scaffold + telemetry. Spec §4."""

from unittest.mock import AsyncMock, patch

import pytest

from lattice_asr import ListTelemetrySink, Transcriber
from lattice_asr.lid import LidResult
from lattice_asr.types import TranscriptionResult


@pytest.fixture
def cpu_only_hw():
    from lattice_asr.hardware import HardwareProfile

    return HardwareProfile(
        os="linux",
        cpu_arch="x86_64",
        apple_silicon=False,
        nvidia_cuda=False,
        cuda_capability=None,
        total_ram_gb=16.0,
        cpu_cores=8,
    )


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_transcribe_routes_explicit_language(cpu_only_hw):
    sink = ListTelemetrySink()
    with patch("lattice_asr.transcriber.detect_hardware", return_value=cpu_only_hw):
        t = Transcriber(telemetry_sink=sink)
        fake_result = TranscriptionResult(
            text="hi",
            language="en",
            confidence=0.95,
            engine_name="faster-whisper",
            segments=(),
            speaker_segments=(),
            audio_duration_ms=1000,
            duration_ms=50,
        )
        with patch.object(t._engines["en"], "transcribe", new=AsyncMock(return_value=fake_result)):
            r = await t.transcribe(b"\x00\x00", language="en")
    assert r.text == "hi"
    assert sink.records[0].language_requested == "en"
    assert sink.records[0].language_detected == "en"


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_transcribe_uses_lid_when_language_none(cpu_only_hw):
    sink = ListTelemetrySink()
    with patch("lattice_asr.transcriber.detect_hardware", return_value=cpu_only_hw):
        t = Transcriber(telemetry_sink=sink)
        fake_result = TranscriptionResult(
            text="hola",
            language="es",
            confidence=0.9,
            engine_name="faster-whisper",
            segments=(),
            speaker_segments=(),
            audio_duration_ms=1000,
            duration_ms=50,
        )
        with (
            patch.object(
                t._lid,
                "detect",
                new=AsyncMock(return_value=LidResult(language="es", confidence=0.95)),
            ),
            patch.object(
                t._engines["multi"], "transcribe", new=AsyncMock(return_value=fake_result)
            ),
        ):
            await t.transcribe(b"\x00\x00\x00\x00" * 24000, language=None)
    assert sink.records[0].language_detected == "es"
    assert sink.records[0].language_requested is None


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_low_lid_confidence_falls_back_to_default_language(cpu_only_hw):
    sink = ListTelemetrySink()
    with patch("lattice_asr.transcriber.detect_hardware", return_value=cpu_only_hw):
        t = Transcriber(default_language="en", telemetry_sink=sink)
        fake_result = TranscriptionResult(
            text="...",
            language="en",
            confidence=0.5,
            engine_name="faster-whisper",
            segments=(),
            speaker_segments=(),
            audio_duration_ms=1000,
            duration_ms=50,
        )
        with (
            patch.object(
                t._lid,
                "detect",
                new=AsyncMock(return_value=LidResult(language="??", confidence=0.3)),
            ),
            patch.object(t._engines["en"], "transcribe", new=AsyncMock(return_value=fake_result)),
        ):
            await t.transcribe(b"\x00" * 96000, language=None)
    assert sink.records[0].language_detected == "en"  # fell back to default


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_force_engine_override(cpu_only_hw):
    with patch("lattice_asr.transcriber.detect_hardware", return_value=cpu_only_hw):
        t = Transcriber(force_engine="faster-whisper")
    assert t._engines["en"].capabilities.name == "faster-whisper"


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_tenant_id_propagates_to_record(cpu_only_hw):
    sink = ListTelemetrySink()
    with patch("lattice_asr.transcriber.detect_hardware", return_value=cpu_only_hw):
        t = Transcriber(telemetry_sink=sink)
        fake_result = TranscriptionResult(
            text="hi",
            language="en",
            confidence=0.95,
            engine_name="faster-whisper",
            segments=(),
            speaker_segments=(),
            audio_duration_ms=1000,
            duration_ms=50,
        )
        with patch.object(t._engines["en"], "transcribe", new=AsyncMock(return_value=fake_result)):
            await t.transcribe(b"\x00", language="en", tenant_id="alice")
    assert sink.records[0].tenant_id == "alice"


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_diarize_requires_enable_diarization(cpu_only_hw):
    with patch("lattice_asr.transcriber.detect_hardware", return_value=cpu_only_hw):
        t = Transcriber(enable_diarization=False)
        with pytest.raises(ValueError, match="enable_diarization"):
            await t.transcribe(b"\x00", language="en", diarize=True)


@pytest.mark.r_tier
def test_enable_diarization_raises_in_v01(cpu_only_hw):
    """v0.1 contract: enable_diarization=True must raise NotImplementedError until W5."""
    with patch("lattice_asr.transcriber.detect_hardware", return_value=cpu_only_hw):
        with pytest.raises(NotImplementedError, match="W5"):
            Transcriber(enable_diarization=True)


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_language_detected_reflects_engine_truth_not_request(cpu_only_hw):
    """When caller requests 'en' but engine detects something else, telemetry
    must record the engine's detection — not echo the caller's request."""
    sink = ListTelemetrySink()
    with patch("lattice_asr.transcriber.detect_hardware", return_value=cpu_only_hw):
        t = Transcriber(telemetry_sink=sink)
        fake_result = TranscriptionResult(
            text="hola",
            language="es",
            confidence=0.9,
            engine_name="faster-whisper",
            segments=(),
            speaker_segments=(),
            audio_duration_ms=1000,
            duration_ms=50,
        )
        with patch.object(t._engines["en"], "transcribe", new=AsyncMock(return_value=fake_result)):
            await t.transcribe(b"\x00\x00", language="en")
    # Caller requested 'en' but engine returned 'es' — record must show engine truth.
    assert sink.records[0].language_requested == "en"
    assert sink.records[0].language_detected == "es"


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_streaming_diarize_requires_enable_diarization(cpu_only_hw):
    with patch("lattice_asr.transcriber.detect_hardware", return_value=cpu_only_hw):
        t = Transcriber()  # enable_diarization defaults to False

        async def _empty_chunks():
            if False:
                yield b""

        with pytest.raises(ValueError, match="enable_diarization"):
            async for _ in t.transcribe_streaming(_empty_chunks(), language="en", diarize=True):
                pass


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_transcribe_streaming_yields_partials_and_records_each(cpu_only_hw):
    """Streaming yields each engine partial and records one AsrCallRecord per partial."""
    sink = ListTelemetrySink()
    with patch("lattice_asr.transcriber.detect_hardware", return_value=cpu_only_hw):
        t = Transcriber(telemetry_sink=sink)
        partial_1 = TranscriptionResult(
            text="hi",
            language="en",
            confidence=0.9,
            engine_name="faster-whisper",
            segments=(),
            speaker_segments=(),
            audio_duration_ms=500,
            duration_ms=20,
        )
        partial_2 = TranscriptionResult(
            text="hi there",
            language="en",
            confidence=0.92,
            engine_name="faster-whisper",
            segments=(),
            speaker_segments=(),
            audio_duration_ms=1000,
            duration_ms=40,
        )

        async def _fake_engine_stream(*args, **kwargs):
            yield partial_1
            yield partial_2

        async def _chunks():
            yield b"\x00" * 1000

        with patch.object(t._engines["en"], "transcribe_streaming", new=_fake_engine_stream):
            received = [
                r async for r in t.transcribe_streaming(_chunks(), language="en", tenant_id="alice")
            ]

    assert len(received) == 2
    assert received[0].text == "hi"
    assert received[1].text == "hi there"
    assert len(sink.records) == 2
    assert sink.records[0].tenant_id == "alice"
    assert sink.records[1].tenant_id == "alice"
    assert sink.records[0].language_requested == "en"
    assert sink.records[0].language_detected == "en"  # engine truth


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_lid_confidence_at_threshold_routes_to_detected(cpu_only_hw):
    """Inclusive >= boundary: exactly the threshold uses the detected language."""
    sink = ListTelemetrySink()
    with patch("lattice_asr.transcriber.detect_hardware", return_value=cpu_only_hw):
        t = Transcriber(telemetry_sink=sink)
        threshold = t._config.lid.confidence_threshold  # default 0.85
        fake_result = TranscriptionResult(
            text="hola",
            language="es",
            confidence=0.9,
            engine_name="faster-whisper",
            segments=(),
            speaker_segments=(),
            audio_duration_ms=1000,
            duration_ms=50,
        )
        lid_result = LidResult(language="es", confidence=threshold)
        with patch.object(t._lid, "detect", new=AsyncMock(return_value=lid_result)):
            with patch.object(
                t._engines["multi"], "transcribe", new=AsyncMock(return_value=fake_result)
            ):
                r = await t.transcribe(b"\x00\x00")
    assert r.text == "hola"
    assert sink.records[0].language_detected == "es"


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_transcribe_empty_audio_passes_through(cpu_only_hw):
    """Empty PCM bytes propagate to engine; engine result + telemetry record normally."""
    sink = ListTelemetrySink()
    with patch("lattice_asr.transcriber.detect_hardware", return_value=cpu_only_hw):
        t = Transcriber(telemetry_sink=sink)
        fake_result = TranscriptionResult(
            text="",
            language="en",
            confidence=0.0,
            engine_name="faster-whisper",
            segments=(),
            speaker_segments=(),
            audio_duration_ms=0,
            duration_ms=10,
        )
        with patch.object(t._engines["en"], "transcribe", new=AsyncMock(return_value=fake_result)):
            r = await t.transcribe(b"", language="en")
    assert r.text == ""
    assert r.audio_duration_ms == 0
    assert len(sink.records) == 1


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_transcribe_unknown_language_routes_to_multi(cpu_only_hw):
    """Unknown language code (e.g., 'zz') is not validated; routes via the en-vs-multi binary to multi."""
    sink = ListTelemetrySink()
    with patch("lattice_asr.transcriber.detect_hardware", return_value=cpu_only_hw):
        t = Transcriber(telemetry_sink=sink)
        fake_result = TranscriptionResult(
            text="??",
            language="zz",
            confidence=0.1,
            engine_name="faster-whisper",
            segments=(),
            speaker_segments=(),
            audio_duration_ms=1000,
            duration_ms=50,
        )
        with patch.object(
            t._engines["multi"], "transcribe", new=AsyncMock(return_value=fake_result)
        ):
            r = await t.transcribe(b"\x00\x00", language="zz")
    assert r.language == "zz"
    assert sink.records[0].language_requested == "zz"


@pytest.mark.r_tier
@pytest.mark.asyncio
async def test_transcribe_engine_raises_propagates_and_skips_telemetry(cpu_only_hw):
    """If the engine raises, Transcriber re-raises and telemetry is NOT recorded (fail-fast)."""
    sink = ListTelemetrySink()
    with patch("lattice_asr.transcriber.detect_hardware", return_value=cpu_only_hw):
        t = Transcriber(telemetry_sink=sink)
        with patch.object(t._engines["en"], "transcribe", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                await t.transcribe(b"\x00\x00", language="en")
    assert len(sink.records) == 0
