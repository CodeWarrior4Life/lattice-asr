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
