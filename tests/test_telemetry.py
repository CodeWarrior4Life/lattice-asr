from datetime import datetime, timezone
import pytest
from lattice_asr.telemetry import NullTelemetrySink, ListTelemetrySink
from lattice_asr import AsrCallRecord


def _record(engine="test"):
    return AsrCallRecord(
        engine_name=engine,
        language_detected="en",
        language_requested=None,
        audio_duration_ms=1000,
        transcription_duration_ms=50,
        diarized=False,
        speaker_count=None,
        tenant_id=None,
        timestamp_utc=datetime.now(timezone.utc),
    )


@pytest.mark.r_tier
def test_null_sink_swallows_records():
    sink = NullTelemetrySink()
    sink.record(_record())  # no-op, must not raise


@pytest.mark.r_tier
def test_list_sink_appends():
    sink = ListTelemetrySink()
    sink.record(_record("a"))
    sink.record(_record("b"))
    assert [r.engine_name for r in sink.records] == ["a", "b"]


@pytest.mark.r_tier
def test_list_sink_clear():
    sink = ListTelemetrySink()
    sink.record(_record())
    sink.clear()
    assert sink.records == []


@pytest.mark.r_tier
def test_call_record_is_hashable():
    rec = _record()
    s = {rec}
    assert rec in s


@pytest.mark.r_tier
def test_null_sink_implements_protocol():
    from lattice_asr import TelemetrySink

    sink: TelemetrySink = NullTelemetrySink()
    assert hasattr(sink, "record")
