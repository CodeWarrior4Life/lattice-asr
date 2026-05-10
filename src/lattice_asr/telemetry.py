"""Telemetry sinks — dependency-injected per spec §9."""

from __future__ import annotations

from dataclasses import dataclass, field

from lattice_asr.types import AsrCallRecord


class NullTelemetrySink:
    """Default sink — swallows all records. For library-only use."""

    def record(self, call: AsrCallRecord) -> None:
        return None


@dataclass
class ListTelemetrySink:
    """In-memory sink — useful for tests + simple consumers."""

    records: list[AsrCallRecord] = field(default_factory=list)

    def record(self, call: AsrCallRecord) -> None:
        self.records.append(call)

    def clear(self) -> None:
        self.records.clear()
