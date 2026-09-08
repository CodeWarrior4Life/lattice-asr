"""ParakeetMlxEngine must do ALL MLX work on one dedicated thread.

Regression for the 2026-09-08 lattice-dictate defect: MLX streams are
thread-local, `from_pretrained` leaves weight casts lazy on the loader's
stream, and `asyncio.to_thread` may run the first transcribe on a different
pool worker -> `RuntimeError('There is no Stream(cpu, 0) in current thread.')`
on the first dictation after every daemon start. No model or MLX needed here:
the fakes record which thread each call ran on.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import types

import pytest

from lattice_asr.engines.parakeet_mlx import ParakeetMlxEngine


class _FakeResult:
    text = "hello"
    sentences = None


class _FakeModel:
    def __init__(self, seen: dict):
        self._seen = seen
        self.evaluated = False

    def transcribe(self, path: str) -> _FakeResult:
        self._seen.setdefault("transcribe", []).append(threading.get_ident())
        return _FakeResult()

    def parameters(self) -> dict:
        return {"w": "lazy-weight"}


def _install_fake_parakeet(monkeypatch, seen: dict) -> None:
    mod = types.ModuleType("parakeet_mlx")

    def from_pretrained(name: str):
        seen["load"] = threading.get_ident()
        return _FakeModel(seen)

    mod.from_pretrained = from_pretrained  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "parakeet_mlx", mod)


def _install_fake_mlx(monkeypatch, seen: dict) -> None:
    core = types.ModuleType("mlx.core")

    def _eval(arrays):
        seen["eval"] = threading.get_ident()
        seen["eval_args"] = list(arrays)

    core.eval = _eval  # type: ignore[attr-defined]
    utils = types.ModuleType("mlx.utils")
    utils.tree_flatten = lambda d: list(d.items())  # type: ignore[attr-defined]
    pkg = types.ModuleType("mlx")
    pkg.core = core  # type: ignore[attr-defined]
    pkg.utils = utils  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx", pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(sys.modules, "mlx.utils", utils)


def _pcm_2s() -> bytes:
    return b"\x00" * 32000


@pytest.mark.asyncio
async def test_load_and_every_transcribe_share_one_thread(monkeypatch):
    seen: dict = {}
    _install_fake_parakeet(monkeypatch, seen)
    _install_fake_mlx(monkeypatch, seen)
    eng = ParakeetMlxEngine()

    await eng.warmup()
    # Saturate the default pool with unrelated work so any to_thread-based
    # engine would be handed a DIFFERENT worker for the next call.
    await asyncio.gather(*(asyncio.to_thread(lambda: None) for _ in range(8)))
    for _ in range(3):
        r = await eng.transcribe(_pcm_2s(), 16000, "en")
        assert r.text == "hello"

    assert seen["load"] != threading.get_ident(), "MLX work must not run on the event loop thread"
    assert set(seen["transcribe"]) == {seen["load"]}, (
        "transcribe ran on a different thread than the load: "
        f"load={seen['load']} transcribe={seen['transcribe']}"
    )


@pytest.mark.asyncio
async def test_weights_are_materialized_on_the_loading_thread(monkeypatch):
    seen: dict = {}
    _install_fake_parakeet(monkeypatch, seen)
    _install_fake_mlx(monkeypatch, seen)
    eng = ParakeetMlxEngine()

    await eng.warmup()

    assert seen.get("eval") == seen["load"], "mx.eval must run on the thread that loaded"
    assert seen["eval_args"] == ["lazy-weight"], "every parameter must be forced concrete"


@pytest.mark.asyncio
async def test_model_without_parameters_is_tolerated(monkeypatch):
    """Test fakes and exotic models without `parameters()` must not break warmup."""
    mod = types.ModuleType("parakeet_mlx")
    mod.from_pretrained = lambda name: types.SimpleNamespace(  # type: ignore[attr-defined]
        transcribe=lambda p: _FakeResult()
    )
    monkeypatch.setitem(sys.modules, "parakeet_mlx", mod)
    eng = ParakeetMlxEngine()
    await eng.warmup()
    r = await eng.transcribe(_pcm_2s(), 16000, "en")
    assert r.text == "hello"


@pytest.mark.asyncio
async def test_two_engines_do_not_share_a_thread(monkeypatch):
    """Each engine owns its thread; two models must not cross streams either."""
    seen: dict = {}
    _install_fake_parakeet(monkeypatch, seen)
    _install_fake_mlx(monkeypatch, seen)
    a, b = ParakeetMlxEngine(), ParakeetMlxEngine()
    await a.warmup()
    load_a = seen["load"]
    await b.warmup()
    load_b = seen["load"]
    assert load_a != load_b
