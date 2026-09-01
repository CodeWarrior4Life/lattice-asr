"""LID result-type tests. Silero is deleted -- see lattice_asr.lid."""

import pytest

from lattice_asr.lid import UNDETERMINED, LidResult


@pytest.mark.r_tier
def test_lid_result_frozen():
    r = LidResult(language="en", confidence=0.95)
    with pytest.raises((AttributeError, TypeError)):
        r.language = "es"  # type: ignore[misc]


@pytest.mark.r_tier
def test_undetermined_is_not_determined():
    assert not LidResult(language=UNDETERMINED, confidence=0.0).is_determined
    assert LidResult(language="es", confidence=0.9367).is_determined


@pytest.mark.r_tier
def test_silero_is_gone():
    """Regression guard for the 2026-05-27 outage.

    `SileroLid` called a `torch.hub` callable upstream had removed, and the
    resulting exception was swallowed into a silent English default. If anyone
    reintroduces the symbol, they get this test instead of another quiet
    three-month outage.
    """
    import lattice_asr.lid as lid_mod

    assert not hasattr(lid_mod, "SileroLid")

    # AST, not a text grep: the module docstring legitimately *discusses*
    # torch.hub as the thing that broke. What must be absent is executable
    # reference to it.
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(lid_mod))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    imported = {
        alias.name.split(".")[0]
        for n in ast.walk(tree)
        if isinstance(n, ast.Import)
        for alias in n.names
    }
    imported |= {
        n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
    }
    assert "torch" not in names | imported
    assert "hub" not in names
