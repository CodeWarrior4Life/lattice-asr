"""Scaffold smoke test — verifies the package imports and exposes __version__.

This is the only test in the v0.1 scaffold. Real R-tier suites land with
the implementation in W1+.
"""

from __future__ import annotations

import pytest


@pytest.mark.r_tier
def test_package_imports() -> None:
    import lattice_asr

    assert hasattr(lattice_asr, "__version__")
    assert isinstance(lattice_asr.__version__, str)
    assert lattice_asr.__version__.startswith("0.")


@pytest.mark.r_tier
def test_version_is_pep440_dev_marker() -> None:
    import lattice_asr

    # v0.1 scaffold uses dev-marker; real release tags drop the suffix.
    assert ".dev" in lattice_asr.__version__ or lattice_asr.__version__ == "0.1.0"
