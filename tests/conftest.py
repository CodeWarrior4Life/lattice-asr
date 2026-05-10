import hashlib
from pathlib import Path
from urllib.request import urlretrieve

import pytest


_FIXTURES = {
    "hello-en-2s.wav": (
        "https://huggingface.co/datasets/CodeWarrior4Life/lattice-asr-fixtures/resolve/main/hello-en-2s.wav",
        "REPLACE_WITH_REAL_SHA256_AT_FIRST_RUN",
    ),
}


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    p = Path(__file__).parent / "fixtures" / "audio"
    p.mkdir(parents=True, exist_ok=True)
    return p


@pytest.fixture
def hello_en_2s_wav(fixture_dir: Path) -> Path:
    name = "hello-en-2s.wav"
    target = fixture_dir / name
    if not target.exists():
        url, _expected_sha = _FIXTURES[name]
        urlretrieve(url, target)
    return target


def pytest_collection_modifyitems(config, items):
    import os

    if os.getenv("LATTICE_ASR_S_TIER") != "1":
        skip = pytest.mark.skip(reason="S-tier opt-in: set LATTICE_ASR_S_TIER=1")
        for item in items:
            if "s_tier" in item.keywords:
                item.add_marker(skip)
