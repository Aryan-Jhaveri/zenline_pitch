"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
INGEST_FIXTURE_CSV = FIXTURES / "ingest_fixture.csv"


@pytest.fixture(scope="session")
def ingest_fixture_csv() -> Path:
    """Path to the committed hand-crafted apparel ingest fixture."""
    return INGEST_FIXTURE_CSV
