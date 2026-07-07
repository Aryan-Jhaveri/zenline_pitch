"""Step 3 classify tests: variant/substitute/unrelated, boundary, determinism."""

from __future__ import annotations

import json

from substitutes_agent.models import OntologyRecord
from substitutes_agent.step2_ontology import colour_similarity
from substitutes_agent.step3_classify import (
    TIE_BREAK_HIGH,
    TIE_BREAK_LOW,
    classify,
    classify_pair,
    score_pair,
)


def _rec(
    code: str,
    brand: str = "brand",
    article: str = "t-shirt",
    master: str = "Apparel",
    colour: str = "blue",
    usage: str = "Casual",
    gender: str = "Men",
    season: str | None = "Summer",
    pattern: str | None = None,
    material: str | None = None,
) -> OntologyRecord:
    return OntologyRecord(
        code=code,
        brand_normalized=brand,
        product_name="x",
        article_type=article,
        master_category=master,
        base_colour=colour,
        usage=usage,
        gender=gender,
        season=season,
        pattern=pattern,
        material=material,
    )


# ---------------------------------------------------------------------------
# classify_pair table-driven cases
# ---------------------------------------------------------------------------


def test_classify_variant() -> None:
    a = _rec("1", brand="nike", colour="blue", usage="Casual")
    b = _rec("2", brand="nike", colour="blue", usage="Casual")
    comp = score_pair(a, b)
    assert classify_pair(a, b, comp) == "VARIANT"


def test_classify_variant_different_season_and_pattern_ok() -> None:
    # VARIANT only requires same brand/type/colour/usage; differing season
    # or missing pattern/material is fine.
    a = _rec(
        "1",
        brand="nike",
        colour="blue",
        usage="Casual",
        season="Summer",
        pattern="Solid",
    )
    b = _rec(
        "2", brand="nike", colour="blue", usage="Casual", season="Winter", pattern=None
    )
    comp = score_pair(a, b)
    assert classify_pair(a, b, comp) == "VARIANT"


def test_classify_substitute_different_brand_same_colour_family() -> None:
    a = _rec("1", brand="nike", colour="Blue", usage="Casual")
    b = _rec("2", brand="adidas", colour="Navy Blue", usage="Casual")
    comp = score_pair(a, b)
    # colour_similarity(Blue, Navy Blue) == 0.6 >= 0.5 -> SUBSTITUTE
    assert comp.colour_similarity == 0.6
    assert classify_pair(a, b, comp) == "SUBSTITUTE"


def test_classify_unrelated_different_usage() -> None:
    a = _rec("1", brand="nike", colour="blue", usage="Casual")
    b = _rec("2", brand="adidas", colour="blue", usage="Sports")
    comp = score_pair(a, b)
    # Different usage -> not VARIANT, not SUBSTITUTE -> UNRELATED
    assert classify_pair(a, b, comp) is None


def test_classify_unrelated_same_brand_different_colour() -> None:
    a = _rec("1", brand="nike", colour="blue", usage="Casual")
    b = _rec("2", brand="nike", colour="red", usage="Casual")
    comp = score_pair(a, b)
    # Same brand but different colour -> not VARIANT; same brand -> not SUBSTITUTE.
    assert classify_pair(a, b, comp) is None


def test_classify_unrelated_different_colour_family() -> None:
    a = _rec("1", brand="nike", colour="blue", usage="Casual")
    b = _rec("2", brand="adidas", colour="red", usage="Casual")
    comp = score_pair(a, b)
    # colour_similarity(blue, red) == 0.0 < 0.5 -> not SUBSTITUTE -> UNRELATED
    assert comp.colour_similarity == 0.0
    assert classify_pair(a, b, comp) is None


# ---------------------------------------------------------------------------
# colour_similarity edge cases (navy = blue?)
# ---------------------------------------------------------------------------


def test_navy_and_blue_same_family() -> None:
    assert colour_similarity("Navy Blue", "Blue") == 0.6


def test_teal_and_navy_same_family() -> None:
    assert colour_similarity("Teal", "Navy") == 0.6


def test_blue_and_black_different_family() -> None:
    assert colour_similarity("Blue", "Black") == 0.0


# ---------------------------------------------------------------------------
# Tie-breaker boundary
# ---------------------------------------------------------------------------


def test_tie_break_band_includes_endpoints() -> None:
    assert TIE_BREAK_LOW <= 0.45 <= TIE_BREAK_HIGH
    assert TIE_BREAK_LOW <= 0.55 <= TIE_BREAK_HIGH
    assert not (TIE_BREAK_LOW <= 0.4499 <= TIE_BREAK_HIGH)
    assert not (TIE_BREAK_LOW <= 0.5501 <= TIE_BREAK_HIGH)


def test_emitted_substitute_score_above_band() -> None:
    # A SUBSTITUTE edge (same article, same usage, colour_similarity >= 0.5)
    # scores at least 0.35 + 0.10 + 0.15 = 0.60, so it never lands in the
    # [0.45, 0.55] tie-break band. Documented behaviour.
    a = _rec(
        "1",
        brand="nike",
        colour="Blue",
        usage="Casual",
        pattern=None,
        material=None,
        season=None,
    )
    b = _rec(
        "2",
        brand="adidas",
        colour="Blue",
        usage="Casual",
        pattern=None,
        material=None,
        season=None,
    )
    comp = score_pair(a, b)
    assert comp.total_score >= 0.60
    assert not (TIE_BREAK_LOW <= comp.total_score <= TIE_BREAK_HIGH)


# ---------------------------------------------------------------------------
# Determinism: byte-identical output across two runs
# ---------------------------------------------------------------------------


def _write_ontology(tmp_path, records: list[OntologyRecord]) -> str:  # type: ignore[no-untyped-def]
    p = tmp_path / "s2.json"
    p.write_text(
        json.dumps([r.model_dump() for r in records], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return str(p)


def test_classify_determinism_byte_identical(tmp_path) -> None:  # type: ignore[no-untyped-def]
    records = [
        _rec("1", brand="nike", colour="blue", usage="Casual"),
        _rec("2", brand="nike", colour="blue", usage="Casual"),
        _rec("3", brand="adidas", colour="blue", usage="Casual"),
        _rec("4", brand="puma", colour="navy", usage="Casual"),
    ]
    ont = _write_ontology(tmp_path, records)
    out1 = tmp_path / "s3_1.json"
    out2 = tmp_path / "s3_2.json"
    classify(ont, out1)
    classify(ont, out2)
    assert out1.read_bytes() == out2.read_bytes()


def test_classify_sorts_pairs_by_sku(tmp_path) -> None:  # type: ignore[no-untyped-def]
    records = [
        _rec("9", brand="nike", colour="blue", usage="Casual"),
        _rec("1", brand="nike", colour="blue", usage="Casual"),
        _rec("5", brand="adidas", colour="blue", usage="Casual"),
    ]
    ont = _write_ontology(tmp_path, records)
    out = tmp_path / "s3.json"
    rels, log = classify(ont, out)
    pairs = [(r.sku_a, r.sku_b) for r in rels]
    assert pairs == sorted(pairs)
    # 1 and 9 same brand/colour/usage -> VARIANT; 1/9 vs 5 -> SUBSTITUTE.
    assert log.filters["variant_edges"] >= 1
    assert log.filters["substitute_edges"] >= 1


def test_classify_skips_empty_article_type(tmp_path) -> None:  # type: ignore[no-untyped-def]
    records = [
        _rec("1", brand="nike", article="", colour="blue", usage="Casual"),
        _rec("2", brand="nike", colour="blue", usage="Casual"),
    ]
    ont = _write_ontology(tmp_path, records)
    _, log = classify(ont)
    assert log.filters["skipped_no_article_type"] == 1


def test_classify_writes_valid_json(tmp_path) -> None:  # type: ignore[no-untyped-def]
    records = [
        _rec("1", brand="nike", colour="blue", usage="Casual"),
        _rec("2", brand="adidas", colour="blue", usage="Casual"),
    ]
    ont = _write_ontology(tmp_path, records)
    out = tmp_path / "s3.json"
    rels, _ = classify(ont, out)
    data = json.loads(out.read_text())
    assert isinstance(data, list)
    assert len(data) == len(rels)
    assert data[0]["sku_a"] < data[0]["sku_b"]
    assert "components" in data[0]
    assert "total_score" in data[0]["components"]
