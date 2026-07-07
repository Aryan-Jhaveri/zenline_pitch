"""Step 2 ontology tests: apparel attribute extraction cases."""

from __future__ import annotations

import json

import polars as pl
import pytest

from substitutes_agent.step2_ontology import (
    build_ontology,
    colour_family,
    colour_similarity,
    extract_material,
    extract_pattern,
)

# ---------------------------------------------------------------------------
# colour_family / colour_similarity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "family"),
    [
        ("Blue", "blue"),
        ("Navy Blue", "blue"),
        ("Navy", "blue"),
        ("Teal", "blue"),
        ("Turquoise Blue", "blue"),
        ("Black", "black"),
        ("Charcoal", "black"),
        ("Grey Melange", "black"),
        ("White", "white"),
        ("Off White", "white"),
        ("Cream", "white"),
        ("Maroon", "red"),
        ("Burgundy", "red"),
        ("Orange", "red"),
        ("Multi", "multi"),
        ("", "other"),
        ("NA", "other"),
        ("Mystery", "other"),
    ],
)
def test_colour_family(raw: str, family: str) -> None:
    assert colour_family(raw) == family


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("Blue", "Blue", 1.0),  # exact
        ("Navy Blue", "Blue", 0.6),  # same family, different exact
        ("Navy", "Teal", 0.6),  # same family
        ("Blue", "Black", 0.0),  # different family
        ("Blue", "Red", 0.0),
        ("Blue", "", 0.5),  # one side unknown
        ("", "", 0.5),  # both unknown
        ("Multi", "Multi", 1.0),
        ("Multi", "Blue", 0.0),
    ],
)
def test_colour_similarity(a: str, b: str, expected: float) -> None:
    assert colour_similarity(a, b) == expected


# ---------------------------------------------------------------------------
# pattern
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Solid Cotton T-shirt", "Solid"),
        ("Striped Casual Shirt", "Striped"),
        ("Stripes Top", "Striped"),
        ("Printed Kurta", "Printed"),
        ("Checked Casual Shirt", "Checked"),
        ("Gingham Shirt", "Checked"),
        ("Floral Dress", "Floral"),
        ("Flower Print Top", "Floral"),
        ("Graphic Tee", "Graphic"),
        ("Polka Dot Top", "Polka"),
        ("Tie-dye Top", "Tie-dye"),
        ("Tie dye Top", "Tie-dye"),
        ("Colorblock Jacket", "Colorblock"),
        ("Colour-block Jacket", "Colorblock"),
        ("Plain Cotton Tee", None),
        ("", None),
    ],
)
def test_extract_pattern(name: str, expected: str | None) -> None:
    assert extract_pattern(name) == expected


def test_extract_pattern_first_match_wins() -> None:
    # "Solid Striped" -> Solid wins (first rule).
    assert extract_pattern("Solid Striped Shirt") == "Solid"


# ---------------------------------------------------------------------------
# material
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Cotton T-shirt", "Cotton"),
        ("Denim Jacket", "Denim"),
        ("Linen Shirt", "Linen"),
        ("Silk Saree", "Silk"),
        ("Polyester Track Pant", "Polyester"),
        ("Leather Boots", "Leather"),
        ("Wool Sweater", "Wool"),
        ("Woollen Cardigan", "Wool"),
        ("Cashmere Sweater", "Cashmere"),
        ("Knit Top", "Knit"),
        ("Knitted Sweater", "Knit"),
        ("Fleece Jacket", "Fleece"),
        ("Mesh Running Top", "Mesh"),
        ("Plain Tee", None),
        ("", None),
    ],
)
def test_extract_material(name: str, expected: str | None) -> None:
    assert extract_material(name) == expected


# ---------------------------------------------------------------------------
# build_ontology end-to-end
# ---------------------------------------------------------------------------


def _make_step1_df(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "id": pl.Int64,
            "brand_normalized": pl.String,
            "productDisplayName": pl.String,
            "gender": pl.String,
            "masterCategory": pl.String,
            "subCategory": pl.String,
            "articleType": pl.String,
            "baseColour": pl.String,
            "season": pl.String,
            "year": pl.Int64,
            "usage": pl.String,
        },
    )


def test_build_ontology_extracts_all_fields(tmp_path) -> None:  # type: ignore[no-untyped-def]
    rows = [
        {
            "id": 1,
            "brand_normalized": "peter england",
            "productDisplayName": "Peter England Men Striped Cotton Blue Jeans",
            "gender": "Men",
            "masterCategory": "Apparel",
            "subCategory": "Bottomwear",
            "articleType": "Jeans",
            "baseColour": "Navy Blue",
            "season": "Summer",
            "year": 2019,
            "usage": "Casual",
        },
        {
            "id": 2,
            "brand_normalized": "reebok",
            "productDisplayName": "Reebok Women Leather Black Track Pant",
            "gender": "Women",
            "masterCategory": "Apparel",
            "subCategory": "Bottomwear",
            "articleType": "Track Pants",
            "baseColour": "Black",
            "season": "Winter",
            "year": 2019,
            "usage": "Sports",
        },
    ]
    p = tmp_path / "s1.parquet"
    _make_step1_df(rows).write_parquet(p)
    out = tmp_path / "s2.json"
    records, log = build_ontology(p, out)
    assert log.input_rows == 2
    assert len(records) == 2
    by_id = {r.code: r for r in records}
    r1 = by_id["1"]
    assert r1.article_type == "jeans"  # aligned via Vastra taxonomy
    assert r1.master_category == "Apparel"
    assert r1.base_colour == "blue"  # Navy Blue -> blue family
    assert r1.usage == "Casual"
    assert r1.gender == "Men"
    assert r1.season == "Summer"
    assert r1.pattern == "Striped"
    assert r1.material == "Cotton"
    assert r1.confidence == 0.7  # regex-derived fields present
    assert r1.source == "rules"
    r2 = by_id["2"]
    assert r2.base_colour == "black"
    assert r2.pattern is None
    assert r2.material == "Leather"
    assert r2.confidence == 0.7


def test_build_ontology_ambiguous_long_name_flagged(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # Long name, no pattern or material keyword -> ambiguous, confidence 0.6.
    rows = [
        {
            "id": 1,
            "brand_normalized": "nike",
            "productDisplayName": "Nike Mean Team India Cricket Jersey",
            "gender": "Men",
            "masterCategory": "Apparel",
            "subCategory": "Topwear",
            "articleType": "Tshirts",
            "baseColour": "Blue",
            "season": "Summer",
            "year": 2019,
            "usage": "Sports",
        },
    ]
    p = tmp_path / "s1.parquet"
    _make_step1_df(rows).write_parquet(p)
    records, log = build_ontology(p)
    assert log.filters["ambiguous_low_confidence_rows"] == 1
    assert records[0].confidence == 0.6
    assert records[0].pattern is None
    assert records[0].material is None


def test_build_ontology_short_name_no_ambiguous_flag(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # Short name with no pattern/material -> not flagged ambiguous (len <= 20).
    rows = [
        {
            "id": 1,
            "brand_normalized": "ferrari",
            "productDisplayName": "Ferrari Tee",
            "gender": "Men",
            "masterCategory": "Apparel",
            "subCategory": "Topwear",
            "articleType": "Tshirts",
            "baseColour": "Red",
            "season": "Summer",
            "year": 2019,
            "usage": "Casual",
        },
    ]
    p = tmp_path / "s1.parquet"
    _make_step1_df(rows).write_parquet(p)
    records, log = build_ontology(p)
    assert log.filters["ambiguous_low_confidence_rows"] == 0
    # Pure column-derived -> confidence 1.0.
    assert records[0].confidence == 1.0


def test_build_ontology_writes_valid_sorted_json(tmp_path) -> None:  # type: ignore[no-untyped-def]
    rows = [
        {
            "id": 99,
            "brand_normalized": "b",
            "productDisplayName": "Tee",
            "gender": "Men",
            "masterCategory": "Apparel",
            "subCategory": "Topwear",
            "articleType": "Tshirts",
            "baseColour": "Blue",
            "season": "Summer",
            "year": 2019,
            "usage": "Casual",
        },
        {
            "id": 10,
            "brand_normalized": "a",
            "productDisplayName": "Tee",
            "gender": "Men",
            "masterCategory": "Apparel",
            "subCategory": "Topwear",
            "articleType": "Tshirts",
            "baseColour": "Blue",
            "season": "Summer",
            "year": 2019,
            "usage": "Casual",
        },
    ]
    p = tmp_path / "s1.parquet"
    _make_step1_df(rows).write_parquet(p)
    out = tmp_path / "s2.json"
    records, _ = build_ontology(p, out)
    assert [r.code for r in records] == ["10", "99"]
    assert out.exists()
    data = json.loads(out.read_text())
    assert isinstance(data, list)
    assert [d["code"] for d in data] == ["10", "99"]
    assert data[0]["article_type"] == "t-shirt"  # aligned


def test_build_ontology_null_season(tmp_path) -> None:  # type: ignore[no-untyped-def]
    rows = [
        {
            "id": 1,
            "brand_normalized": "b",
            "productDisplayName": "Cotton Tee",
            "gender": "Men",
            "masterCategory": "Apparel",
            "subCategory": "Topwear",
            "articleType": "Tshirts",
            "baseColour": "Blue",
            "season": None,
            "year": 2019,
            "usage": "Casual",
        },
    ]
    p = tmp_path / "s1.parquet"
    _make_step1_df(rows).write_parquet(p)
    records, _ = build_ontology(p)
    assert records[0].season is None
    assert records[0].material == "Cotton"
