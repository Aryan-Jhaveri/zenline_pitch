"""Step 2 ontology tests: table-driven attribute extraction cases."""

from __future__ import annotations

import polars as pl
import pytest

from substitutes_agent.step2_ontology import (
    build_ontology,
    extract_format,
    extract_key_actives,
    extract_pack_size_ml,
    extract_skin_type_claims,
    extract_target_area,
)

# ---------------------------------------------------------------------------
# format
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Hydrating Face Cream", "cream"),
        ("Crème Hydratante", "cream"),
        ("Vitamin C Serum", "serum"),
        ("Clearing Gel", "gel"),
        ("Body Lotion", "lotion"),
        ("Repair Balm", "balm"),
        ("Facial Oil", "oil"),
        ("Sheet Mask", "mask"),
        ("Oil-Free Moisturizer", "cream"),  # priority: cream wins over oil
        ("Moisturizer", "cream"),  # moisturizer is a cream synonym
        ("", "unknown"),
    ],
)
def test_extract_format(name: str, expected: str) -> None:
    assert extract_format(name) == expected


# ---------------------------------------------------------------------------
# skin_type_claims
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "labels", "expected"),
    [
        ("Dry Skin Cream", "", "dry"),
        ("For Sensitive Skin", "", "sensitive"),
        ("Oily & Combination Gel", "", "oily,combination"),
        ("Mature Skin Cream", "en:hypoallergenic", "mature"),
        ("Normal Moisturizer", "", "normal"),
        ("Cream", "en:for-dry-skin", "dry"),
        ("Plain Cream", "", ""),
    ],
)
def test_extract_skin_type_claims(name: str, labels: str, expected: str) -> None:
    assert extract_skin_type_claims(name, labels) == expected


# ---------------------------------------------------------------------------
# key_actives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "tags", "expected"),
    [
        ("water, niacinamide, glycerin", [], ["niacinamide"]),
        ("retinol and water", ["en:retinol"], ["retinol"]),
        ("hyaluronic acid, sodium hyaluronate", [], ["hyaluronic acid"]),
        ("ascorbic acid", [], ["vitamin c"]),
        ("vitamin c ester", [], ["vitamin c"]),
        ("salicylic acid 2%", ["en:salicylic-acid"], ["salicylic acid"]),
        ("glycolic acid + lactic acid", [], ["glycolic acid", "lactic acid"]),
        ("ceramide complex", ["en:ceramides"], ["ceramides"]),
        ("peptide blend", [], ["peptides"]),
        ("octinoxate, avobenzone, zinc oxide", [], ["spf"]),
        ("water, glycerin", [], []),
        ("", [], []),
    ],
)
def test_extract_key_actives(text: str, tags: list[str], expected: list[str]) -> None:
    assert extract_key_actives(text, tags) == expected


def test_key_actives_dedupes_overlapping_patterns() -> None:
    # "ascorbic acid" and "vitamin c" both map to the same label.
    result = extract_key_actives("vitamin c and ascorbic acid", [])
    assert result == ["vitamin c"]


# ---------------------------------------------------------------------------
# pack_size_ml
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("quantity", "expected"),
    [
        ("50 ml", 50.0),
        ("250 g", 250.0),
        ("1 oz", 29.57),
        ("2 oz", 59.15),
        ("1.5 fl oz", 44.36),
        ("150ml", 150.0),
        ("30 ML", 30.0),
        ("", None),
        ("package of 2", None),
        ("50 ml / 1.69 fl oz", 50.0),  # first match wins
    ],
)
def test_extract_pack_size_ml(quantity: str, expected: float | None) -> None:
    result = extract_pack_size_ml(quantity)
    if expected is None:
        assert result is None
    else:
        assert result is not None and abs(result - expected) < 0.01


# ---------------------------------------------------------------------------
# target_area
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Eye Contour Cream", "eye"),
        ("Lip Balm", "lip"),
        ("Body Lotion", "body"),
        ("Face Cream", "face"),
        ("Facial Serum", "face"),
        ("Moisturizer", "unknown"),
        ("", "unknown"),
    ],
)
def test_extract_target_area(name: str, expected: str) -> None:
    assert extract_target_area(name) == expected


# ---------------------------------------------------------------------------
# Multilingual / list-vs-string handling via build_ontology.
# ---------------------------------------------------------------------------


def _make_step1_df(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "code": pl.String,
            "brand_normalized": pl.String,
            "product_name": pl.String,
            "ingredients_text": pl.List(
                pl.Struct({"lang": pl.String, "text": pl.String})
            ),
            "ingredients_tags": pl.List(pl.String),
            "quantity": pl.String,
            "labels": pl.String,
        },
    )


def test_build_ontology_handles_list_and_string_ingredients(tmp_path) -> None:  # type: ignore[no-untyped-def]
    rows = [
        {
            "code": "A",
            "brand_normalized": "brand",
            "product_name": "Face Cream",
            "ingredients_text": [{"lang": "en", "text": "niacinamide, water"}],
            "ingredients_tags": [],
            "quantity": "50 ml",
            "labels": "",
        },
        {
            "code": "B",
            "brand_normalized": "brand",
            "product_name": "Eye Serum",
            "ingredients_text": [{"lang": "fr", "text": "acide hyaluronique"}],
            "ingredients_tags": ["en:hyaluronic-acid"],
            "quantity": "1 oz",
            "labels": "",
        },
    ]
    df = _make_step1_df(rows)
    p = tmp_path / "s1.parquet"
    df.write_parquet(p)
    out = tmp_path / "s2.json"
    records, log = build_ontology(p, out)
    assert log.input_rows == 2
    assert len(records) == 2
    by_code = {r.code: r for r in records}
    assert by_code["A"].format == "cream"
    assert by_code["A"].key_actives == ["niacinamide"]
    assert by_code["A"].pack_size_ml == 50.0
    assert by_code["B"].format == "serum"
    assert by_code["B"].target_area == "eye"
    assert by_code["B"].key_actives == ["hyaluronic acid"]
    assert by_code["B"].pack_size_ml == 29.57


def test_build_ontology_writes_valid_json(tmp_path) -> None:  # type: ignore[no-untyped-def]
    rows = [
        {
            "code": "1",
            "brand_normalized": "x",
            "product_name": "Face Cream",
            "ingredients_text": [{"lang": "en", "text": "retinol"}],
            "ingredients_tags": [],
            "quantity": "30 ml",
            "labels": "",
        },
    ]
    p = tmp_path / "s1.parquet"
    _make_step1_df(rows).write_parquet(p)
    out = tmp_path / "s2.json"
    records, _ = build_ontology(p, out)
    assert out.exists()
    import json

    data = json.loads(out.read_text())
    assert isinstance(data, list)
    assert data[0]["code"] == "1"
    assert data[0]["format"] == "cream"
    assert data[0]["key_actives"] == ["retinol"]
    # Records match the JSON dump.
    assert len(records) == len(data)


def test_build_ontology_sorts_by_code(tmp_path) -> None:  # type: ignore[no-untyped-def]
    rows = [
        {
            "code": "ZZZ",
            "brand_normalized": "x",
            "product_name": "Face Cream",
            "ingredients_text": [{"lang": "en", "text": ""}],
            "ingredients_tags": [],
            "quantity": "",
            "labels": "",
        },
        {
            "code": "AAA",
            "brand_normalized": "x",
            "product_name": "Face Cream",
            "ingredients_text": [{"lang": "en", "text": ""}],
            "ingredients_tags": [],
            "quantity": "",
            "labels": "",
        },
    ]
    p = tmp_path / "s1.parquet"
    _make_step1_df(rows).write_parquet(p)
    records, _ = build_ontology(p)
    assert [r.code for r in records] == ["AAA", "ZZZ"]
