"""Shared pytest fixtures.

`build_fixture_parquet` constructs the hand-crafted ingest fixture with the
real Open Beauty Facts schema (list[struct{lang,text}] for product_name and
ingredients_text) so Step 1's defensive extraction is exercised truthfully.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURE_PARQUET = FIXTURES / "ingest_fixture.parquet"


def _lang(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"lang": lang, "text": text} for lang, text in pairs]


def build_fixture_parquet() -> Path:
    """Build (or rebuild) the 20-row ingest fixture parquet."""
    rows = [
        # 0: en face cream, kept.
        {
            "code": "1000000000001",
            "product_name": _lang(("en", "Hydrating Face Cream"), ("fr", "Crème")),
            "brands": "L’Oréal",
            "categories": "Face creams",
            "categories_tags": ["en:Face creams", "en:Moisturizers"],
            "ingredients_text": _lang(("en", "water, niacinamide, glycerin")),
            "ingredients_tags": ["en:niacinamide", "en:glycerin"],
            "quantity": "50 ml",
            "labels": "en:organic",
            "countries_tags": ["en:france"],
            "last_modified_t": 1_700_000_000,
        },
        # 1: no-en product_name -> falls back to first (French).
        {
            "code": "1000000000002",
            "product_name": _lang(("fr", "Crème Hydratante")),
            "brands": "CeraVe, Ltd.",
            "categories": "moisturizer",
            "categories_tags": ["en:Moisturizers"],
            "ingredients_text": _lang(("fr", "eau, ceramides")),
            "ingredients_tags": ["en:ceramides"],
            "quantity": "250 g",
            "labels": "",
            "countries_tags": ["en:united-states"],
            "last_modified_t": 1_700_000_001,
        },
        # 2: not a face cream -> filtered out.
        {
            "code": "1000000000003",
            "product_name": _lang(("en", "Shampoo")),
            "brands": "Head & Shoulders",
            "categories": "shampoo",
            "categories_tags": ["en:shampoos"],
            "ingredients_text": _lang(("en", "water, sulfates")),
            "ingredients_tags": [],
            "quantity": "400 ml",
            "labels": "",
            "countries_tags": [],
            "last_modified_t": 1_700_000_002,
        },
        # 3: face cream but empty name -> dropped at required step.
        {
            "code": "1000000000004",
            "product_name": _lang(("en", "")),
            "brands": "SomeBrand",
            "categories": "face cream",
            "categories_tags": ["en:face-creams"],
            "ingredients_text": _lang(("en", "water")),
            "ingredients_tags": [],
            "quantity": "30 ml",
            "labels": "",
            "countries_tags": [],
            "last_modified_t": 1_700_000_003,
        },
        # 4: face cream but empty brand -> dropped at required step.
        {
            "code": "1000000000005",
            "product_name": _lang(("en", "Nice Cream")),
            "brands": "",
            "categories": "day cream",
            "categories_tags": ["en:Day creams"],
            "ingredients_text": _lang(("en", "water, retinol")),
            "ingredients_tags": ["en:retinol"],
            "quantity": "75 ml",
            "labels": "",
            "countries_tags": [],
            "last_modified_t": 1_700_000_004,
        },
        # 5 + 6: duplicate code; newer (6) must win on dedupe.
        {
            "code": "1000000000006",
            "product_name": _lang(("en", "Older Name")),
            "brands": "Olay Inc.",
            "categories": "night cream",
            "categories_tags": ["en:Night creams"],
            "ingredients_text": _lang(("en", "water")),
            "ingredients_tags": [],
            "quantity": "1 oz",
            "labels": "",
            "countries_tags": [],
            "last_modified_t": 1_600_000_000,
        },
        {
            "code": "1000000000006",
            "product_name": _lang(("en", "Newer Name")),
            "brands": "Olay Inc.",
            "categories": "night cream",
            "categories_tags": ["en:Night creams", "en:face-moisturizers"],
            "ingredients_text": _lang(("en", "water, hyaluronic acid")),
            "ingredients_tags": ["en:hyaluronic-acid"],
            "quantity": "2 oz",
            "labels": "",
            "countries_tags": [],
            "last_modified_t": 1_700_000_010,
        },
        # 7: accent + suffix brand normalization edge case.
        {
            "code": "1000000000007",
            "product_name": _lang(("en", "Vitamin C Serum")),
            "brands": "Nüdé S.A.",
            "categories": "face serum",
            "categories_tags": ["en:face-moisturizers"],
            "ingredients_text": _lang(("en", "ascorbic acid, water")),
            "ingredients_tags": ["en:ascorbic-acid"],
            "quantity": "30 ml",
            "labels": "",
            "countries_tags": [],
            "last_modified_t": 1_700_000_007,
        },
        # 8: trailing legal suffix without comma.
        {
            "code": "1000000000008",
            "product_name": _lang(("en", "Gel Cleanser")),
            "brands": "CleanCo LLC",
            "categories": "face care",
            "categories_tags": ["en:Moisturizers"],
            "ingredients_text": _lang(("en", "salicylic acid")),
            "ingredients_tags": ["en:salicylic-acid"],
            "quantity": "150ml",
            "labels": "",
            "countries_tags": [],
            "last_modified_t": 1_700_000_008,
        },
        # 9-13: plain string product_name (defensive string form).
        {
            "code": "1000000000009",
            "product_name": "Simple Moisturizer",
            "brands": "Simple",
            "categories": "",
            "categories_tags": ["en:moisturizers"],
            "ingredients_text": "water, glycerin",
            "ingredients_tags": ["en:glycerin"],
            "quantity": "100 ml",
            "labels": "",
            "countries_tags": [],
            "last_modified_t": 1_700_000_009,
        },
        {
            "code": "1000000000010",
            "product_name": "Day Cream SPF",
            "brands": "Eucerin",
            "categories": "",
            "categories_tags": ["en:day-creams"],
            "ingredients_text": "avobenzone, octinoxate",
            "ingredients_tags": ["en:avobenzone"],
            "quantity": "50ml",
            "labels": "",
            "countries_tags": [],
            "last_modified_t": 1_700_000_011,
        },
        {
            "code": "1000000000011",
            "product_name": "Eye Balm",
            "brands": "Kiehl's",
            "categories": "",
            "categories_tags": ["en:face-creams"],
            "ingredients_text": "ceramides, peptides",
            "ingredients_tags": [],
            "quantity": "15 g",
            "labels": "",
            "countries_tags": [],
            "last_modified_t": 1_700_000_012,
        },
        {
            "code": "1000000000012",
            "product_name": "Body Lotion",
            "brands": "Nivea",
            "categories": "",
            "categories_tags": ["en:moisturizers"],
            "ingredients_text": "water, glycerin",
            "ingredients_tags": [],
            "quantity": "400 ml",
            "labels": "",
            "countries_tags": [],
            "last_modified_t": 1_700_000_013,
        },
        {
            "code": "1000000000013",
            "product_name": "Retinol Night Cream",
            "brands": "RoC",
            "categories": "",
            "categories_tags": ["en:night-creams"],
            "ingredients_text": "retinol, water",
            "ingredients_tags": ["en:retinol"],
            "quantity": "30 ml",
            "labels": "",
            "countries_tags": [],
            "last_modified_t": 1_700_000_014,
        },
        # 14-19: six more face-cream rows with varying actives for classify tests.
        {
            "code": "1000000000014",
            "product_name": _lang(("en", "Niacinamide + HA Cream")),
            "brands": "The Ordinary",
            "categories": "",
            "categories_tags": ["en:face-creams"],
            "ingredients_text": _lang(("en", "niacinamide, hyaluronic acid")),
            "ingredients_tags": ["en:niacinamide", "en:hyaluronic-acid"],
            "quantity": "30 ml",
            "labels": "",
            "countries_tags": [],
            "last_modified_t": 1_700_000_015,
        },
        {
            "code": "1000000000015",
            "product_name": _lang(("en", "Niacinamide + HA Lotion")),
            "brands": "The Ordinary",
            "categories": "",
            "categories_tags": ["en:face-creams"],
            "ingredients_text": _lang(("en", "niacinamide, hyaluronic acid")),
            "ingredients_tags": ["en:niacinamide", "en:hyaluronic-acid"],
            "quantity": "60 ml",
            "labels": "",
            "countries_tags": [],
            "last_modified_t": 1_700_000_016,
        },
        {
            "code": "1000000000016",
            "product_name": _lang(("en", "Glycolic Gel")),
            "brands": "Paula's Choice",
            "categories": "",
            "categories_tags": ["en:face-creams"],
            "ingredients_text": _lang(("en", "glycolic acid, water")),
            "ingredients_tags": ["en:glycolic-acid"],
            "quantity": "100 ml",
            "labels": "",
            "countries_tags": [],
            "last_modified_t": 1_700_000_017,
        },
        {
            "code": "1000000000017",
            "product_name": _lang(("en", "Lactic Gel")),
            "brands": "The Inkey List",
            "categories": "",
            "categories_tags": ["en:face-creams"],
            "ingredients_text": _lang(("en", "lactic acid, water")),
            "ingredients_tags": ["en:lactic-acid"],
            "quantity": "30 ml",
            "labels": "",
            "countries_tags": [],
            "last_modified_t": 1_700_000_018,
        },
        {
            "code": "1000000000018",
            "product_name": _lang(("en", "SPF Day Cream")),
            "brands": "La Roche-Posay",
            "categories": "",
            "categories_tags": ["en:day-creams"],
            "ingredients_text": _lang(("en", "octinoxate, avobenzone, zinc oxide")),
            "ingredients_tags": ["en:octinoxate"],
            "quantity": "50 ml",
            "labels": "",
            "countries_tags": [],
            "last_modified_t": 1_700_000_019,
        },
        {
            "code": "1000000000019",
            "product_name": _lang(("en", "Peptide Cream")),
            "brands": "Olay",
            "categories": "",
            "categories_tags": ["en:day-creams"],
            "ingredients_text": _lang(("en", "peptides, ceramides")),
            "ingredients_tags": [],
            "quantity": "50 ml",
            "labels": "",
            "countries_tags": [],
            "last_modified_t": 1_700_000_020,
        },
    ]

    schema = {
        "code": pl.String,
        "product_name": pl.List(pl.Struct({"lang": pl.String, "text": pl.String})),
        "brands": pl.String,
        "categories": pl.String,
        "categories_tags": pl.List(pl.String),
        "ingredients_text": pl.List(pl.Struct({"lang": pl.String, "text": pl.String})),
        "ingredients_tags": pl.List(pl.String),
        "quantity": pl.String,
        "labels": pl.String,
        "countries_tags": pl.List(pl.String),
        "last_modified_t": pl.Int64,
    }
    # Coerce plain-string rows into the struct-list schema.
    norm_rows: list[dict[str, object]] = []
    for r in rows:
        row: dict[str, object] = {}
        for col, _dtype in schema.items():
            val = r.get(col)
            if col in ("product_name", "ingredients_text") and isinstance(val, str):
                val = [{"lang": "en", "text": val}]
            row[col] = val
        norm_rows.append(row)

    df = pl.DataFrame(norm_rows, schema=schema)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    df.write_parquet(FIXTURE_PARQUET)
    return FIXTURE_PARQUET


@pytest.fixture(scope="session")
def fixture_parquet() -> Path:
    """Path to the committed hand-crafted ingest fixture."""
    if not FIXTURE_PARQUET.exists():
        build_fixture_parquet()
    return FIXTURE_PARQUET
