"""Step 1 ingest tests: apparel brand derivation, drops, dedupe, malformed rows."""

from __future__ import annotations

from substitutes_agent.step1_ingest import (
    derive_brand,
    ingest,
    load_styles_csv,
    normalize_brand,
)

# ---------------------------------------------------------------------------
# normalize_brand
# ---------------------------------------------------------------------------


def test_normalize_brand_strips_accents_and_suffixes() -> None:
    assert normalize_brand("L’Oréal") == "l’oreal"
    assert normalize_brand("SomeBrand S.A.") == "somebrand"
    assert normalize_brand("CleanCo LLC") == "cleanco"
    assert normalize_brand("Levi's") == "levi's"
    assert normalize_brand("") == ""


# ---------------------------------------------------------------------------
# derive_brand (table-driven)
# ---------------------------------------------------------------------------


def test_derive_brand_two_tokens_before_gender() -> None:
    assert (
        derive_brand("Peter England Men Party Blue Jeans", "Jeans") == "peter england"
    )


def test_derive_brand_one_token_before_gender() -> None:
    assert derive_brand("Nike Men Blue T-shirt", "Tshirts") == "nike"


def test_derive_brand_stops_at_colour() -> None:
    assert derive_brand("Reebok Black Track Pant", "Track Pants") == "reebok"


def test_derive_brand_stops_at_article_noun() -> None:
    assert derive_brand("Ferrari Tee", "Tshirts") == "ferrari"


def test_derive_brand_caps_at_three_tokens() -> None:
    # No boundary token before the cap; consumes three tokens then stops.
    assert (
        derive_brand("Nike Mean Team India Cricket Jersey", "Tshirts")
        == "nike mean team"
    )


def test_derive_brand_caps_with_filler_words() -> None:
    assert (
        derive_brand("United Colors of Benetton Men Red Sweater", "Sweaters")
        == "united colors of"
    )


def test_derive_brand_fallback_when_first_token_is_boundary() -> None:
    # First token is a colour -> boundary; fall back to the first token.
    assert derive_brand("Black T-shirt", "Tshirts") == "black"


def test_derive_brand_strips_legal_suffix_after_extraction() -> None:
    assert derive_brand("SomeBrand S.A. Men Black Jacket", "Jackets") == "somebrand"


def test_derive_brand_mens_apostrophe_is_boundary() -> None:
    assert (
        derive_brand("Lotto Men's Court Logo White Sneakers", "Sports Shoes") == "lotto"
    )


def test_derive_brand_empty() -> None:
    assert derive_brand("", "Tshirts") == ""
    assert derive_brand("   ", "") == ""


# ---------------------------------------------------------------------------
# ingest end-to-end on the fixture
# ---------------------------------------------------------------------------


def test_ingest_filter_counts(ingest_fixture_csv: object) -> None:
    df, log = ingest(str(ingest_fixture_csv))
    assert log.input_rows == 20
    f = log.filters
    # 20 - 2 (empty name, empty articleType) = 18 after required drop.
    assert f["rows_after_required_drop"] == 18
    # 18 - 1 (duplicate id=10) = 17 after dedupe.
    assert f["rows_after_dedupe"] == 17
    # 17 - 1 (Accessories masterCategory) = 16 after category filter.
    assert f["rows_after_category_filter"] == 16
    assert df.height == 16
    assert f["category_counts"] == {"Apparel": 13, "Footwear": 3}


def test_ingest_drops_missing_required(ingest_fixture_csv: object) -> None:
    df, _ = ingest(str(ingest_fixture_csv))
    ids = set(df["id"].to_list())
    assert 4 not in ids  # empty productDisplayName
    assert 5 not in ids  # empty articleType


def test_ingest_drops_non_apparel(ingest_fixture_csv: object) -> None:
    df, _ = ingest(str(ingest_fixture_csv))
    ids = set(df["id"].to_list())
    assert 2 not in ids  # Accessories


def test_ingest_dedupe_keeps_first(ingest_fixture_csv: object) -> None:
    df, _ = ingest(str(ingest_fixture_csv))
    row = df.filter(df["id"] == 10)
    assert row.height == 1
    assert row["productDisplayName"][0] == "Ferrari Tee"


def test_ingest_brand_normalized(ingest_fixture_csv: object) -> None:
    df, _ = ingest(str(ingest_fixture_csv))
    brands = dict(
        zip(df["id"].to_list(), df["brand_normalized"].to_list(), strict=True)
    )
    assert brands[1] == "peter england"
    assert brands[3] == "nike"
    assert brands[7] == "levi's"
    assert brands[10] == "ferrari"
    assert brands[16] == "black"
    assert brands[20] == "somebrand"


def test_ingest_malformed_row_reconstructed(ingest_fixture_csv: object) -> None:
    # id=17 has an unquoted comma in productDisplayName; the loader must
    # reconstruct it rather than drop or mis-split the row.
    df, _ = ingest(str(ingest_fixture_csv))
    row = df.filter(df["id"] == 17)
    assert row.height == 1
    assert row["productDisplayName"][0] == "Tom Tailor Men Denim Shirt, Blue"
    assert row["brand_normalized"][0] == "tom tailor"


def test_ingest_output_sorted_by_id(ingest_fixture_csv: object) -> None:
    df, _ = ingest(str(ingest_fixture_csv))
    ids = df["id"].to_list()
    assert ids == sorted(ids)


def test_load_styles_csv_handles_malformed(ingest_fixture_csv: object) -> None:
    df = load_styles_csv(str(ingest_fixture_csv))
    assert df.height == 20
    # The malformed row is present with its reconstructed name.
    row = df.filter(df["id"] == 17)
    assert row["productDisplayName"][0] == "Tom Tailor Men Denim Shirt, Blue"


def test_ingest_run_log_schema(ingest_fixture_csv: object) -> None:
    _, log = ingest(str(ingest_fixture_csv))
    assert log.step == "step1_ingest"
    assert log.duration_s >= 0.0
    assert log.started_at
    assert "category_counts" in log.filters
    assert "brand_derivation" in log.filters
