"""Step 1 ingest tests: filter counts, dedupe, brand normalization."""

from __future__ import annotations

from substitutes_agent.step1_ingest import (
    extract_text,
    ingest,
    normalize_brand,
)


def test_extract_text_english_preferred() -> None:
    val = [{"lang": "fr", "text": "Crème"}, {"lang": "en", "text": "Cream"}]
    assert extract_text(val) == "Cream"


def test_extract_text_falls_back_to_first_when_no_en() -> None:
    val = [{"lang": "de", "text": "Creme"}, {"lang": "fr", "text": "Crème"}]
    assert extract_text(val) == "Creme"


def test_extract_text_handles_plain_string() -> None:
    assert extract_text("Simple Cream") == "Simple Cream"


def test_extract_text_handles_empty_and_none() -> None:
    assert extract_text(None) == ""
    assert extract_text([]) == ""
    assert extract_text([{"lang": "en", "text": ""}]) == ""
    assert extract_text([{"lang": "fr", "text": "  "}]) == ""


def test_normalize_brand_strips_accents() -> None:
    assert normalize_brand("L’Oréal") == "l’oreal"
    assert normalize_brand("Nüdé") == "nude"


def test_normalize_brand_strips_legal_suffixes() -> None:
    assert normalize_brand("CeraVe, Ltd.") == "cerave"
    assert normalize_brand("Olay Inc.") == "olay"
    assert normalize_brand("CleanCo LLC") == "cleanco"
    assert normalize_brand("Nüdé S.A.") == "nude"


def test_normalize_brand_collapses_whitespace() -> None:
    assert normalize_brand("St.   Ives") == "st. ives".replace("  ", " ")
    assert normalize_brand("  Nivea  ") == "nivea"


def test_normalize_brand_empty() -> None:
    assert normalize_brand("") == ""
    assert normalize_brand("   ") == ""


def test_ingest_filter_counts(fixture_parquet: object) -> None:
    df, log = ingest(str(fixture_parquet))
    # Input is 20 rows.
    assert log.input_rows == 20
    f = log.filters
    # 19 face-cream rows (row 2 "Shampoo" filtered out by category).
    assert f["rows_after_category_filter"] == 19
    # 19 - 2 dropped (empty name, empty brand) = 17.
    assert f["rows_after_required_drop"] == 17
    # 17 - 1 deduped (duplicate code 1000000000006) = 16.
    assert f["rows_after_dedupe"] == 16
    assert df.height == 16


def test_ingest_dedupe_keeps_most_recent(fixture_parquet: object) -> None:
    df, _ = ingest(str(fixture_parquet))
    row = df.filter(df["code"] == "1000000000006")
    assert row.height == 1
    assert row["product_name"][0] == "Newer Name"


def test_ingest_drops_non_face_cream(fixture_parquet: object) -> None:
    df, _ = ingest(str(fixture_parquet))
    codes = set(df["code"].to_list())
    assert "1000000000003" not in codes  # shampoo, filtered by category


def test_ingest_drops_empty_name_and_brand(fixture_parquet: object) -> None:
    df, _ = ingest(str(fixture_parquet))
    codes = set(df["code"].to_list())
    assert "1000000000004" not in codes  # empty name
    assert "1000000000005" not in codes  # empty brand


def test_ingest_brand_normalized_column(fixture_parquet: object) -> None:
    df, _ = ingest(str(fixture_parquet))
    brands = dict(
        zip(df["code"].to_list(), df["brand_normalized"].to_list(), strict=True)
    )
    assert brands["1000000000001"] == "l’oreal"
    assert brands["1000000000002"] == "cerave"
    assert brands["1000000000007"] == "nude"
    assert brands["1000000000008"] == "cleanco"


def test_ingest_output_sorted_by_code(fixture_parquet: object) -> None:
    df, _ = ingest(str(fixture_parquet))
    codes = df["code"].to_list()
    assert codes == sorted(codes)


def test_ingest_run_log_schema(fixture_parquet: object) -> None:
    _, log = ingest(str(fixture_parquet))
    assert log.step == "step1_ingest"
    assert log.duration_s >= 0.0
    assert log.started_at  # non-empty ISO string
    assert "face_cream_tag_matches" in log.filters
    assert isinstance(log.filters["face_cream_tag_matches"], dict)
