"""Step 1 — ingest and normalize (deterministic).

Loads the Open Beauty Facts beauty parquet, filters to face-cream /
moisturizer categories, extracts an English product name, drops rows
missing name or brand, dedupes by EAN keeping the most-recently-modified,
and normalizes the brand string.

No LLM in this step. Same input -> same output.
"""

from __future__ import annotations

import re
import time
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from substitutes_agent.models import RunLogEntry

RELEVANT_COLUMNS = [
    "code",
    "product_name",
    "brands",
    "categories",
    "categories_tags",
    "ingredients_text",
    "ingredients_tags",
    "quantity",
    "labels",
    "countries_tags",
    "last_modified_t",
]

# Exact tag set from the task spec. Open Beauty Facts stores tags with
# mixed case and spaces (e.g. "en:Face creams"); we normalize both sides
# to lowercase + dash so the spec's tag strings match the real data.
FACE_CREAM_TAGS: frozenset[str] = frozenset(
    {
        "en:face-creams",
        "en:moisturizers",
        "en:face-moisturizers",
        "en:facial-skin-care",
        "en:day-creams",
        "en:night-creams",
    }
)

# Trailing legal suffixes to strip during brand normalization. Small closed
# list on purpose — see spec. Matched case-insensitively at the end of the
# brand string, optionally preceded by a comma or whitespace.
_BRAND_SUFFIX_RE = re.compile(
    r"\s*[,\s]?\s*"
    r"(?:ltd\.?|limited|inc\.?|llc|co\.|corp\.?|corporation|s\.?a\.?|"
    r"s\.?r\.?l\.?|gmbh|ag|pvt\.?|pvt\.?ltd\.?|bv|n\.?v\.?|s\.?a\.?s\.?)"
    r"\s*$",
    re.IGNORECASE,
)

_WS_RE = re.compile(r"\s+")


def normalize_tag(tag: str) -> str:
    """Normalize a category tag: lowercase, spaces -> dashes."""
    return tag.strip().lower().replace(" ", "-")


def extract_text(value: object) -> str:
    """Extract an English-preferred text from a product_name/ingredients_text cell.

    The Open Beauty Facts parquet stores these as a list of
    ``{lang, text}`` structs (one per language). Defensive handling also
    accepts a plain string or a list of strings.
    """
    if value is None:
        return ""
    # Struct-list form: list[struct{lang, text}]
    if isinstance(value, list):
        if not value:
            return ""
        first_text = ""
        for item in value:
            if isinstance(item, dict):
                lang = str(item.get("lang") or "").lower()
                text = str(item.get("text") or "").strip()
                if not text:
                    continue
                if not first_text:
                    first_text = text
                if lang == "en":
                    return text
            elif isinstance(item, str):
                if not first_text and item.strip():
                    first_text = item.strip()
            # polars struct rows may arrive as mapping-like; ignore others.
        return first_text
    if isinstance(value, str):
        return value.strip()
    return ""


def normalize_brand(brand: str) -> str:
    """Lowercase, strip accents, collapse whitespace, drop legal suffixes."""
    if not brand:
        return ""
    text = unicodedata.normalize("NFKD", brand)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().strip()
    # Strip trailing legal suffixes (possibly several).
    for _ in range(3):
        new_text = _BRAND_SUFFIX_RE.sub("", text)
        if new_text == text:
            break
        text = new_text
    text = _WS_RE.sub(" ", text).strip()
    return text


def _pick_text_rowwise(series: pl.Series) -> pl.Series:
    """Row-wise English-preferred extraction for a list[struct] / str column.

    Converting the whole column to Python objects once (via ``.to_list()``)
    lets ``extract_text`` see plain dicts/strings instead of polars struct
    rows, which ``map_elements`` would otherwise pass through opaquely.
    """
    as_py = series.to_list()
    return pl.Series([extract_text(v) for v in as_py], dtype=pl.String)


def ingest(input_path: str | Path) -> tuple[pl.DataFrame, RunLogEntry]:
    """Run Step 1. Returns (normalized_dataframe, run_log_entry)."""
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat(timespec="seconds")

    raw = pl.read_parquet(input_path)
    input_rows = raw.height

    # Select only relevant columns (defensive: keep those that exist).
    present = [c for c in RELEVANT_COLUMNS if c in raw.columns]
    df = raw.select(present)

    # Filter to face-cream categories. Normalize each tag (lowercase, space->dash)
    # so the spec tag strings match Open Beauty Facts' mixed-case/space tags.
    if "categories_tags" in df.columns:
        tags_norm = df["categories_tags"].list.eval(
            pl.element().map_elements(normalize_tag, return_dtype=pl.String)
        )
        match_mask = pl.lit(False)
        for tag in sorted(FACE_CREAM_TAGS):
            match_mask = match_mask | tags_norm.list.contains(tag)
        before_filter = df.height
        df = df.filter(match_mask)
        rows_after_category = df.height
        tag_match_table = {
            tag: int(tags_norm.list.contains(tag).sum())
            for tag in sorted(FACE_CREAM_TAGS)
        }
    else:
        before_filter = df.height
        df = df.head(0)
        rows_after_category = 0
        tag_match_table = {}

    # Extract English-preferred product name from the struct-list column.
    if "product_name" in df.columns:
        df = df.with_columns(
            _pick_text_rowwise(df["product_name"]).alias("product_name"),
        )

    # Require non-empty product_name AND non-empty brands.
    has_name = df["product_name"].cast(pl.String).str.strip_chars().str.len_chars() > 0
    brands_col = (
        df["brands"] if "brands" in df.columns else pl.lit(None, dtype=pl.String)
    )
    has_brand = brands_col.cast(pl.String).str.strip_chars().str.len_chars() > 0
    before_drop = df.height
    df = df.filter(has_name & has_brand)
    rows_after_required = df.height

    # Dedupe by code (EAN), keeping the most recently modified.
    if "code" in df.columns and "last_modified_t" in df.columns:
        df = df.sort("code", "last_modified_t", descending=[False, True]).unique(
            subset=["code"], keep="first"
        )
    elif "code" in df.columns:
        df = df.unique(subset=["code"], keep="first")
    rows_after_dedupe = df.height

    # Normalize brand into a new column; keep original for traceability.
    if "brands" in df.columns:
        df = df.with_columns(
            df["brands"]
            .map_elements(normalize_brand, return_dtype=pl.String)
            .alias("brand_normalized")
        )
    else:
        df = df.with_columns(pl.lit("").alias("brand_normalized"))

    # Reorder columns for a stable artifact.
    out_cols = [
        "code",
        "brand_normalized",
        "product_name",
        "brands",
        "categories",
        "categories_tags",
        "ingredients_text",
        "ingredients_tags",
        "quantity",
        "labels",
        "countries_tags",
        "last_modified_t",
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    df = df.select(out_cols).sort("code")

    duration = time.monotonic() - started
    entry = RunLogEntry(
        step="step1_ingest",
        started_at=started_at,
        duration_s=round(duration, 4),
        input_rows=input_rows,
        output_rows=df.height,
        filters={
            "selected_columns": len(present),
            "rows_before_category_filter": before_filter,
            "rows_after_category_filter": rows_after_category,
            "rows_before_required_drop": before_drop,
            "rows_after_required_drop": rows_after_required,
            "rows_after_dedupe": rows_after_dedupe,
            "face_cream_tag_matches": tag_match_table,
        },
        notes="Normalized category tags (lowercase, space->dash) to match spec "
        "tag strings against Open Beauty Facts' mixed-case tags.",
    )
    return df, entry
