"""Step 1 — ingest and normalize apparel styles.csv (deterministic).

Loads the paramaggarwal fashion-product-images dataset (styles.csv),
robustly handling its known malformed rows (unquoted commas inside
`productDisplayName`), drops rows missing required attributes, derives a
brand heuristically from `productDisplayName`, dedupes by `id`, and
filters to the Apparel + Footwear master categories.

No LLM in this step. Same input -> same output.

Brand derivation is a heuristic, not ground truth: the dataset embeds the
brand as the leading tokens of `productDisplayName` (e.g. "Peter England
Men Party Blue Jeans" -> "Peter England") with no separate brand column.
We recover it by consuming leading tokens until we hit a boundary token
(gender / usage / colour / article noun), capped at three tokens. This is
imperfect and documented as a limitation in the README.
"""

from __future__ import annotations

import csv
import re
import time
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from substitutes_agent.models import RunLogEntry

RELEVANT_COLUMNS = [
    "id",
    "gender",
    "masterCategory",
    "subCategory",
    "articleType",
    "baseColour",
    "season",
    "year",
    "usage",
    "productDisplayName",
]

APPAREL_MASTER_CATEGORIES: frozenset[str] = frozenset({"Apparel", "Footwear"})

# Boundary-token vocabularies for brand derivation. Lowercased for matching.
_GENDER_TOKENS: frozenset[str] = frozenset(
    {"men", "women", "boys", "girls", "unisex", "men's", "women's", "boy's", "girl's"}
)
_USAGE_TOKENS: frozenset[str] = frozenset(
    {"casual", "formal", "sports", "ethnic", "party", "travel", "smart", "home"}
)
_COLOUR_TOKENS: frozenset[str] = frozenset(
    {
        "black",
        "white",
        "blue",
        "navy",
        "brown",
        "grey",
        "gray",
        "green",
        "red",
        "yellow",
        "pink",
        "orange",
        "purple",
        "beige",
        "cream",
        "olive",
        "khaki",
        "maroon",
        "burgundy",
        "teal",
        "turquoise",
        "silver",
        "gold",
        "charcoal",
        "peach",
        "lavender",
        "magenta",
        "mauve",
        "copper",
        "bronze",
        "rust",
        "tan",
        "taupe",
        "mustard",
        "nude",
        "skin",
        "steel",
        "metallic",
        "multi",
        "mushroom",
        "coffee",
        "fluorescent",
        "lime",
        "rose",
        "off",
    }
)
_ARTICLE_NOUNS: frozenset[str] = frozenset(
    {
        "shirt",
        "shirts",
        "tshirt",
        "t-shirts",
        "tee",
        "t-shirt",
        "tshirts",
        "top",
        "tops",
        "blouse",
        "sweater",
        "sweatshirt",
        "hoodie",
        "tunic",
        "tank",
        "polo",
        "cardigan",
        "jeans",
        "trouser",
        "trousers",
        "pant",
        "pants",
        "chino",
        "chinos",
        "track",
        "short",
        "shorts",
        "skirt",
        "skorts",
        "legging",
        "leggings",
        "jegging",
        "jeggings",
        "jumpsuit",
        "dress",
        "dresses",
        "kurta",
        "kurti",
        "jacket",
        "jackets",
        "blazer",
        "blazers",
        "coat",
        "coats",
        "waistcoat",
        "vest",
        "vests",
        "shoe",
        "shoes",
        "sneaker",
        "sneakers",
        "loafer",
        "loafers",
        "boot",
        "boots",
        "heel",
        "heels",
        "sandal",
        "sandals",
        "flip",
        "flop",
        "flat",
        "flats",
        "oxford",
        "oxfords",
        "sock",
        "socks",
        "brief",
        "briefs",
        "boxer",
        "boxers",
        "trunk",
        "trunks",
        "innerwear",
        "watch",
        "watches",
        "sunglass",
        "sunglasses",
        "sunglases",
        "belt",
        "belts",
        "bag",
        "bags",
        "backpack",
        "backpacks",
        "handbag",
        "handbags",
        "cap",
        "caps",
        "hat",
        "hats",
        "scarf",
        "scarves",
        "stole",
        "stoles",
        "jewellery",
        "jewelry",
        "wallet",
        "wallets",
        "slipper",
        "slippers",
    }
)


def extract_text(value: object) -> str:
    """Extract an English-preferred text from a struct-list cell.

    Retained from the skincare version because the (still-present) skincare
    Step 2 imports it. Removed once Step 2 is rewritten for apparel.
    """
    if value is None:
        return ""
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
            elif isinstance(item, str) and item.strip() and not first_text:
                first_text = item.strip()
        return first_text
    if isinstance(value, str):
        return value.strip()
    return ""


_BRAND_SUFFIX_RE = re.compile(
    r"\s*[,\s]?\s*"
    r"(?:ltd\.?|limited|inc\.?|llc|co\.|corp\.?|corporation|s\.?a\.?|"
    r"s\.?r\.?l\.?|gmbh|ag|pvt\.?|pvt\.?ltd\.?|bv|n\.?v\.?|s\.?a\.?s\.?)"
    r"\s*$",
    re.IGNORECASE,
)
_WS_RE = re.compile(r"\s+")


def normalize_brand(brand: str) -> str:
    """Lowercase, strip accents, collapse whitespace, drop legal suffixes."""
    if not brand:
        return ""
    text = unicodedata.normalize("NFKD", brand)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().strip()
    for _ in range(3):
        new_text = _BRAND_SUFFIX_RE.sub("", text)
        if new_text == text:
            break
        text = new_text
    text = _WS_RE.sub(" ", text).strip()
    return text


def derive_brand(product_display_name: str, article_type: str) -> str:
    """Heuristically recover the brand from the leading tokens of the name.

    Consume tokens left-to-right until a boundary token (gender / usage /
    colour / article noun) is hit, then stop. Cap the brand at 3 tokens so
    a missing boundary doesn't swallow the whole name. If the very first
    token is a boundary, fall back to the first token alone.
    """
    if not product_display_name:
        return ""
    tokens = product_display_name.split()
    if not tokens:
        return ""

    article_nouns = _ARTICLE_NOUNS
    # Also treat the articleType's own lowercased tokens as boundary nouns
    # so "Nike Casual Shoes" stops at "casual" or "shoes".
    if article_type:
        article_nouns = article_nouns | frozenset(
            t.lower() for t in re.split(r"[\s/-]+", article_type) if t
        )

    brand_tokens: list[str] = []
    for tok in tokens:
        low = re.sub(r"[^a-z'-]", "", tok.lower())
        if not low:
            continue
        if (
            low in _GENDER_TOKENS
            or low in _USAGE_TOKENS
            or low in _COLOUR_TOKENS
            or low in article_nouns
        ):
            break
        brand_tokens.append(tok)
        if len(brand_tokens) >= 3:
            break

    if not brand_tokens:
        # First meaningful token was itself a boundary; fall back to token 0.
        brand_tokens = [tokens[0]]
    return normalize_brand(" ".join(brand_tokens))


def load_styles_csv(path: str | Path) -> pl.DataFrame:
    """Load styles.csv robustly, reconstructing malformed rows.

    The dataset has ~22 rows where `productDisplayName` contains an
    unquoted comma, producing extra fields. polars' CSV reader rejects
    these; we use the stdlib `csv` module (which handles quoted/unquoted
    fields per RFC 4180) and rebuild rows whose trailing field count > 10
    by re-joining the surplus fields back into `productDisplayName`.
    """
    rows: list[list[str]] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        for row in reader:
            if len(row) > len(header):
                # Surplus fields belong to the last (productDisplayName) col.
                row = row[: len(header) - 1] + [",".join(row[len(header) - 1 :])]
            elif len(row) < len(header):
                continue
            rows.append(row)

    schema = {
        "id": pl.String,
        "gender": pl.String,
        "masterCategory": pl.String,
        "subCategory": pl.String,
        "articleType": pl.String,
        "baseColour": pl.String,
        "season": pl.String,
        "year": pl.String,
        "usage": pl.String,
        "productDisplayName": pl.String,
    }
    df = pl.DataFrame(rows, schema=schema, orient="row")
    df = _coerce_and_normalize(df)
    return df


def load_styles(path: str | Path) -> pl.DataFrame:
    """Load the apparel dataset from a CSV or parquet path.

    The committed `data/sample.parquet` fixture is a parquet slice of the
    same schema, so `--sample` runs without the full CSV. Both paths feed
    into the same normalization.
    """
    p = Path(path)
    if p.suffix == ".parquet":
        df = pl.read_parquet(p)
        # Ensure the columns we need are strings for uniform handling.
        for col in ("id", "year"):
            if col in df.columns and df[col].dtype != pl.String:
                df = df.with_columns(pl.col(col).cast(pl.String))
        return _coerce_and_normalize(df)
    return load_styles_csv(p)


def _coerce_and_normalize(df: pl.DataFrame) -> pl.DataFrame:
    """Cast id/year to Int64 leniently and turn empty/NA sentinels to nulls."""
    if "id" in df.columns:
        df = df.with_columns(pl.col("id").cast(pl.Int64, strict=False).alias("id"))
    if "year" in df.columns and df["year"].dtype != pl.Int64:
        df = df.with_columns(pl.col("year").cast(pl.Int64, strict=False).alias("year"))
    for col in (
        "gender",
        "masterCategory",
        "subCategory",
        "articleType",
        "baseColour",
        "season",
        "usage",
        "productDisplayName",
    ):
        if col in df.columns:
            df = df.with_columns(
                pl.when(
                    pl.col(col)
                    .cast(pl.String)
                    .str.strip_chars()
                    .is_in(["", "NA", "N/A"])
                )
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )
    return df


def ingest(input_path: str | Path) -> tuple[pl.DataFrame, RunLogEntry]:
    """Run Step 1. Returns (normalized_dataframe, run_log_entry)."""
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat(timespec="seconds")

    df = load_styles(input_path)
    input_rows = df.height

    present = [c for c in RELEVANT_COLUMNS if c in df.columns]
    df = df.select(present)

    # Drop rows missing any required attribute.
    required = ["productDisplayName", "masterCategory", "articleType", "baseColour"]
    before_required = df.height
    mask = pl.lit(True)
    for col in required:
        if col in df.columns:
            mask = mask & pl.col(col).is_not_null()
    df = df.filter(mask)
    rows_after_required = df.height

    # Derive brand heuristically from productDisplayName + articleType.
    if "productDisplayName" in df.columns:
        df = df.with_columns(
            pl.struct(["productDisplayName", "articleType"])
            .map_elements(
                lambda r: derive_brand(
                    str(r["productDisplayName"]), str(r["articleType"])
                ),
                return_dtype=pl.String,
            )
            .alias("brand_normalized")
        )
    else:
        df = df.with_columns(pl.lit("").alias("brand_normalized"))

    # Dedupe by id.
    before_dedupe = df.height
    df = df.unique(subset=["id"], keep="first")
    rows_after_dedupe = df.height

    # Filter to Apparel + Footwear master categories.
    if "masterCategory" in df.columns:
        before_cat = df.height
        df = df.filter(pl.col("masterCategory").is_in(list(APPAREL_MASTER_CATEGORIES)))
        rows_after_category = df.height
        category_counts = {
            cat: int((df["masterCategory"] == cat).sum())
            for cat in sorted(APPAREL_MASTER_CATEGORIES)
        }
    else:
        before_cat = df.height
        df = df.head(0)
        rows_after_category = 0
        category_counts = {}

    # Reorder + sort for a stable artifact.
    out_cols = [
        "id",
        "brand_normalized",
        "productDisplayName",
        "gender",
        "masterCategory",
        "subCategory",
        "articleType",
        "baseColour",
        "season",
        "year",
        "usage",
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    df = df.select(out_cols).sort("id")

    duration = time.monotonic() - started
    entry = RunLogEntry(
        step="step1_ingest",
        started_at=started_at,
        duration_s=round(duration, 4),
        input_rows=input_rows,
        output_rows=df.height,
        filters={
            "rows_before_required_drop": before_required,
            "rows_after_required_drop": rows_after_required,
            "rows_before_dedupe": before_dedupe,
            "rows_after_dedupe": rows_after_dedupe,
            "rows_before_category_filter": before_cat,
            "rows_after_category_filter": rows_after_category,
            "category_counts": category_counts,
            "brand_derivation": "leading tokens of productDisplayName until a "
            "boundary token (gender/usage/colour/article noun), capped at 3",
        },
        notes="Brand has no dedicated column in this dataset; derived "
        "heuristically from productDisplayName. Imperfect by design.",
    )
    return df, entry
