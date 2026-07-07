"""Step 2 — build the product ontology (rules; LLM optional).

For each SKU from Step 1, extract structured attributes:
  format, skin_type_claims, key_actives, pack_size_ml, target_area.

Rules are deliberately simple and auditable. The optional LLM path
(env-gated, implemented in ``llm.py``) only fills ``format`` /
``key_actives`` for rows the rules leave ambiguous, and is cached to
``output/.llm_cache/{code}.json`` so re-runs are deterministic.
"""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import polars as pl

from substitutes_agent.models import Format, OntologyRecord, RunLogEntry, TargetArea
from substitutes_agent.step1_ingest import extract_text

# ---------------------------------------------------------------------------
# Curated vocabularies (closed lists — do not get fancy).
# ---------------------------------------------------------------------------

# format: token -> canonical label. Searched in priority order so that
# "oil-free cream" classifies as cream, not oil.
_FORMAT_TOKENS: list[tuple[str, str]] = [
    ("cream", "cream"),
    ("creme", "cream"),
    ("crème", "cream"),
    # Moisturizers are creams; ordered before "oil" so "oil-free moisturizer"
    # classifies as cream rather than matching the "oil" token.
    ("moisturizer", "cream"),
    ("moisturiser", "cream"),
    ("lotion", "lotion"),
    ("serum", "serum"),
    ("gel", "gel"),
    ("balm", "balm"),
    ("mask", "mask"),
    ("sheet", "mask"),
    ("oil", "oil"),
]

_SKIN_TYPE_TOKENS: list[tuple[str, str]] = [
    ("dry", "dry"),
    ("drying", "dry"),
    ("oily", "oily"),
    ("oil-control", "oily"),
    ("combination", "combination"),
    ("combo", "combination"),
    ("sensitive", "sensitive"),
    ("mature", "mature"),
    ("normal", "normal"),
]

# key_actives: (regex pattern, canonical label). Matched case-insensitively
# against the lowercased ingredients text AND the normalized ingredients_tags.
# Order is the canonical output order.
_ACTIVE_RULES: list[tuple[str, str]] = [
    (r"\bretinol\b", "retinol"),
    (r"\bretinal\b|retinaldehyde", "retinal"),
    (r"\bniacinamide\b", "niacinamide"),
    (r"hyaluronic acid|sodium hyaluronate", "hyaluronic acid"),
    (r"ascorbic acid|vitamin c|l-ascorbic", "vitamin c"),
    (r"salicylic acid|bha\b", "salicylic acid"),
    (r"glycolic acid", "glycolic acid"),
    (r"lactic acid", "lactic acid"),
    (r"ceramide", "ceramides"),
    (r"peptide", "peptides"),
    (r"octinoxate|avobenzone|zinc oxide|titanium dioxide|spf\b|sunscreen", "spf"),
]

_TARGET_AREA_TOKENS: list[tuple[str, str]] = [
    ("eye", "eye"),
    ("eyes", "eye"),
    ("contour", "eye"),
    ("lip", "lip"),
    ("lips", "lip"),
    ("body", "body"),
    ("face", "face"),
    ("facial", "face"),
]

_OZ_TO_ML = 29.5735
_PACK_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(ml|g|oz|fl\.?\s*oz)", re.IGNORECASE)
_WORD_BOUNDARY_CACHE: dict[str, re.Pattern[str]] = {}


def _word_pattern(token: str) -> re.Pattern[str]:
    pat = _WORD_BOUNDARY_CACHE.get(token)
    if pat is None:
        pat = re.compile(r"\b" + re.escape(token.lower()) + r"\b", re.IGNORECASE)
        _WORD_BOUNDARY_CACHE[token] = pat
    return pat


# ---------------------------------------------------------------------------
# Pure attribute extractors (unit-testable).
# ---------------------------------------------------------------------------


def extract_format(product_name: str) -> str:
    """Token-match the product format from the product name."""
    if not product_name:
        return "unknown"
    low = product_name.lower()
    for token, label in _FORMAT_TOKENS:
        if _word_pattern(token).search(low):
            return label
    return "unknown"


def extract_skin_type_claims(product_name: str, labels: str) -> str:
    """Token-match skin-type claims from product name + labels."""
    haystack = f"{product_name} {labels}".lower()
    found: list[str] = []
    for token, label in _SKIN_TYPE_TOKENS:
        if label in found:
            continue
        if _word_pattern(token).search(haystack):
            found.append(label)
    return ",".join(found)


def extract_key_actives(
    ingredients_text: str, ingredients_tags: list[str]
) -> list[str]:
    """Match the curated INCI vocabulary against ingredients text + tags.

    ingredients_tags from Open Beauty Facts arrive normalized as
    ``en:hyaluronic-acid``; we strip the ``en:`` prefix and turn dashes
    into spaces so the same patterns match both sources.
    """
    text = (ingredients_text or "").lower()
    tags_norm = " ".join(
        t.split(":", 1)[-1].replace("-", " ").lower() for t in (ingredients_tags or [])
    )
    haystack = f"{text} {tags_norm}"
    found: list[str] = []
    for pattern, label in _ACTIVE_RULES:
        if label in found:
            continue
        if re.search(pattern, haystack):
            found.append(label)
    return found


def extract_pack_size_ml(quantity: str) -> float | None:
    """Parse a numeric pack size from the quantity string, in ml.

    oz is converted to ml (1 oz ~ 29.5735 ml). g is treated as 1:1 ml
    (cream density ~1.0); this is a documented simplification.
    """
    if not quantity:
        return None
    m = _PACK_RE.search(quantity)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2).lower().replace("fl.", "").replace("fl", "").strip()
    if unit == "oz":
        return round(value * _OZ_TO_ML, 2)
    # ml and g map 1:1 to ml-equivalent.
    return round(value, 2)


def extract_target_area(product_name: str) -> str:
    """Token-match the target area from the product name."""
    if not product_name:
        return "unknown"
    low = product_name.lower()
    for token, label in _TARGET_AREA_TOKENS:
        if _word_pattern(token).search(low):
            return label
    return "unknown"


# ---------------------------------------------------------------------------
# Orchestrator.
# ---------------------------------------------------------------------------


def _row_to_record(row: dict[str, object]) -> OntologyRecord:
    code = str(row.get("code") or "")
    brand = str(row.get("brand_normalized") or "")
    product_name = str(row.get("product_name") or "")

    ingredients_text = extract_text(row.get("ingredients_text"))
    ingredients_tags_raw = row.get("ingredients_tags")
    if isinstance(ingredients_tags_raw, list):
        ingredients_tags = [str(t) for t in ingredients_tags_raw]
    else:
        ingredients_tags = []
    labels = str(row.get("labels") or "")
    quantity = str(row.get("quantity") or "")

    fmt = extract_format(product_name)
    skin = extract_skin_type_claims(product_name, labels)
    actives = extract_key_actives(ingredients_text, ingredients_tags)
    pack = extract_pack_size_ml(quantity)
    area = extract_target_area(product_name)

    needs_llm = (fmt == "unknown") or (not actives and bool(ingredients_text))
    confidence = 0.6 if needs_llm else 1.0

    return OntologyRecord(
        code=code,
        brand_normalized=brand,
        product_name=product_name,
        format=cast(Format, fmt),
        skin_type_claims=[s for s in skin.split(",") if s],
        key_actives=actives,
        pack_size_ml=pack,
        target_area=cast(TargetArea, area),
        source="rules",
        confidence=confidence,
    )


def build_ontology(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> tuple[list[OntologyRecord], RunLogEntry]:
    """Run Step 2: read Step 1 parquet, emit ontology records + log entry."""
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat(timespec="seconds")

    df = pl.read_parquet(input_path)
    input_rows = df.height
    rows = df.to_dicts()

    records: list[OntologyRecord] = []
    ambiguous = 0
    for row in rows:
        rec = _row_to_record(row)
        if rec.source == "rules" and rec.confidence < 1.0:
            ambiguous += 1
        records.append(rec)

    records.sort(key=lambda r: r.code)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps([r.model_dump() for r in records], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    duration = time.monotonic() - started
    entry = RunLogEntry(
        step="step2_ontology",
        started_at=started_at,
        duration_s=round(duration, 4),
        input_rows=input_rows,
        output_rows=len(records),
        filters={
            "ambiguous_low_confidence_rows": ambiguous,
            "llm_enabled": False,
        },
        notes="Pure-rules extraction. LLM enrichment is env-gated and off "
        "by default; ambiguous rows (unknown format or empty actives) "
        "are flagged with confidence=0.6.",
    )
    return records, entry
