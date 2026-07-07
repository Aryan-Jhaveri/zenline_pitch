"""Step 2 — build the product ontology (rules; LLM optional).

For each SKU from Step 1, extract structured attributes into a typed
record: article_type (aligned to my Vastra taxonomy), master_category,
base_colour (normalized to a colour family), usage, gender, season,
plus pattern and material regex-extracted from productDisplayName.

Rules are deliberately simple and auditable. The optional LLM path
(env-gated by ANTHROPIC_API_KEY or GOOGLE_API_KEY) only runs for rows
where pattern AND material are both null despite a long product name,
and the prompt includes the set of already-observed values in the run
so the model aligns to a stable vocabulary rather than free-form output
— the same pattern my Vastra categorizer uses. Cached to
output/.llm_cache/{id}.json so re-runs are deterministic.

No key set -> pure rules, no warnings. That is the point.
"""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from substitutes_agent.llm import (
    any_key_present,
    extract_attributes,
    pick_default_model,
)
from substitutes_agent.models import OntologyRecord, RunLogEntry, Source
from substitutes_agent.vastra_taxonomy import align_article_type

# ---------------------------------------------------------------------------
# Colour family map (documented inline, per spec).
#
# Exact-same colour -> similarity 1.0; same family -> 0.6; different
# family -> 0.0. Built by hand from the 47 baseColour values present in
# the paramaggarwal dataset. "Multi" is its own family. Anything not
# listed falls through to "other" (similarity 0.0 vs anything specific).
# ---------------------------------------------------------------------------

_COLOUR_FAMILY: dict[str, str] = {
    # greyscale dark
    "black": "black",
    "charcoal": "black",
    "grey": "black",
    "grey melange": "black",
    "steel": "black",
    "metallic": "black",
    "silver": "black",
    # light neutrals
    "white": "white",
    "off white": "white",
    "cream": "white",
    "nude": "white",
    "skin": "white",
    "beige": "white",
    "peach": "white",
    # blue family (navy/blue/teal cluster as one family)
    "blue": "blue",
    "navy blue": "blue",
    "navy": "blue",
    "teal": "blue",
    "turquoise blue": "blue",
    "sea green": "blue",
    # green family
    "green": "green",
    "fluorescent green": "green",
    "lime green": "green",
    "olive": "green",
    "khaki": "green",
    # yellow family
    "yellow": "yellow",
    "mustard": "yellow",
    "gold": "yellow",
    # brown family
    "brown": "brown",
    "coffee brown": "brown",
    "mushroom brown": "brown",
    "tan": "brown",
    "taupe": "brown",
    "copper": "brown",
    "bronze": "brown",
    "rust": "brown",
    # red family
    "red": "red",
    "maroon": "red",
    "burgundy": "red",
    "orange": "red",
    # pink family
    "pink": "pink",
    "magenta": "pink",
    "rose": "pink",
    "mauve": "pink",
    # purple family
    "purple": "purple",
    "lavender": "purple",
    # multi
    "multi": "multi",
}


def colour_family(base_colour: str) -> str:
    """Map a raw baseColour string onto its colour family (lowercased)."""
    if not base_colour:
        return "other"
    return _COLOUR_FAMILY.get(base_colour.strip().lower(), "other")


def colour_similarity(a: str, b: str) -> float:
    """Similarity in {0.0, 0.6, 1.0} based on the colour-family map.

    1.0 same exact colour, 0.6 same family, 0.0 different family.
    Either side "other"/null -> 0.5 (neutral, no signal).
    """
    fa = colour_family(a)
    fb = colour_family(b)
    if fa == "other" or fb == "other":
        return 0.5
    if a.strip().lower() == b.strip().lower():
        return 1.0
    if fa == fb:
        return 0.6
    return 0.0


# ---------------------------------------------------------------------------
# Pattern and material vocabularies (regex on productDisplayName).
# ---------------------------------------------------------------------------

_PATTERN_RULES: list[tuple[str, str]] = [
    # Specific patterns before the generic "Printed" rule so that
    # "Flower Print Top" classifies as Floral, not Printed.
    (r"\bsolid\b", "Solid"),
    (r"\bstrip(?:e|ed|es)\b|\bstripes\b", "Striped"),
    (r"\bcheck(?:ed|s)?\b|\bchecks\b|\bgingham\b|\bplaid\b", "Checked"),
    (r"\bfloral\b|\bflowers?\b", "Floral"),
    (r"\bprint(?:ed)?\b|\bprints\b", "Printed"),
    (r"\bgraphic\b|\bgraphics\b", "Graphic"),
    (r"\bpolka\b", "Polka"),
    (r"\btie[- ]?dye\b", "Tie-dye"),
    (r"\bcolor[- ]?block\b|\bcolour[- ]?block\b", "Colorblock"),
]

_MATERIAL_RULES: list[tuple[str, str]] = [
    (r"\bcotton\b", "Cotton"),
    (r"\bdenim\b", "Denim"),
    (r"\blinen\b|\blinen\b", "Linen"),
    (r"\bsilk\b", "Silk"),
    (r"\bpolyester\b", "Polyester"),
    (r"\bleather\b", "Leather"),
    (r"\bwool\b|\bwoollen\b|\bwoolen\b", "Wool"),
    (r"\bcashmere\b", "Cashmere"),
    (r"\bnylon\b", "Nylon"),
    (r"\bspandex\b", "Spandex"),
    (r"\bviscose\b", "Viscose"),
    (r"\brayon\b", "Rayon"),
    (r"\bcanvas\b", "Canvas"),
    (r"\bsuede\b", "Suede"),
    (r"\bknit\b|\bknitted\b", "Knit"),
    (r"\bflannel\b", "Flannel"),
    (r"\bchiffon\b", "Chiffon"),
    (r"\bgeorgette\b", "Georgette"),
    (r"\bcrepe\b", "Crepe"),
    (r"\bsatin\b", "Satin"),
    # "Jersey" omitted: in this dataset it almost always denotes the
    # garment (cricket jersey), not the knit fabric -> avoid false positives.
    (r"\bfleece\b", "Fleece"),
    (r"\bmesh\b", "Mesh"),
    (r"\blycra\b", "Lycra"),
    (r"\bneoprene\b", "Neoprene"),
    (r"\bacrylic\b", "Acrylic"),
    (r"\bjute\b", "Jute"),
]


def extract_pattern(product_name: str) -> str | None:
    """Regex-extract a pattern label from the product name, if present."""
    if not product_name:
        return None
    low = product_name.lower()
    for pattern, label in _PATTERN_RULES:
        if re.search(pattern, low):
            return label
    return None


def extract_material(product_name: str) -> str | None:
    """Regex-extract a material label from the product name, if present."""
    if not product_name:
        return None
    low = product_name.lower()
    for pattern, label in _MATERIAL_RULES:
        if re.search(pattern, low):
            return label
    return None


# ---------------------------------------------------------------------------
# Orchestrator.
# ---------------------------------------------------------------------------

_LLM_NAME_LONG_THRESHOLD = 20


def _row_to_record(row: dict[str, object]) -> OntologyRecord:
    code = str(row.get("id") or "")
    brand = str(row.get("brand_normalized") or "")
    product_name = str(row.get("productDisplayName") or "")
    article_type_raw = str(row.get("articleType") or "")
    master_category = str(row.get("masterCategory") or "")
    base_colour_raw = str(row.get("baseColour") or "")
    usage = str(row.get("usage") or "")
    gender = str(row.get("gender") or "")
    season_val = row.get("season")
    season = str(season_val) if season_val else None

    article_type = align_article_type(article_type_raw)
    base_colour = colour_family(base_colour_raw)
    pattern = extract_pattern(product_name)
    material = extract_material(product_name)

    needs_llm = (
        pattern is None
        and material is None
        and len(product_name) > _LLM_NAME_LONG_THRESHOLD
    )

    # LLM hook: filled out in the multi-provider wrapper (commit #13).
    # When a key is present and a row needs the LLM, the wrapper would be
    # invoked here to fill pattern/material, aligning to already-observed
    # values (Vastra buildPrompt reuse pattern). For now, pure rules.
    source: Source = "rules"
    if needs_llm:
        # Ambiguous row that the rules left with no pattern/material and no
        # LLM is available (or the hook isn't wired yet) -> flag low confidence.
        confidence = 0.6
    elif pattern is not None or material is not None:
        confidence = 0.7
    else:
        confidence = 1.0

    return OntologyRecord(
        code=code,
        brand_normalized=brand,
        product_name=product_name,
        article_type=article_type,
        master_category=master_category,
        base_colour=base_colour,
        usage=usage,
        gender=gender,
        season=season,
        pattern=pattern,
        material=material,
        source=source,
        confidence=confidence,
    )


def build_ontology(
    input_path: str | Path,
    output_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[list[OntologyRecord], RunLogEntry]:
    """Run Step 2: read Step 1 parquet, emit ontology records + log entry."""
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat(timespec="seconds")

    df = pl.read_parquet(input_path)
    input_rows = df.height
    rows = df.to_dicts()

    records: list[OntologyRecord] = []
    regex_pattern_rows = 0
    regex_material_rows = 0
    ambiguous_rows = 0
    for row in rows:
        rec = _row_to_record(row)
        if rec.pattern is not None:
            regex_pattern_rows += 1
        if rec.material is not None:
            regex_material_rows += 1
        if rec.confidence == 0.6:
            ambiguous_rows += 1
        records.append(rec)

    records.sort(key=lambda r: r.code)

    # Optional LLM enrichment (env-gated). For rows the rules left with no
    # pattern AND no material despite a long name, ask the LLM once per SKU,
    # passing already-observed values so it aligns to a stable vocabulary
    # (the Vastra buildPrompt reuse pattern). Cached per-code so re-runs are
    # deterministic. Live network path -> pragma: no cover.
    llm_used = 0
    if any_key_present() and cache_dir is not None:
        model = pick_default_model()  # pragma: no cover
        if model is not None:  # pragma: no cover
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            observed_patterns = sorted({r.pattern for r in records if r.pattern})
            observed_materials = sorted({r.material for r in records if r.material})
            for r in records:
                if (
                    r.pattern is None
                    and r.material is None
                    and len(r.product_name) > _LLM_NAME_LONG_THRESHOLD
                ):
                    extra = extract_attributes(
                        model,
                        r.product_name,
                        observed_patterns,
                        observed_materials,
                        cache_path / f"{r.code}.json",
                    )
                    if extra["pattern"] or extra["material"]:
                        r.pattern = extra["pattern"]
                        r.material = extra["material"]
                        r.source = "llm"
                        r.confidence = 0.5
                        llm_used += 1

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
            "rows_with_pattern": regex_pattern_rows,
            "rows_with_material": regex_material_rows,
            "ambiguous_low_confidence_rows": ambiguous_rows,
            "llm_enriched_rows": llm_used,
            "llm_enabled": any_key_present(),
        },
        notes="Pure-rules extraction. LLM enrichment is env-gated and off by "
        "default; ambiguous rows (long name but no pattern/material) are "
        "flagged with confidence=0.6.",
    )
    return records, entry
