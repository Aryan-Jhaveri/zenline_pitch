"""Step 3 — relationship classification (the ONLY judgment step).

Input: output/step2_ontology.json. For each candidate pair (after
blocking on master_category + article_type + gender), classify into
VARIANT / SUBSTITUTE / UNRELATED, score with transparent components,
and optionally escalate borderline scores to an LLM tie-breaker.

Same input -> same output. No randomness. Pairs sorted by (sku_a, sku_b)
for stable diffs.

Note on colour_similarity: Step 2 stores base_colour as a colour family
(e.g. "blue"), so here colour_similarity compares families -> 1.0 when
the families match, 0.0 when they differ, 0.5 when either side is
"other"/unknown. The 0.6 "same family, different exact" tier is a
property of the raw colour_similarity function (still unit-tested) but
is not exercised once colours are pre-normalized to families.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

from substitutes_agent.models import (
    OntologyRecord,
    Relationship,
    RunLogEntry,
    ScoreComponents,
)
from substitutes_agent.step2_ontology import colour_similarity

RelType = str  # "VARIANT" | "SUBSTITUTE"

# Weighted scoring (per spec).
_W_ARTICLE = 0.35
_W_COLOUR = 0.20
_W_USAGE = 0.15
_W_PATTERN = 0.10
_W_MATERIAL = 0.15
_W_SEASON = 0.05

TIE_BREAK_LOW = 0.45
TIE_BREAK_HIGH = 0.55


def _jaccard_single(a: str | None, b: str | None) -> float:
    """Jaccard on single-valued attributes; 0.5 if either is null."""
    if a is None or b is None:
        return 0.5
    sa, sb = {a}, {b}
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.5


def _pattern_match(a: str | None, b: str | None) -> float:
    """0/0.5/1 — null (either side) treated as 0.5."""
    if a is None or b is None:
        return 0.5
    return 1.0 if a == b else 0.0


def _season_overlap(a: str | None, b: str | None) -> float:
    """0.5 if either null, else exact match (1.0/0.0)."""
    if a is None or b is None:
        return 0.5
    return 1.0 if a == b else 0.0


def score_pair(a: OntologyRecord, b: OntologyRecord) -> ScoreComponents:
    """Compute transparent scoring components for a pair."""
    article_type_match = 1.0 if a.article_type == b.article_type else 0.0
    cs = colour_similarity(a.base_colour, b.base_colour)
    usage_match = 1.0 if a.usage == b.usage else 0.0
    pm = _pattern_match(a.pattern, b.pattern)
    ms = _jaccard_single(a.material, b.material)
    so = _season_overlap(a.season, b.season)
    total = (
        _W_ARTICLE * article_type_match
        + _W_COLOUR * cs
        + _W_USAGE * usage_match
        + _W_PATTERN * pm
        + _W_MATERIAL * ms
        + _W_SEASON * so
    )
    return ScoreComponents(
        article_type_match=article_type_match,
        colour_similarity=cs,
        usage_match=usage_match,
        pattern_match=pm,
        material_similarity=ms,
        season_overlap=so,
        total_score=round(total, 4),
    )


def classify_pair(
    a: OntologyRecord, b: OntologyRecord, components: ScoreComponents
) -> RelType | None:
    """Classify a candidate pair. Returns VARIANT/SUBSTITUTE, or None (UNRELATED)."""
    same_brand = a.brand_normalized == b.brand_normalized
    same_article = a.article_type == b.article_type
    same_colour = a.base_colour == b.base_colour
    same_usage = a.usage == b.usage

    # VARIANT: same brand + same article_type + same base_colour + same usage.
    if same_brand and same_article and same_colour and same_usage:
        return "VARIANT"

    # SUBSTITUTE: different brand + same article_type + colour_similarity >= 0.5
    # + same usage + same gender (gender already matches via blocking).
    if (
        not same_brand
        and same_article
        and components.colour_similarity >= 0.5
        and same_usage
    ):
        return "SUBSTITUTE"

    return None


def _llm_keys_present() -> bool:
    import os

    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def _llm_tiebreak(
    a: OntologyRecord, b: OntologyRecord, components: ScoreComponents
) -> tuple[str, str | None]:
    """LLM tie-breaker for borderline scores. Wired in the LLM wrapper commit.

    Returns (decided_by, reason). When no key is set, the rules verdict
    stands (decided_by="rules", reason=None).
    """
    if not _llm_keys_present():
        return "rules", None
    # Actual multi-provider call is wired in commit #13 (llm.classify_pair).
    # For now the rules verdict stands even when a key is present, because
    # the wrapper isn't imported yet. The hook is here so #13 can fill it.
    return "rules", None


def iter_candidate_pairs(
    records: list[OntologyRecord],
) -> list[tuple[OntologyRecord, OntologyRecord, ScoreComponents, str | None]]:
    """Yield every blocked candidate pair with its components and verdict.

    Blocking key is (master_category, article_type, gender); pairs with an
    empty article_type on either side are skipped. verdict is
    "VARIANT"/"SUBSTITUTE" or None (UNRELATED). Exposed so Step 4 can
    compute near-misses from the same scoring path without re-implementing
    blocking.
    """
    groups: dict[tuple[str, str, str], list[OntologyRecord]] = {}
    for r in records:
        if not r.article_type:
            continue
        key = (r.master_category, r.article_type, r.gender)
        groups.setdefault(key, []).append(r)

    pairs: list[tuple[OntologyRecord, OntologyRecord, ScoreComponents, str | None]] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda r: r.code)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a.code > b.code:
                    a, b = b, a
                comp = score_pair(a, b)
                verdict = classify_pair(a, b, comp)
                pairs.append((a, b, comp, verdict))
    return pairs


def classify(
    input_path: str | Path,
    output_path: str | Path | None = None,
    decisions_path: str | Path | None = None,
) -> tuple[list[Relationship], RunLogEntry]:
    """Run Step 3: blocking + classification + scoring."""
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat(timespec="seconds")

    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    records = [OntologyRecord(**d) for d in data]
    input_rows = len(records)

    candidate_pairs_list = iter_candidate_pairs(records)
    candidate_pairs = len(candidate_pairs_list)
    skipped_no_article = input_rows - sum(1 for r in records if r.article_type)

    relationships: list[Relationship] = []
    tiebreak_invocations = 0

    for a, b, components, verdict in candidate_pairs_list:
        if verdict is None:
            continue  # UNRELATED pairs are not emitted.

        decided_by: str = "rules"
        reason: str | None = None
        # LLM tie-breaker only on borderline scores.
        if TIE_BREAK_LOW <= components.total_score <= TIE_BREAK_HIGH:
            decided_by, reason = _llm_tiebreak(a, b, components)
            tiebreak_invocations += 1
            _ = reason

        relationships.append(
            Relationship(
                sku_a=a.code,
                sku_b=b.code,
                relationship=verdict,  # type: ignore[arg-type]
                score=components.total_score,
                components=components,
                decided_by=decided_by,  # type: ignore[arg-type]
                reason=reason,
            )
        )

    relationships.sort(key=lambda r: (r.sku_a, r.sku_b))

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                [r.model_dump() for r in relationships],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    duration = time.monotonic() - started
    variant_n = sum(1 for r in relationships if r.relationship == "VARIANT")
    substitute_n = sum(1 for r in relationships if r.relationship == "SUBSTITUTE")
    entry = RunLogEntry(
        step="step3_classify",
        started_at=started_at,
        duration_s=round(duration, 4),
        input_rows=input_rows,
        output_rows=len(relationships),
        filters={
            "candidate_pairs": candidate_pairs,
            "variant_edges": variant_n,
            "substitute_edges": substitute_n,
            "skipped_no_article_type": skipped_no_article,
            "tiebreak_invocations": tiebreak_invocations,
            "llm_enabled": _llm_keys_present(),
        },
        notes="Blocking on (master_category, article_type, gender). LLM "
        "tie-breaker fires only for total_score in [0.45, 0.55] when a "
        "provider key is present; rules verdict stands otherwise.",
    )

    _ = decisions_path  # wired in commit #13 (step3_llm_decisions.jsonl).
    return relationships, entry
