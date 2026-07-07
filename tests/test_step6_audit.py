"""Step 6 audit + vendor-swap diff tests. No network, no LLM calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from substitutes_agent.models import OntologyRecord, ScoreComponents
from substitutes_agent.step6_audit import (
    compute_per_model_majority,
    compute_vendor_diff,
    load_transcripts,
    majority_vote,
    render_audit_trail_md,
    render_vendor_diff_md,
    run_step6,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# majority_vote
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "verdicts,expected",
    [
        (["yes", "yes", "yes"], "yes"),
        (["no", "no", "no"], "no"),
        (["yes", "no", "no"], "no"),
        (["yes", "yes", "no"], "yes"),
    ],
)
def test_majority_vote_unanimous_and_split(verdicts: list[str], expected: str) -> None:
    assert majority_vote(verdicts) == expected


def test_majority_vote_tied_returns_no() -> None:
    assert majority_vote(["yes", "no"]) == "no"


def test_majority_vote_empty_raises() -> None:
    with pytest.raises(ValueError):
        majority_vote([])


# ---------------------------------------------------------------------------
# compute_per_model_majority
# ---------------------------------------------------------------------------


def test_compute_per_model_majority_shape() -> None:
    verdicts = {
        "1|2": {
            "m1": ["yes", "yes", "yes"],
            "m2": ["no", "no", "no"],
        }
    }
    out = compute_per_model_majority(verdicts)
    assert set(out.keys()) == {"1|2"}
    assert set(out["1|2"].keys()) == {"m1", "m2"}
    assert out["1|2"]["m1"] == "yes"
    assert out["1|2"]["m2"] == "no"


# ---------------------------------------------------------------------------
# compute_vendor_diff
# ---------------------------------------------------------------------------


def test_compute_vendor_diff_no_splits() -> None:
    majorities = {
        "1|2": {"m1": "yes", "m2": "yes", "m3": "yes"},
        "3|4": {"m1": "no", "m2": "no", "m3": "no"},
        "5|6": {"m1": "yes", "m2": "yes", "m3": "yes"},
    }
    out = compute_vendor_diff(majorities)
    assert out["total_pairs"] == 3
    assert out["all_vendors_agree_count"] == 3
    assert out["vendor_split_count"] == 0
    assert out["vendor_split_pairs"] == []
    assert out["per_vendor_yes_count"] == {"m1": 2, "m2": 2, "m3": 2}


def test_compute_vendor_diff_one_split() -> None:
    majorities = {
        "1|2": {"m1": "yes", "m2": "yes", "m3": "yes"},
        "3|4": {"m1": "no", "m2": "no", "m3": "yes"},
        "5|6": {"m1": "yes", "m2": "yes", "m3": "yes"},
    }
    out = compute_vendor_diff(majorities)
    assert out["total_pairs"] == 3
    assert out["all_vendors_agree_count"] == 2
    assert out["vendor_split_count"] == 1
    assert out["vendor_split_pairs"] == ["3|4"]
    assert out["per_vendor_yes_count"] == {"m1": 2, "m2": 2, "m3": 3}


def test_compute_vendor_diff_matches_real_run() -> None:
    """Hard verification against committed step 5 transcripts."""
    transcripts = REPO_ROOT / "output" / "step5_transcripts.jsonl"
    if not transcripts.exists():
        pytest.skip("real-run step5 transcripts not present")
    verdicts = load_transcripts(transcripts)
    majorities = compute_per_model_majority(verdicts)
    diff = compute_vendor_diff(majorities)
    assert diff["vendor_split_count"] == 1
    assert diff["vendor_split_pairs"] == ["8200|8393"]


# ---------------------------------------------------------------------------
# render_audit_trail_md / render_vendor_diff_md: no em-dashes
# ---------------------------------------------------------------------------


def _comp() -> ScoreComponents:
    return ScoreComponents(
        article_type_match=1.0,
        colour_similarity=0.0,
        usage_match=0.0,
        pattern_match=0.5,
        material_similarity=0.5,
        season_overlap=0.0,
        total_score=0.475,
    )


def _rec(code: str, name: str = "Widget") -> OntologyRecord:
    return OntologyRecord(
        code=code,
        brand_normalized="acme",
        product_name=name,
        article_type="t-shirt",
        master_category="Apparel",
        base_colour="black",
        usage="Casual",
        gender="Men",
        season="Summer",
        pattern="Solid",
        material=None,
    )


def test_render_audit_trail_md_no_emdash() -> None:
    borderline = [("8200", "8393", _comp(), 0.475)]
    ontology = {"8200": _rec("8200", "A shirt"), "8393": _rec("8393", "B shirt")}
    verdicts = {
        "8200|8393": {
            "gemini-2.5-flash": ["no", "no", "no"],
            "anthropic/claude-haiku-4.5": ["no", "no", "no"],
            "openai/gpt-4o-mini": ["yes", "yes", "yes"],
        }
    }
    md = render_audit_trail_md(borderline, ontology, verdicts)
    assert "\u2014" not in md
    assert "## Pair 8200|8393" in md
    assert "article_type_match" in md
    assert "[cross-model split]" in md


def test_render_vendor_diff_md_no_emdash() -> None:
    majorities = {
        "8200|8393": {
            "anthropic/claude-haiku-4.5": "no",
            "gemini-2.5-flash": "no",
            "openai/gpt-4o-mini": "yes",
        }
    }
    diff_stats = compute_vendor_diff(majorities)
    md = render_vendor_diff_md(diff_stats, majorities)
    assert "\u2014" not in md
    assert "vendor split" in md.lower()
    assert "8200|8393" in md


# ---------------------------------------------------------------------------
# run_step6 end-to-end on real artifacts
# ---------------------------------------------------------------------------


def test_run_step6_end_to_end_on_real_artifacts(tmp_path: Path) -> None:
    ont = REPO_ROOT / "output" / "step2_ontology.json"
    transcripts = REPO_ROOT / "output" / "step5_transcripts.jsonl"
    consistency = REPO_ROOT / "output" / "step5_consistency.json"
    if not all(p.exists() for p in (ont, transcripts, consistency)):
        pytest.skip("real-run step 5 artifacts not present")
    audit_md = tmp_path / "step6_audit_trail.md"
    vdiff_md = tmp_path / "step6_vendor_diff.md"
    vdiff_json = tmp_path / "step6_vendor_diff.json"
    entry = run_step6(
        ontology_path=ont,
        transcripts_path=transcripts,
        consistency_json_path=consistency,
        audit_md_out=audit_md,
        vendor_diff_md_out=vdiff_md,
        vendor_diff_json_out=vdiff_json,
    )
    assert entry.step == "step6_audit"
    assert entry.output_rows == 30
    assert audit_md.exists() and audit_md.stat().st_size > 0
    assert vdiff_md.exists() and vdiff_md.stat().st_size > 0
    assert vdiff_json.exists() and vdiff_json.stat().st_size > 0
    stats = json.loads(vdiff_json.read_text(encoding="utf-8"))
    assert stats["total_pairs"] == 30
    assert stats["vendor_split_count"] == 1
    audit_text = audit_md.read_text(encoding="utf-8")
    vdiff_text = vdiff_md.read_text(encoding="utf-8")
    assert "vendor split" in audit_text.lower() or "vendor split" in vdiff_text.lower()
    # No em-dashes in either markdown.
    assert "\u2014" not in audit_text
    assert "\u2014" not in vdiff_text
