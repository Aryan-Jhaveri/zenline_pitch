"""Step 5 consistency tests: arithmetic on synthetic verdicts, skip path."""

from __future__ import annotations

from substitutes_agent.models import OntologyRecord
from substitutes_agent.step3_classify import iter_candidate_pairs
from substitutes_agent.step5_consistency import (
    MAX_PAIRS,
    compute_consistency,
    run_consistency,
)

# ---------------------------------------------------------------------------
# compute_consistency: pure arithmetic on synthetic verdicts
# ---------------------------------------------------------------------------


def test_consistency_empty() -> None:
    out = compute_consistency({})
    assert out["pairs"] == 0
    assert out["cross_model_agreement_pct"] == 0.0
    assert out["rules_baseline_pct"] == 100.0


def test_consistency_all_runs_agree_per_model() -> None:
    # 3/3 agreement on both models for both pairs.
    verdicts = {
        "1|2": {"haiku": ["yes", "yes", "yes"], "sonnet": ["no", "no", "no"]},
        "3|4": {"haiku": ["yes", "yes", "yes"], "sonnet": ["no", "no", "no"]},
    }
    out = compute_consistency(verdicts)
    assert out["per_model_self_agreement"]["haiku"]["self_agreement_pct"] == 100.0
    assert out["per_model_self_agreement"]["sonnet"]["self_agreement_pct"] == 100.0
    # Cross-model: haiku always yes, sonnet always no -> never all agree -> 0%.
    assert out["cross_model_agreement_pct"] == 0.0


def test_consistency_two_of_three_disagrees() -> None:
    # 2/3 same is NOT full self-agreement (need all 3 identical).
    verdicts = {
        "1|2": {"haiku": ["yes", "no", "yes"]},  # 2 yes, 1 no -> not agreed
    }
    out = compute_consistency(verdicts)
    assert out["per_model_self_agreement"]["haiku"]["self_agreement_pct"] == 0.0


def test_consistency_partial_self_agreement() -> None:
    verdicts = {
        "1|2": {"haiku": ["yes", "yes", "yes"]},  # agreed
        "3|4": {"haiku": ["yes", "no", "yes"]},  # not agreed
    }
    out = compute_consistency(verdicts)
    # 1 of 2 pairs agreed -> 50%.
    assert out["per_model_self_agreement"]["haiku"]["self_agreement_pct"] == 50.0


def test_consistency_cross_model_full_agreement() -> None:
    verdicts = {
        "1|2": {"haiku": ["yes", "yes", "yes"], "sonnet": ["yes", "yes", "yes"]},
        "3|4": {"haiku": ["no", "no", "no"], "sonnet": ["no", "no", "no"]},
    }
    out = compute_consistency(verdicts)
    assert out["cross_model_agreement_pct"] == 100.0


def test_consistency_rules_baseline_always_100() -> None:
    out = compute_consistency({"1|2": {"haiku": ["yes", "no", "yes"]}})
    assert out["rules_baseline_pct"] == 100.0


# ---------------------------------------------------------------------------
# _select_pairs cap (via iter_candidate_pairs on a constructed ontology)
# ---------------------------------------------------------------------------


def _rec(
    code: str, brand: str = "b", colour: str = "blue", usage: str = "Casual"
) -> OntologyRecord:
    return OntologyRecord(
        code=code,
        brand_normalized=brand,
        product_name="x",
        article_type="t-shirt",
        master_category="Apparel",
        base_colour=colour,
        usage=usage,
        gender="Men",
        season="Summer",
    )


def test_max_pairs_cap_is_30() -> None:
    assert MAX_PAIRS == 30


def test_candidate_pairs_deterministic() -> None:
    records = [_rec("1"), _rec("2"), _rec("3", colour="red", usage="Sports")]
    pairs = iter_candidate_pairs(records)
    # 3 SKUs in one blocking group -> 3 candidate pairs, deterministic.
    assert len(pairs) == 3
    again = iter_candidate_pairs(records)
    assert [(a.code, b.code) for a, b, _, _ in pairs] == [
        (a.code, b.code) for a, b, _, _ in again
    ]


# ---------------------------------------------------------------------------
# run_consistency skip + dry-run paths (no live network)
# ---------------------------------------------------------------------------


def _write_ontology(tmp_path, records: list[OntologyRecord]) -> str:  # type: ignore[no-untyped-def]
    import json

    p = tmp_path / "s2.json"
    p.write_text(
        json.dumps([r.model_dump() for r in records], ensure_ascii=False),
        encoding="utf-8",
    )
    return str(p)


def test_run_consistency_skips_without_keys(
    tmp_path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    ont = _write_ontology(tmp_path, [_rec("1"), _rec("2")])
    rep, log = run_consistency(ont)
    assert rep["skipped"] is True
    assert "no LLM provider keys" in rep["reason"]
    assert log.notes.startswith("Skipped")


def test_run_consistency_dry_run_with_fake_keys(
    tmp_path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    # Build enough SKUs to form at least one candidate pair.
    records = [_rec(str(i)) for i in range(40)]
    ont = _write_ontology(tmp_path, records)
    rep, log = run_consistency(ont, dry_run=True)
    assert rep["skipped"] is True
    assert rep["reason"].startswith("dry-run")
    assert rep["estimated_calls"] >= 0
    assert log.notes == "Dry run; no calls made."


def test_run_consistency_writes_skipped_markdown(
    tmp_path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    ont = _write_ontology(tmp_path, [_rec("1"), _rec("2")])
    jout = tmp_path / "s5.json"
    mout = tmp_path / "s5.md"
    run_consistency(ont, jout, mout)
    assert jout.exists()
    assert mout.exists()
    md = mout.read_text()
    assert "# Consistency Experiment" in md
    assert "Skipped" in md
