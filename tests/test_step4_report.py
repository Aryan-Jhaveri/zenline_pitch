"""Step 4 report tests: gap math, schema, markdown rendering."""

from __future__ import annotations

import json

from substitutes_agent.models import OntologyRecord, Relationship, ScoreComponents
from substitutes_agent.step4_report import build_report


def _rec(
    code: str,
    brand: str,
    colour: str = "blue",
    usage: str = "Casual",
    gender: str = "Men",
    article: str = "t-shirt",
    master: str = "Apparel",
) -> OntologyRecord:
    return OntologyRecord(
        code=code,
        brand_normalized=brand,
        product_name="x",
        article_type=article,
        master_category=master,
        base_colour=colour,
        usage=usage,
        gender=gender,
        season="Summer",
    )


def _rel(a: str, b: str, kind: str, score: float = 0.8) -> Relationship:
    return Relationship(
        sku_a=a,
        sku_b=b,
        relationship=kind,  # type: ignore[arg-type]
        score=score,
        components=ScoreComponents(
            article_type_match=1.0,
            colour_similarity=1.0,
            usage_match=1.0,
            pattern_match=0.5,
            material_similarity=0.5,
            season_overlap=1.0,
            total_score=score,
        ),
    )


def _write_inputs(
    tmp_path,  # type: ignore[no-untyped-def]
    records: list[OntologyRecord],
    rels: list[Relationship],
) -> tuple[str, str]:
    ont = tmp_path / "s2.json"
    ont.write_text(
        json.dumps([r.model_dump() for r in records], ensure_ascii=False),
        encoding="utf-8",
    )
    rel = tmp_path / "s3.json"
    rel.write_text(
        json.dumps([r.model_dump() for r in rels], ensure_ascii=False),
        encoding="utf-8",
    )
    return str(ont), str(rel)


def test_gap_math(tmp_path) -> None:  # type: ignore[no-untyped-def]
    records = [
        _rec("1", "x"),
        _rec("2", "x"),
        _rec("3", "y"),
        _rec("4", "z", colour="red"),
        _rec("5", "w", usage="Sports"),
    ]
    rels = [
        _rel("1", "2", "VARIANT"),
        _rel("1", "3", "SUBSTITUTE"),
        _rel("2", "3", "SUBSTITUTE"),
    ]
    ont, rel = _write_inputs(tmp_path, records, rels)
    rep, _ = build_report(ont, rel)
    assert rep["total_skus"] == 5
    # SKUs 1, 2, 3 have substitute edges; 4 and 5 do not -> 2 gap of 5.
    assert rep["substitution_gap_count"] == 2
    assert rep["substitution_gap_pct"] == 40.0
    assert rep["substitute_edges"] == 2
    assert rep["variant_edges"] == 1


def test_report_schema(tmp_path) -> None:  # type: ignore[no-untyped-def]
    records = [_rec("1", "x"), _rec("2", "y")]
    rels = [_rel("1", "2", "SUBSTITUTE")]
    ont, rel = _write_inputs(tmp_path, records, rels)
    rep, _ = build_report(ont, rel)
    for key in (
        "total_skus",
        "substitution_gap_count",
        "substitution_gap_pct",
        "substitute_edges",
        "variant_edges",
        "top_brands_by_sku",
        "top_gap_skus_by_near_misses",
        "wall_clock_s",
    ):
        assert key in rep


def test_top_brands_structure(tmp_path) -> None:  # type: ignore[no-untyped-def]
    records = [_rec("1", "x"), _rec("2", "x"), _rec("3", "y")]
    rels = [_rel("1", "3", "SUBSTITUTE")]
    ont, rel = _write_inputs(tmp_path, records, rels)
    rep, _ = build_report(ont, rel)
    brands = rep["top_brands_by_sku"]
    assert isinstance(brands, list)
    assert brands[0]["brand"] == "x"
    assert brands[0]["sku_count"] == 2
    # brand x has SKUs 1,2; SKU 1 has a substitute, SKU 2 does not -> 1 gap.
    assert brands[0]["gap_skus"] == 1


def test_near_misses_for_gap_skus(tmp_path) -> None:  # type: ignore[no-untyped-def]
    records = [
        _rec("1", "x"),
        _rec("2", "x"),
        _rec("3", "y"),
        _rec("4", "z", colour="red"),
        _rec("5", "w", usage="Sports"),
    ]
    rels = [
        _rel("1", "2", "VARIANT"),
        _rel("1", "3", "SUBSTITUTE"),
        _rel("2", "3", "SUBSTITUTE"),
    ]
    ont, rel = _write_inputs(tmp_path, records, rels)
    rep, _ = build_report(ont, rel)
    gap_skus = {row["sku"]: row for row in rep["top_gap_skus_by_near_misses"]}
    # SKUs 4 and 5 are gap SKUs; each has several UNRELATED candidate pairs.
    assert "4" in gap_skus
    assert "5" in gap_skus
    assert gap_skus["4"]["near_misses"] > 0
    assert gap_skus["4"]["best_score"] >= 0.0


def test_markdown_renders(tmp_path) -> None:  # type: ignore[no-untyped-def]
    records = [_rec("1", "x"), _rec("2", "y")]
    rels = [_rel("1", "2", "SUBSTITUTE")]
    ont, rel = _write_inputs(tmp_path, records, rels)
    jout = tmp_path / "s4.json"
    mout = tmp_path / "s4.md"
    build_report(ont, rel, None, jout, mout)
    md = mout.read_text()
    assert "# Substitution Gap Report" in md
    assert "| Brand | SKU count |" in md
    assert "| SKU | Near-misses |" in md
    assert "<" not in md  # no raw HTML
    assert jout.exists()
    data = json.loads(jout.read_text())
    assert data["total_skus"] == 2


def test_wall_clock_from_run_log(tmp_path) -> None:  # type: ignore[no-untyped-def]
    records = [_rec("1", "x"), _rec("2", "y")]
    rels = [_rel("1", "2", "SUBSTITUTE")]
    ont, rel = _write_inputs(tmp_path, records, rels)
    run_log = tmp_path / "run_log.json"
    run_log.write_text(
        json.dumps(
            [
                {"step": "step1_ingest", "duration_s": 0.1},
                {"step": "step2_ontology", "duration_s": 0.2},
            ]
        ),
        encoding="utf-8",
    )
    rep, _ = build_report(ont, rel, run_log)
    assert rep["wall_clock_s"]["step1_ingest"] == 0.1
    assert rep["wall_clock_s"]["step2_ontology"] == 0.2


def test_zero_skus_no_divide_by_zero(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ont, rel = _write_inputs(tmp_path, [], [])
    rep, _ = build_report(ont, rel)
    assert rep["total_skus"] == 0
    assert rep["substitution_gap_pct"] == 0.0
