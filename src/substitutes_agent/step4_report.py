"""Step 4 — gap report (deterministic).

Computes the substitution gap from Step 2 ontology + Step 3 relationships
and emits both a JSON and a Markdown report. Numbers are actual results
from this run, not styled to match any published metric.

Near-misses: a gap SKU's near-misses are its candidate pairs (from Step 3
blocking) that classified as UNRELATED. Reusing Step 3's
iter_candidate_pairs keeps the near-miss scoring on the exact same path
as the emitted edges — no parallel logic.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from substitutes_agent.models import OntologyRecord, Relationship, RunLogEntry
from substitutes_agent.step3_classify import iter_candidate_pairs


def build_report(
    ontology_path: str | Path,
    relationships_path: str | Path,
    run_log_path: str | Path | None = None,
    output_json: str | Path | None = None,
    output_md: str | Path | None = None,
) -> tuple[dict[str, object], RunLogEntry]:
    """Run Step 4: compute gap stats and emit JSON + Markdown."""
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat(timespec="seconds")

    records = [
        OntologyRecord(**d)
        for d in json.loads(Path(ontology_path).read_text(encoding="utf-8"))
    ]
    relationships = [
        Relationship(**d)
        for d in json.loads(Path(relationships_path).read_text(encoding="utf-8"))
    ]

    total_skus = len(records)
    substitute_edges = [r for r in relationships if r.relationship == "SUBSTITUTE"]
    variant_edges = [r for r in relationships if r.relationship == "VARIANT"]

    skus_with_substitute = {r.sku_a for r in substitute_edges} | {
        r.sku_b for r in substitute_edges
    }
    gap_count = total_skus - len(skus_with_substitute)
    gap_pct = round(100.0 * gap_count / total_skus, 2) if total_skus else 0.0

    # Top-10 brands by SKU count with gap %.
    brand_skus: dict[str, int] = {}
    brand_gap_skus: dict[str, int] = {}
    for r in records:
        b = r.brand_normalized or "(unknown)"
        brand_skus[b] = brand_skus.get(b, 0) + 1
        if r.code not in skus_with_substitute:
            brand_gap_skus[b] = brand_gap_skus.get(b, 0) + 1
    top_brands = sorted(brand_skus.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    top_brands_rows = [
        {
            "brand": b,
            "sku_count": n,
            "gap_skus": brand_gap_skus.get(b, 0),
            "gap_pct": round(100.0 * brand_gap_skus.get(b, 0) / n, 2) if n else 0.0,
        }
        for b, n in top_brands
    ]

    # Top-10 gap SKUs by near-miss count.
    all_pairs = iter_candidate_pairs(records)
    near_misses: dict[str, list[float]] = {}
    for ra, rb, comp, verdict in all_pairs:
        if verdict is not None:
            continue  # only UNRELATED pairs are near-misses
        for code in (ra.code, rb.code):
            if code in skus_with_substitute:
                continue  # only gap SKUs
            near_misses.setdefault(code, []).append(comp.total_score)
    top_gap_skus = sorted(near_misses.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:10]
    top_gap_rows = [
        {
            "sku": code,
            "near_misses": len(scores),
            "best_score": round(max(scores), 4) if scores else 0.0,
        }
        for code, scores in top_gap_skus
    ]

    # Wall-clock breakdown per step from run_log.json.
    wall_clock: dict[str, float] = {}
    if run_log_path is not None:
        rp = Path(run_log_path)
        if rp.exists():
            log_entries = json.loads(rp.read_text(encoding="utf-8"))
            for e in log_entries:
                wall_clock[e.get("step", "")] = e.get("duration_s", 0.0)

    report: dict[str, Any] = {
        "total_skus": total_skus,
        "substitution_gap_count": gap_count,
        "substitution_gap_pct": gap_pct,
        "substitute_edges": len(substitute_edges),
        "variant_edges": len(variant_edges),
        "top_brands_by_sku": top_brands_rows,
        "top_gap_skus_by_near_misses": top_gap_rows,
        "wall_clock_s": wall_clock,
    }

    duration = time.monotonic() - started
    entry = RunLogEntry(
        step="step4_report",
        started_at=started_at,
        duration_s=round(duration, 4),
        input_rows=total_skus,
        output_rows=len(relationships),
        filters={
            "gap_count": gap_count,
            "gap_pct": gap_pct,
            "near_miss_skus": len(near_misses),
        },
        notes="Numbers are actual results from this run.",
    )

    if output_json is not None:
        jp = Path(output_json)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if output_md is not None:
        mp = Path(output_md)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(_render_markdown(report), encoding="utf-8")

    return report, entry


def _render_markdown(report: dict[str, Any]) -> str:
    """Render the gap report as GitHub-friendly Markdown (no raw HTML)."""
    lines: list[str] = []
    lines.append("# Substitution Gap Report")
    lines.append("")
    lines.append("Numbers below are actual results from this run on the committed")
    lines.append("sample fixture. They are not styled to match any published metric.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total SKUs analyzed: **{report['total_skus']}**")
    lines.append(
        f"- Substitution gap: **{report['substitution_gap_count']}** "
        f"SKUs ({report['substitution_gap_pct']}%) with zero SUBSTITUTE edges"
    )
    lines.append(f"- SUBSTITUTE edges proposed: **{report['substitute_edges']}**")
    lines.append(f"- VARIANT edges proposed: **{report['variant_edges']}**")
    lines.append("")
    lines.append("## Top-10 brands by SKU count")
    lines.append("")
    lines.append("| Brand | SKU count | Gap SKUs | Gap % |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in report["top_brands_by_sku"]:
        lines.append(
            f"| {row['brand']} | {row['sku_count']} | "
            f"{row['gap_skus']} | {row['gap_pct']} |"
        )
    lines.append("")
    lines.append("## Top-10 gap SKUs by near-miss count")
    lines.append("")
    lines.append("Near-misses = candidate pairs (same blocking group) that classified")
    lines.append("as UNRELATED. Best score is the highest total_score among them.")
    lines.append("")
    lines.append("| SKU | Near-misses | Best score |")
    lines.append("| --- | ---: | ---: |")
    for row in report["top_gap_skus_by_near_misses"]:
        lines.append(f"| {row['sku']} | {row['near_misses']} | {row['best_score']} |")
    lines.append("")
    lines.append("## Wall-clock per step (seconds)")
    lines.append("")
    lines.append("| Step | Duration (s) |")
    lines.append("| --- | ---: |")
    for step, dur in report["wall_clock_s"].items():
        lines.append(f"| {step} | {dur} |")
    lines.append("")
    return "\n".join(lines)
