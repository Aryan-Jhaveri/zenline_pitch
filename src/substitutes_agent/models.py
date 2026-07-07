"""Pydantic dataclasses for pipeline artifacts (apparel schema)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Source = Literal["rules", "llm"]


class OntologyRecord(BaseModel):
    """One SKU's structured attributes (Step 2 output)."""

    code: str
    brand_normalized: str
    product_name: str
    article_type: str
    master_category: str
    base_colour: str
    usage: str
    gender: str
    season: str | None = None
    pattern: str | None = None
    material: str | None = None
    source: Source = "rules"
    confidence: float = 1.0


class ScoreComponents(BaseModel):
    """Transparent scoring components for one classified pair."""

    article_type_match: float
    colour_similarity: float
    usage_match: float
    pattern_match: float
    material_similarity: float
    season_overlap: float
    total_score: float


class Relationship(BaseModel):
    """One classified pair (Step 3 output)."""

    sku_a: str
    sku_b: str
    relationship: Literal["VARIANT", "SUBSTITUTE"]
    score: float
    components: ScoreComponents
    decided_by: Literal["rules", "llm"] = "rules"
    reason: str | None = None


class PairVerdict(BaseModel):
    """An LLM's verdict on whether two SKUs are realistic substitutes."""

    verdict: Literal["yes", "no"]
    reason: str
    model: str


class RunLogEntry(BaseModel):
    """One step's audit log entry."""

    step: str
    started_at: str
    duration_s: float
    input_rows: int | None = None
    output_rows: int | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None
