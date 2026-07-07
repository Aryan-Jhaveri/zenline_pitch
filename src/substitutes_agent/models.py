"""Pydantic dataclasses for pipeline artifacts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Format = Literal["cream", "gel", "serum", "lotion", "balm", "oil", "mask", "unknown"]
TargetArea = Literal["face", "eye", "lip", "body", "unknown"]
SkinType = Literal["dry", "oily", "combination", "sensitive", "mature", "normal"]
Active = Literal[
    "retinol",
    "retinal",
    "niacinamide",
    "hyaluronic acid",
    "vitamin c",
    "salicylic acid",
    "glycolic acid",
    "lactic acid",
    "ceramides",
    "peptides",
    "spf",
]


class OntologyRecord(BaseModel):
    """One SKU's structured attributes (Step 2 output)."""

    code: str
    brand_normalized: str
    product_name: str
    format: Format
    skin_type_claims: list[SkinType] = Field(default_factory=list)
    key_actives: list[str] = Field(default_factory=list)
    pack_size_ml: float | None = None
    target_area: TargetArea = "unknown"
    source: Literal["rules", "llm"] = "rules"
    confidence: float = 1.0


class ScoreComponents(BaseModel):
    active_overlap: float
    skin_type_overlap: float
    size_similarity: float
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


class RunLogEntry(BaseModel):
    """One step's audit log entry."""

    step: str
    started_at: str
    duration_s: float
    input_rows: int | None = None
    output_rows: int | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None
