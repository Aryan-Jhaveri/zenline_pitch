# Consistency Experiment

Zenline's Q2 2026 update argues that raw autonomous agents give inconsistent answers to the same question. This step tests that claim on this project's own ambiguous pairs.

## Within-model consistency

| Model | Self-agreement % | Pairs |
| --- | ---: | ---: |
| anthropic/claude-haiku-4.5 | 96.67 | 30 |
| gemini-2.5-flash | 100.0 | 30 |
| openai/gpt-4o-mini | 86.67 | 30 |

## Cross-model agreement

- Pairs where ALL models across ALL runs agreed: **80.0%**
- Rules-based baseline (by construction): **100%**

## Interpretation

Not sure what to make of the within-model consistency. Gemini agrees with itself on every pair, Claude on 29 of 30, but the interesting number is cross-model agreement at 80%. For a retailer that reframes the failure mode: it's not "ask the same LLM twice, get two answers" (less of a problem at the small tier than the rhetoric suggests), it's "swap vendors and watch the substitute map shift" on some fraction of borderline SKUs.

How big that fraction is depends on how you deploy. One query per pair with no retry and 20% of borderline pairs have at least one disagreeing verdict across the 9 runs. Majority vote across three runs per model, which is what any rational production system would do, and only 1 pair in 30 (3.3%) actually flips. That pair is 8200 (Locomotive Men Solid Lander Black T-Shirts) vs 8393 (ADIDAS Men White T-shirt): GPT-4o-mini says yes on all three runs, Gemini and Claude both say no on all three. Every other cross-model disagreement is within-model noise on a single model, most often GPT-4o-mini.

20% is the ceiling under naive deployment, 3.3% the floor under sensible retry. Either way the rules pipeline handles the deterministic majority before an LLM sees it, so drift only ever applies to the borderline residual. A vendor change becomes a versioned diff on a small auditable subset, not a silent shift across the whole map.

Per-pair audit trail: output/step6_audit_trail.md. Vendor-swap diff: output/step6_vendor_diff.md.
