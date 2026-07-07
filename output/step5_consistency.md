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

Not sure what to make of the within-model Consistency but here we see that gemini agrees with itself on every pair, Claude on 29 of 30, but the real gap is cross-model agreement (80%).Roughly one in five borderline calls flips when the vendor changes, even though each model largely agrees with itself.

For a retailer, it couldn't be "ask the same LLM twice, get two answers" (less of a problem at the small tier than the rhetoric suggests), but instead the real issue is "swap vendors and watch the substitute map shift on 20% of borderline SKUs." 


The rules pipeline keeps LLMs off the deterministic majority of decisions, so a vendor change becomes a versioned diff instead of silent drift. GPT-4o-mini's 86.67% self-agreement is a secondary datum: even without vendor-swapping, picking the wrong small model gives flapping decisions on roughly one in seven borderline calls.

Looking at run-level detail, only one of the six cross-model disagreements is a clean vendor split (pair 8200|8393, where GPT-4o-mini says yes on all three runs while both Gemini and Claude say no on all three). The other five are within-model noise on a single model, most often GPT-4o-mini. Under majority voting across three runs per model, the real vendor-swap flip rate on this sample is 1 pair in 30, roughly 3.3 percent. That's the number a retailer's ops team would actually experience if they deployed any of these models with the sensible retry pattern. Small, but nonzero, and it accumulates across catalog versions. See output/step6_audit_trail.md for the full per-pair audit and output/step6_vendor_diff.md for the vendor comparison.
