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


The rules-first pipeline keeps LLMs off the deterministic majority of decisions and logs every residual judgment per model, so a vendor change becomes a versioned diff instead of silent drift. GPT-4o-mini's 86.67% self-agreement — four pairs where its three runs didn't all agree — is a secondary datum: even without vendor-swapping, picking the wrong small model gives a retailer flapping decisions on roughly one in seven borderline calls.
