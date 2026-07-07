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

<fill in after seeing the numbers>
