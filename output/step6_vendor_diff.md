# Vendor-swap diff (majority-vote per model)

## Headline stats
- Total pairs: 30
- All vendors agree: 29
- Vendor splits: 1
- Vendor-split pairs: 8200|8393

## Per-vendor yes-count (majority voting)
| Vendor | Yes-count |
| --- | ---: |
| anthropic/claude-haiku-4.5 | 2 |
| gemini-2.5-flash | 2 |
| openai/gpt-4o-mini | 3 |

## Vendor-split pairs
| Pair | anthropic/claude-haiku-4.5 | gemini-2.5-flash | openai/gpt-4o-mini |
| --- | --- | --- | --- |
| 8200|8393 | no | no | yes |

## Reading

The majority-vote vendor-swap flip rate on this sample is 1 of 30 pairs (3.3%). The raw run-level cross-model disagreement rate is higher because it counts within-model noise as disagreement; taking the modal verdict per model collapses most of that noise. The two numbers differ because run-level disagreement includes pairs where one model flapped on its own three runs while the other two vendors stayed consistent. See output/step5_consistency.md for the run-level figure.
