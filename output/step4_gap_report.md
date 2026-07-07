# Substitution Gap Report

Numbers below are actual results from this run on the committed
sample fixture. They are not styled to match any published metric.

## Summary

- Total SKUs analyzed: **500**
- Substitution gap: **129** SKUs (25.8%) with zero SUBSTITUTE edges
- SUBSTITUTE edges proposed: **947**
- VARIANT edges proposed: **109**

## Top-10 brands by SKU count

| Brand | SKU count | Gap SKUs | Gap % |
| --- | ---: | ---: | ---: |
| adidas | 39 | 11 | 28.21 |
| nike | 34 | 13 | 38.24 |
| united colors of | 17 | 4 | 23.53 |
| catwalk | 15 | 5 | 33.33 |
| puma | 15 | 0 | 0.0 |
| red | 14 | 4 | 28.57 |
| wrangler | 11 | 0 | 0.0 |
| gini and jony | 10 | 5 | 50.0 |
| fabindia | 9 | 7 | 77.78 |
| locomotive | 9 | 0 | 0.0 |

## Top-10 gap SKUs by near-miss count

Near-misses = candidate pairs (same blocking group) that classified
as UNRELATED. Best score is the highest total_score among them.

| SKU | Near-misses | Best score |
| --- | ---: | ---: |
| 13088 | 81 | 0.725 |
| 2058 | 81 | 0.675 |
| 28856 | 81 | 0.675 |
| 31088 | 52 | 0.775 |
| 53152 | 52 | 0.725 |
| 57495 | 36 | 0.725 |
| 57498 | 36 | 0.725 |
| 7777 | 36 | 0.675 |
| 19645 | 29 | 0.675 |
| 30948 | 29 | 0.725 |

## Wall-clock per step (seconds)

| Step | Duration (s) |
| --- | ---: |
| step1_ingest | 0.123 |
| step2_ontology | 0.0144 |
| step3_classify | 0.0382 |
