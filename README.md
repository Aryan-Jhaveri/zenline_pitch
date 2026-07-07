# Substitutes Agent Demo

## What this is

A small implementation of retail substitute mapping, built as an **agentic workflow** on open apparel data. Runs end-to-end on a 500-row slice of the paramaggarwal fashion-product-images dataset (Kaggle, MIT) and produces a substitution gap report plus a small experiment measuring LLM consistency on borderline pairs. The taxonomy and prompt shape are adapted from my own prior work on [Vastra](https://vastra.cc), an AI wardrobe app. Zenline AI's public writing on agentic assortment intelligence is the reference frame, and their case studies on face creams and dairy are the pattern being mirrored one category over.

## Why an agentic workflow (not an autonomous agent)

Zenline has argued publicly that fully autonomous
agents are inconsistent for high-stakes retail decisions; the value is in
**agentic workflows** — a fixed, visible, verifiable pipeline where an LLM
does judgment work only at one or two clearly defined points and every
other step is deterministic and auditable.  (Which is strongly strongly agree)


This demo highlights that position: Steps 1, 2 (rules path), and 4 are pure code, and the LLM only
ever fires in Step 2 (filling ambiguous attributes) and Step 3
(tie-breaking borderline pairs), and only when an API key is set. Same
input always produces the same output.

This design is the pre-SOTA baseline for a substitute-mapping pipeline. The 2026 literature on cost-efficient entity matching (Ma et al., *CE-RAG4EM*) converges on the same shape: classical blocking first, then rules, then an LLM only as a verifier on the residual. Amazon's *CatalogRAG* (Zhang et al., 2025) makes the same argument specifically for attribute completion: seed the LLM prompt with well-filled examples from the same catalog, don't rely on the model's internal knowledge. Both patterns are already in this pipeline (Step 3's blocking rule, Step 2's "reuse observed values" prompt). I built them from first principles before finding the papers, which was reassuring. Where this pipeline consciously departs from current SOTA is discussed in "Where this fits" below.

I read Zenline's public positioning as arguing for a **product ontology** — a
stable, retailer-specific vocabulary of attributes that AI reasoning is
anchored to. Step 2's structured attribute layer (article type, colour family, usage,
pattern, material) is that idea.

## How this reuses my prior work

[Vastra](https://vastra.cc) is an AI wardrobe app I built that does
photo-based garment categorization. Its categorizer is anchored to a
small, stable vocabulary of article types, and its prompt passes the set
of subcategories already present in the wardrobe so the model reuses
existing spellings before inventing new ones.

I lifted both ideas here:
the 8-category / ~50-subcategory taxonomy lives in
`src/substitutes_agent/vastra_taxonomy.py`, and the Step 2 LLM prompt
passes already-observed pattern/material values so the model aligns to a
stable vocabulary rather than emitting free-form labels. Same intuition,
retail-catalog scope.

## The four steps (% ai_generated)

```
  styles.csv
      |
      v
  1. Ingest & normalize   [deterministic]   -> step1_normalized.parquet
      |
      v
  2. Product ontology     [rules; LLM optional]  -> step2_ontology.json
      |
      v
  3. Classify pairs       [rules + LLM tie-break] -> step3_relationships.json
      |
      v
  4. Gap report           [deterministic]   -> step4_gap_report.{json,md}
```

- **Step 1 — Ingest & normalize** (deterministic): loads `styles.csv`,
  drops rows missing required attributes, derives a brand heuristically
  from `productDisplayName`, dedupes by id, filters to Apparel + Footwear.
- **Step 2 — Product ontology** (rules; LLM optional): per SKU, aligns
  `articleType` to the Vastra taxonomy, normalizes `baseColour` to a
  colour family, and regex-extracts pattern and material from the product
  name. The LLM path only fires for rows the rules leave ambiguous.
- **Step 3 — Classify pairs** (rules + LLM tie-break): blocks on
  (master_category, article_type, gender), then classifies each candidate
  pair as VARIANT / SUBSTITUTE / UNRELATED with transparent scoring. An
  LLM tie-breaker fires only for borderline scores when a key is present.
- **Step 4 — Gap report** (deterministic): counts the substitution gap,
  top brands by gap %, and top gap SKUs by near-miss count.

An optional **Step 5 — Consistency experiment** tests the newsletter claim
that raw autonomous agents give inconsistent answers (see below).

## How to run

```bash
uv sync
uv run substitutes-agent download        # fetch styles.csv via Kaggle (needs ~/.kaggle/access_token)
uv run substitutes-agent run             # run steps 1-4 on the full dataset
uv run substitutes-agent run --sample    # run on the committed 500-row fixture (no download needed)
uv run substitutes-agent consistency     # run Step 5 (skips cleanly without LLM keys)
uv run substitutes-agent consistency --dry-run   # print the call estimate and abort
```

`run --sample` completes in under a second with no API key set and
produces all Step 1–4 artifacts under `output/`. Reviewers can read the
committed `output/` artifacts without running anything.

## Results from the committed run

From `output/step4_gap_report.md` (run on `data/sample.parquet`, 500
Apparel + Footwear SKUs, no LLM):

- Total SKUs analyzed: **500**
- Substitution gap: **129 SKUs (25.8%)** with zero SUBSTITUTE edges
- SUBSTITUTE edges proposed: **947**
- VARIANT edges proposed: **109**
- Wall-clock: ~0.2s total across all four steps

Actual results

## Testing the consistency claim

Step 5 (`uv run substitutes-agent consistency`) tests Zenline's stated thesis
that raw autonomous agents give inconsistent answers to the same
question: it takes up to 30 borderline pairs (total_score in [0.40, 0.60]),
runs each available model 3 times, and reports per-model self-agreement
and cross-model agreement rates. The rules baseline is 100% by
construction.

Step 5 was run for this commit. See `output/step5_consistency.md` for
the per-model self-agreement table and cross-model agreement figure. The
interpretation paragraph in that file is intentionally left as a
placeholder — the point is to report what the numbers say after seeing
them, not to write a conclusion in advance.

## Where this fits

Four papers directly relevant to what this pipeline does or chooses not to do.

- **Ma et al., *Cost-Efficient RAG for Entity Matching with LLMs*
  ([arXiv:2602.05708](https://arxiv.org/abs/2602.05708), Feb 2026).**
  Classical blocking heuristics from the Magellan era plus modern LLMs is
  the economically viable pattern at scale. Step 3 of this pipeline does
  the same blocking-based grouping, minus the external knowledge-graph
  lookup. On 9 standard entity-matching benchmarks, their approach beats
  supervised PLMs on 6 of 9.

- **Zhang, Khan, Walter (Amazon), *CatalogRAG* (LLM4ECommerce Workshop
  at KDD'25, Aug 2025).**
  Multi-stage BM25 retrieval plus brand and popularity reranking plus
  few-shot examples fed into attribute-specific LLM prompts to fill
  missing product attributes. Deployed across Amazon US, DE, FR stores.
  Step 2's LLM path in this pipeline is a scrappier version of the same
  intuition, passing already-observed values instead of retrieved similar
  entries. Notably, CatalogRAG explicitly chose BM25 over vector
  embeddings for operational simplicity, which is a direct precedent for
  the no-embeddings choice here.

- **Tang et al., *Large Reasoning Embedding Models*
  ([arXiv:2510.14321](https://arxiv.org/abs/2510.14321), Oct 2025).**
  Deployed on Taobao and Tmall since August 2025. On difficult queries
  whose lexical form is far from target items, direct-embedding retrievers
  fail. Their fix is to have the LLM generate a compact keyword chain of
  thought first, then embed the combined representation. This is the
  direction this pipeline consciously does not take. Our retrieval is
  deterministic blocking on structured attributes, not semantic search
  over free-form queries.

- **Kothari et al. (Flipkart), semantic retrieval for e-commerce product
  search ([arXiv:2606.01504](https://arxiv.org/abs/2606.01504), May 2026).**
  Production Siamese Qwen3-Embedding-4B dual-encoder trained in two
  stages: contrastive plus Relative Odds Alignment for Retrieval (ROAR).
  Their graded relevance hierarchy is Perfect Match > Substitute >
  Complementary > Irrelevant, which maps onto this pipeline's VARIANT /
  SUBSTITUTE / UNRELATED with a fourth class this pipeline does not
  handle. What you build when you have labeled substitute data. This
  demo does not, and the rules pipeline is the pre-training baseline you
  would start from.

## Honest limitations

- **No ground truth in this demo, though public benchmarks exist.** There
  are no labeled "these are actual substitutes" annotations on the
  paramaggarwal Kaggle dataset. Two public
  alternatives could be dropped in for evaluation: the Amazon Shopping
  Queries Dataset (Reddy et al., 2022) with ESCI labels (Exact,
  Substitute, Complement, Irrelevant), and the Magellan and WDC
  entity-matching benchmarks used in Ma et al. (2026). I did not use
  them for this demo because the paramaggarwal dataset is closer to the
  raw-retailer-catalog state Zenline describes working with in their
  case studies. Adopting one of those benchmarks would be the obvious
  first extension to close the evaluation loop.

- **English-only, on a single dataset.** The paramaggarwal dataset is
  primarily English with some transliterated terms. Zhang et al. (2025)
  demonstrate that in-catalog few-shot approaches yield the largest
  gains on non-English stores, where LLM base performance is weaker.
  Extending this pipeline to a multilingual retailer would likely make
  the Step 2 LLM path more valuable, not less.

- The paramaggarwal dataset is sourced from myntra.com and is
  volunteer/catalog-contributed; it is sparse in places and India-skewed.
- The dataset has no dedicated brand column. I derive brand
  heuristically from the leading tokens of `productDisplayName` (consume
  until a boundary token — gender / usage / colour / article noun —
  capped at 3). It is imperfect by design; mis-derived brands flow
  directly into the VARIANT/SUBSTITUTE rules.
- The rules-based ontology is deliberately simple: a small regex
  vocabulary for pattern and material, and a hand-built colour-family map.
  ~72% of sample rows have neither pattern nor material extracted and are
  flagged low-confidence — exactly the gap the optional LLM path is meant
  to lift.
- This work does NOT MAKE ANY CLAIMS ON BEHALF OF ZENLINE IF YOURE READING THIS HERE. 

## References

- https://www.theretailbulletin.com/retail-solutions/qa-arber-sejdiji-co-founder-ceo-zenline-ai-16-04-2026/
- https://www.tryzenline.ai/insights
- https://www.startuprad.io/post/zenline-ai-agentic-assortment-decisions-win-retail-margins-startuprad-io


- Dataset: paramaggarwal fashion-product-images-dataset (Kaggle, MIT),
  sourced from myntra.com:
  https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset

## License & attribution

- Repo code: **MIT** (see `pyproject.toml`).
- Data: attributed to **paramaggarwal** on Kaggle, sourced from
  myntra.com, under the dataset's MIT license as stated on Kaggle. The
  full `styles.csv` is gitignored; only the 500-row `data/sample.parquet`
  slice is committed.
- The Vastra taxonomy adaptation is my own transcription of vocabulary I
  built for vastra.cc; no Vastra source code is included.

## Why I built this

Because it is also a good test for Phase 2 of Vastra and where I'd take my own project next. Building this validated something we'd hit in Vastra's internal prototype testing: model changes meant inconsistent tagging on the same garment. Vastra's own architecture already has a pipelined flow where agents are called in for verification and tagging but do not make autonomous decisions, precisely because of these inconsistencies. Zenline's public argument for agentic workflows lands on the same conclusion from the retailer side of the aisle, which is why the demo was worth building.

