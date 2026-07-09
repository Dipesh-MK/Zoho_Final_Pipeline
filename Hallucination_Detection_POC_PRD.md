# PRD: Post-Mortem Hallucination Detection POC

**Author:** [Your name]
**Date:** [Date]
**Status:** Draft for review

---

## 1. Problem Statement

Our support bots (RAG-based) sometimes generate answers that are not actually
supported by the documents retrieved for that query. A specific and important
failure pattern: the retriever pulls documents that are *topically related but
not actually relevant* to what's being asked, and the model uses them to
confidently state something false — e.g., a customer asks whether Site24x7
supports GitHub push/pull integration; the retriever surfaces GitHub
*monitoring* docs (related, wrong feature); the model answers as if the
integration exists.

We need a way to **detect** this after the fact, from logs. We are not able to
re-query, re-retrieve, or do live web search — analysis is strictly post-mortem
on already-logged data.

## 2. Goals

- Determine whether off-the-shelf faithfulness/hallucination-detection tools
  can reliably flag answers that are **not supported** by the documents
  actually retrieved for that query — including the harder "adjacent topic,
  wrong capability" case, not just obvious unsupported claims.
- Produce a quantitative comparison of 2–3 candidate tools on a labeled set.
- Recommend whether an off-the-shelf tool is sufficient, or whether a
  custom-tuned checker is needed.

## 3. Non-Goals

- Not fixing retrieval or suggesting better documents (detection only).
- Not performing corrective/live RAG — no new searches, no web access.
- Not modifying or fine-tuning the production LLM.
- Not building a production pipeline yet — this is a feasibility POC.

## 4. Inputs Available

For each logged interaction, we only have:
- The user query
- The source documents that were included in that RAG call
- The model's response

No access to model internals, logits, or embeddings (black-box constraint).

## 5. Approach

```
For each (query, retrieved_docs, response) record:

  Step 1 — Relevance grading (CRAG-style, detection only, no re-retrieval)
           Score whether retrieved_docs are actually relevant to query.

  Step 2 — Faithfulness check
           Score whether every claim in response is supported by retrieved_docs.

  Step 3 — Combine
           Flag as high-risk hallucination if response makes a confident,
           specific claim NOT backed by retrieved_docs — regardless of whether
           docs were topically related.
```

## 6. Dataset Plan

Two-part dataset, because a public dataset alone won't cover our specific risk case:

**A. Public dataset (sanity check)**
Use a subset of **RAGTruth** (Niu et al., ACL 2024) — human-annotated,
span-level hallucination labels, designed exactly for this task.
Purpose: confirm the tools work at all and give a baseline accuracy number
on a standard benchmark before trusting them on our own data.

**B. Custom mock dataset (the actual test of the concern raised)**
Hand-built set of ~20–30 examples specifically modeling the
"adjacent-topic, wrong-capability" pattern, e.g.:

| Query | Retrieved doc (real content) | Correct answer | Hallucinated answer (to test detection) |
|---|---|---|---|
| "Can I push/pull code via GitHub integration?" | Doc about GitHub *monitoring* only | "Not supported — GitHub integration is monitoring-only" | "Yes, you can push and pull directly" |

Design rule: each mock example pairs a real capability with a plausible
*but nonexistent* adjacent capability, so the retrieved doc is genuinely
related — this is what makes it a hard case, not a random distractor.
Each example is labeled `faithful` or `hallucinated` by us in advance
(ground truth), so we can measure precision/recall directly.

## 7. Tools to Evaluate

| Tool | Type | Notes |
|---|---|---|
| LettuceDetect | Pretrained span classifier (RAGTruth-tuned) | Free, local, no API cost |
| Vectara HHEM-2.1-Open | Pretrained consistency classifier | Free, local, no API cost |
| RAGAS (faithfulness metric) | LLM-as-judge | Needs a judge LLM call; test with explicit prompt instructing strict capability-level matching, not topic-level |

## 8. Success Metrics

- **Overall accuracy** on RAGTruth subset (sanity check vs. published baselines)
- **Precision/Recall specifically on the mock "adjacent-capability" subset**
  — this is the number that actually answers the manager's question,
  since overall accuracy can look fine while missing the hard cases entirely
- False positive rate on known-correct answers (don't want to flag good answers)

## 9. Deliverables

1. Jupyter notebook: pipeline code, per-example results, metrics tables
2. One-page summary: results table + recommendation (go with off-the-shelf tool
   vs. need custom fine-tuning), for the non-technical walkthrough

## 10. Timeline (proposed)

| Day | Task |
|---|---|
| 1 | Build mock dataset (20–30 labeled examples) + pull RAGTruth subset |
| 2 | Wire up pipeline for all 3 tools |
| 3 | Run evaluation, compute metrics, error analysis on misses |
| 4 | Write up results + recommendation, prep notebook for walkthrough |

## 11. Open Risks

- LLM-judge (RAGAS) approach may itself hallucinate/misjudge on edge cases —
  needs its own spot-checking.
- Off-the-shelf tools are trained on general-domain data, not Zoho's product
  docs — may need domain fine-tuning if precision/recall on the mock set is weak.
- Mock dataset size is small (~20–30 examples); good for a directional POC
  signal, not a statistically rigorous evaluation — flag this explicitly when
  presenting results.
