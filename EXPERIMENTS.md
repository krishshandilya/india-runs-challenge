# Engineering Experiments & Design Decisions: Candidate Discovery Engine

This document outlines the core technical experiments, benchmarks, and design decisions evaluated during the development of the two-stage candidate retrieval and ranking engine for the Redrob Hackathon.

---

## 1. Experiment 1: Ingestion Representation (Concise vs. Rich Embeddings)

### Objective
Determine the optimal text feature compilation format for the first-stage Bi-Encoder (`all-MiniLM-L6-v2`) retrieval.

### Methodology
We compared two profile encoding layouts across the 100,000 candidate dataset:
*   **Variant A (Rich Embedding - Estimated):** Concatenated the entire candidate profile, including current title, headline, summary, skills list, and description texts for all past career history roles.
*   **Variant B (Concise Embedding - Measured):** Concatenated only current title, headline, and top 15 skills:
$$\text{Text}(c) = \text{Title}(c) \ \Vert \ \text{Headline}(c) \ \Vert \ \text{"Skills: "} \ \Vert \ \left( \bigcup_{i=1}^{15} \text{Skill}\_i(c) \right)$$

### Results
*   **Pre-computation Latency (CPU):** Variant B (Concise) completed in **$\approx 10\text{ minutes}$** (Measured). Based on sequence length benchmarks, Variant A (Rich) was projected to take **$\approx 28\text{ minutes}$** (Estimated), representing a **2.8x speedup** in favor of the concise representation.
*   **Retrieval Quality (First-Pass Recall):** Variant A introduced severe semantic noise into the vector space. Dense representations matched generic operational descriptions (e.g. "client support", "resolved issues") rather than core AI/ML systems engineering. Variant B produced tight, high-signal vector clusters centered specifically around candidate titles and core skills.

### Decision
**Variant B (Concise Representation) was chosen for Stage-1 Retrieval.** The verbose history is reserved for the second-stage Cross-Encoder, where query-document cross-attention can process long sequences contextually.

---

## 2. Experiment 2: Retrieval Cutoff Comparison (300 vs. 500 Candidates)

### Objective
Measure the impact of increasing the retrieval candidate pool size ($K$) passed from the Bi-Encoder first-stage to the Cross-Encoder second-stage.

### Benchmarks
We evaluated retrieval pools of $K = 300$ and $K = 500$:

| Metric | Cutoff = 300 (Measured) | Cutoff = 500 (Measured) | Impact / Delta |
| :--- | :---: | :---: | :---: |
| **Inference Time (Cross-Encoder)** | $4.04\text{ seconds}$ | $6.08\text{ seconds}$ | $+2.04\text{ seconds}$ |
| **Peak RAM Footprint** | $\sim 1.2\text{ GB}$ | $\sim 1.6\text{ GB}$ | $+400\text{ MB}$ |
| **Top 20 Rankings Overlap** | $95\%$ ($19/20$ candidates) | $100\%$ | High Stability |
| **Recall Cliff Check** | Missed high-value targets | Surfaced `CAND_0030468` | **Recall Cliff Eliminated** |

### Findings
At $K = 300$, `CAND_0030468` (*Pooja Bose*, Senior Applied Scientist) was cut off in the first stage due to minor keyword differences from the JD. Increasing the cutoff to $K = 500$ allowed the Cross-Encoder to evaluate this candidate contextually, ranking them at **Rank 10** in the final list. At the same time, this change pushed `CAND_0075252` (*Atharv Vora*, a Mechanical Engineer with an inconsistent profile) out of the Top 20.

### Decision
**A Stage-1 retrieval cutoff of 500 candidates was selected.** It improves recall quality with a negligible $2.04$-second CPU latency increase.

---

## 3. Experiment 3: Title Relevance Matching (Hardcoded vs. Semantic)

### Objective
Compare candidate title relevance scoring methods in the heuristic layer.

### Methodology
*   **Heuristic 1 (Hardcoded Keywords):** Used a binary filter matching a list of fixed strings (e.g. `["ml engineer", "ai engineer"]`).
*   **Heuristic 2 (Semantic Cosine Space):** Dynamically computes cosine similarity between candidate titles/headlines and `"Senior AI Engineer"` target embedding.
*   **Deduplication Optimization:** To prevent CPU thermal throttling from embedding 200,000 candidate strings at runtime, we extracted only unique titles and headlines (reducing strings to encode from $200,000$ to **$2,515$** unique phrases).

### Results
*   **Scoring Resolution:** Under Heuristic 1, adjacent titles like `"AI Specialist"` or `"NLP Researcher"` received a baseline score of `0.20`. Under Heuristic 2, `"AI Specialist"` scored **`0.691`** and `"Senior MLE"` scored **`0.786`**, reflecting their true proximity.
*   **Execution Time:** deduplicated title batch encoding runs in **$\approx 20\text{ seconds}$** (Measured), whereas non-deduplicated encoding was projected to exceed $5\text{ minutes}$ (Estimated) on CPU, violating hackathon limits.

### Decision
**Semantic Cosine Similarity with Deduplicated Batch Encoding was chosen.** It generalizes dynamically to any adjacent titles without manual keyword updates.

---

## 4. Summary of Key Design Trade-offs

### 4.1 Multiplicative vs. Additive Heuristics
*   *Alternative:* Additive scoring ($\text{Semantic} + \text{Heuristics}$).
*   *Rejection:* Allowed local candidates with great experience but zero machine learning knowledge to rank highly.
*   *Selection:* **Multiplicative scoring** ($\text{Semantic} \times (0.7 + 0.3 \times \text{Heuristics})$) enforces a strict semantic gate. If a candidate has zero AI/ML relevance, their final ranking score stays close to zero.

### 4.2 Two-Stage Neural Cascade vs. Single-Stage Bi-Encoder Retrieval
*   *Alternative:* Sorting candidates using only first-pass Bi-Encoder Cosine similarity.
*   *Rejection:* Vector-only retrieval misses complex keyword intersections and full-context alignments.
*   *Selection:* **Two-Stage Cascade (Bi-Encoder $\rightarrow$ Cross-Encoder)**. The Cross-Encoder performs token-level query-document attention, recovering top-tier candidates at a cost of only $6.0$ seconds of additional runtime.
