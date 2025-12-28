# DecayRAG Evaluation Plan

## Multi-Hop Retrieval Benchmark for Distance-Decay RAG

---

## 0. OpenAI API Configuration

### GPT-5.2 Model & Responses API

We use OpenAI's latest **GPT-5.2** model with the new **Responses API** (released March 2025) for question generation and validation.

**Model Selection**:
| Model | Use Case | Pricing |
|-------|----------|---------|
| `gpt-5.2` | Question generation, validation | $1.75/1M input, $14/1M output |
| `gpt-5.2-pro` | Complex reasoning tasks | Higher cost, better quality |
| `gpt-5.2-chat-latest` | Fast, simple tasks | Lower latency |

**Key Features**:
- 400,000 token context window
- 128,000 max output tokens
- Built-in tools: web search, file search, code interpreter
- Reasoning effort control: `minimal`, `low`, `medium`, `high`, `xhigh`
- 90% discount on cached inputs

**API Syntax** (Responses API):
```python
from openai import OpenAI

client = OpenAI()

# Basic call
response = client.responses.create(
    model="gpt-5.2",
    input="Your prompt here"
)
print(response.output_text)

# With instructions and reasoning
response = client.responses.create(
    model="gpt-5.2",
    instructions="You are a precise question generator for RAG evaluation.",
    input=[
        {"role": "user", "content": "Generate a multi-hop question..."}
    ],
    reasoning={"effort": "high"},
    max_output_tokens=1024,
)

# Multi-turn conversation
response = client.responses.create(
    model="gpt-5.2",
    input=[
        {"role": "user", "content": "First message"},
        {"role": "assistant", "content": "Response"},
        {"role": "user", "content": "Follow-up"},
    ],
)
```

**Why Responses API over Chat Completions**:
- 3% improvement on SWE-bench with same prompt
- 40-80% better cache utilization (lower costs)
- Server-side conversation state management
- Future features land here first

**References**:
- [Introducing GPT-5.2](https://openai.com/index/introducing-gpt-5-2/)
- [GPT-5.2 Model Documentation](https://platform.openai.com/docs/models/gpt-5.2)
- [Why we built the Responses API](https://developers.openai.com/blog/responses-api/)
- [Migrate to Responses API](https://platform.openai.com/docs/guides/migrate-to-responses)

---

## 1. Research Objective

Evaluate whether DecayRAG's distance-decay mechanism improves retrieval of semantically connected chunk pairs compared to baseline methods. This addresses a key limitation in existing RAG systems: the inability to retrieve multiple related chunks needed for multi-hop reasoning.

### Hypotheses

**H1**: DecayRAG's distance-decay mechanism achieves higher Pair Recall@K than baselines because neighboring chunks boost each other's retrieval scores.

**H2**: The improvement is more pronounced for chunk pairs within the decay window (2-3 positions apart) compared to distant pairs (5+ positions).

---

## 2. Related Work

| Paper/System | Key Contribution | Limitation Addressed by DecayRAG |
|--------------|------------------|----------------------------------|
| [MultiHop-RAG (COLM 2024)](https://arxiv.org/abs/2401.15391) | First multi-hop RAG benchmark | We provide systematic evaluation methodology |
| [MuSiQue (TACL 2022)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00475/110996/) | Bottom-up multi-hop question construction | We adapt this for RAG-specific evaluation |
| [Late Chunking (Jina AI 2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) | Context-preserving embeddings | DecayRAG is model-agnostic, no retraining |
| [HotpotQA (EMNLP 2018)](https://hotpotqa.github.io/) | Multi-hop QA dataset | Known shortcut problems; we use stricter filters |

---

## 3. Dataset Construction

### 3.1 Document Collection

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Source | Wikipedia | Diverse, factual, well-structured |
| # Articles | 5 | Sufficient for statistical significance |
| Min Length | 5,000 words | Ensures multiple chunks per article |
| Domains | Science, History, Geography, Biography, Technology | Domain diversity |

**Candidate Articles** (long, information-rich):
1. `Artificial_intelligence` - Technology
2. `World_War_II` - History
3. `Amazon_rainforest` - Geography
4. `Albert_Einstein` - Biography
5. `Climate_change` - Science

### 3.2 Chunking Parameters

| Parameter | Value |
|-----------|-------|
| Chunk size | 200 tokens |
| Overlap | 20 tokens |
| Expected chunks/article | 30-50 |
| **Total chunks** | 150-250 |

### 3.3 Multi-Hop Question Generation (Bottom-Up Approach)

Inspired by MuSiQue's methodology to ensure genuine multi-hop reasoning:

```
Step 1: Bridge Chunk Identification
├── For each article, identify chunk pairs (A, B) where:
│   ├── Chunk A contains Entity/Concept X
│   ├── Chunk B references X or its consequence
│   ├── Distance: 2 ≤ |pos_A - pos_B| ≤ 10
│   └── Neither chunk alone answers potential questions
│
Step 2: Single-Hop Fact Extraction (GPT-5.2)
├── Extract key facts from Chunk A about Entity X
└── Extract key facts from Chunk B that depend on X
│
Step 3: Multi-Hop Question Composition (GPT-5.2)
├── Compose question requiring BOTH facts
├── Apply synonym/paraphrase transformation
└── Ensure no keyword overlap with source
│
Step 4: Disconnected Reasoning Validation (GPT-5.2)
├── Verify: Chunk A alone → Cannot answer
├── Verify: Chunk B alone → Cannot answer
└── Verify: Chunk A + B → Can answer correctly
```

### 3.4 Question Specification

| Parameter | Value |
|-----------|-------|
| Questions per article | 10-20 |
| **Total questions** | 50-100 |
| Target chunk distance | 2-10 chunks apart |
| Validation rate | 100% (automated + spot-check) |

---

## 4. Baseline Methods

### 4.1 Retrieval Baselines

| Method | Description | Key Difference from DecayRAG |
|--------|-------------|------------------------------|
| **Naive Chunking** | Fixed-size chunks, independent embeddings | No neighbor awareness |
| **Overlapping Chunks** | Sliding window with overlap | Overlap but no score blending |
| **Sentence Window** | Retrieve chunk, expand ±N at generation | Post-hoc expansion only |
| **BM25 (Sparse)** | TF-IDF keyword matching | No semantic understanding |
| **Hybrid (BM25 + Dense)** | RRF fusion of sparse + dense | No decay mechanism |
| **DecayRAG (Ours)** | Distance-decay + embedding blending | Full neighbor-aware retrieval |

### 4.2 DecayRAG Ablations

| Variant | Description |
|---------|-------------|
| DecayRAG-Exp | Exponential decay kernel |
| DecayRAG-Gauss | Gaussian decay kernel |
| DecayRAG-Power | Power-law decay kernel |
| DecayRAG-NoBlend | Decay only, no document embedding blend |
| DecayRAG-Full | Full pipeline (decay + blend) |

---

## 5. Evaluation Metrics

### 5.1 Retrieval Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Recall@K** | `\|retrieved ∩ target\| / \|target\|` | Fraction of target chunks retrieved |
| **Pair Recall@K** | `1 if both targets in top-K else 0` | Both chunks retrieved (critical for multi-hop) |
| **MRR** | `1 / rank_first_target` | Ranking quality |
| **NDCG@K** | `DCG@K / IDCG@K` | Normalized ranking quality |

### 5.2 Multi-Hop Specific Metrics

| Metric | Description |
|--------|-------------|
| **Bridge Recall@K** | Is the "bridge" chunk (connecting A and B) retrieved? |
| **Answer EM** | Exact match of generated answer vs. ground truth |
| **Answer F1** | Token-level F1 of generated answer |

### 5.3 Evaluation Protocol

```python
K_VALUES = [3, 5, 10, 20]  # Top-K thresholds

for question in dataset:
    target_chunks = question.ground_truth_chunks  # [chunk_A, chunk_B]

    for method in METHODS:
        retrieved = method.retrieve(question.text, top_k=max(K_VALUES))

        for k in K_VALUES:
            top_k_retrieved = retrieved[:k]

            # Compute metrics
            recall_k = len(set(top_k_retrieved) & set(target_chunks)) / len(target_chunks)
            pair_recall_k = 1 if all(t in top_k_retrieved for t in target_chunks) else 0
            mrr = 1 / (min_rank_of_target + 1)

            record_metric(method, k, recall_k, pair_recall_k, mrr)
```

---

## 6. Implementation Plan

### 6.1 Module Structure

```
eval/
├── __init__.py
├── dataset/
│   ├── __init__.py
│   ├── wikipedia_loader.py       # Download & parse Wikipedia
│   ├── chunk_pair_finder.py      # Identify bridge chunk pairs
│   ├── question_generator.py     # GPT-5.2 multi-hop question synthesis
│   └── dataset_validator.py      # Validate question quality
├── baselines/
│   ├── __init__.py
│   ├── naive_retrieval.py        # Naive chunking baseline
│   ├── bm25_retrieval.py         # Sparse retrieval baseline
│   ├── sentence_window.py        # Sentence window expansion
│   └── hybrid_retrieval.py       # BM25 + Dense hybrid
├── metrics/
│   ├── __init__.py
│   ├── recall.py                 # Recall@K, Pair Recall@K
│   ├── ranking.py                # MRR, NDCG
│   └── answer_metrics.py         # EM, F1 for answers
├── run_evaluation.py             # Main evaluation harness
└── analyze_results.py            # Statistical analysis & plots
```

### 6.2 Development Phases

| Phase | Tasks | Deliverables |
|-------|-------|--------------|
| **Phase 1** | Wikipedia loader, chunking | `wikipedia_loader.py`, processed articles |
| **Phase 2** | Chunk pair identification | `chunk_pair_finder.py`, candidate pairs |
| **Phase 3** | Question generation (GPT-5.2) | `question_generator.py`, 50-100 questions |
| **Phase 4** | Baseline implementations | All baseline retrievers |
| **Phase 5** | Metrics & evaluation harness | Complete evaluation pipeline |
| **Phase 6** | Run experiments & analysis | Results tables, figures |

---

## 7. Question Generation Prompts

### 7.1 Bridge Chunk Identification Prompt

```
You are analyzing text chunks to find pairs suitable for multi-hop questions.

Given chunks from document "{doc_title}":
{chunks_with_positions}

Identify pairs (Chunk_i, Chunk_j) where:
1. They share a connecting entity, concept, or causal relationship
2. They are 2-10 positions apart
3. Information in both is needed to answer a non-trivial question

Output JSON array of pairs:
[
  {
    "chunk_a_pos": <int>,
    "chunk_b_pos": <int>,
    "connecting_entity": "<string>",
    "relationship": "<string>"
  }
]
```

### 7.2 Multi-Hop Question Generation Prompt

```
You are creating a multi-hop reasoning question for RAG evaluation.

Document: {doc_title}
Chunk A (position {pos_a}):
{chunk_a_text}

Chunk B (position {pos_b}):
{chunk_b_text}

Connecting entity/concept: {connecting_entity}

Generate a question that:
1. REQUIRES information from BOTH chunks to answer
2. Uses SYNONYMS and PARAPHRASING - do NOT use these words: {keywords_to_avoid}
3. Cannot be answered by reading only one chunk
4. Has a clear, factual answer (not opinion)

Output JSON:
{
  "question": "<paraphrased question>",
  "answer": "<factual answer>",
  "reasoning": "<why both chunks needed>",
  "chunk_a_contribution": "<what info comes from A>",
  "chunk_b_contribution": "<what info comes from B>",
  "difficulty": "easy|medium|hard"
}
```

### 7.3 Validation Prompt

```
You are validating a multi-hop question for quality.

Question: {question}
Expected Answer: {answer}

Chunk A: {chunk_a_text}
Chunk B: {chunk_b_text}

Evaluate:
1. Can this question be answered using ONLY Chunk A? (yes/no + explanation)
2. Can this question be answered using ONLY Chunk B? (yes/no + explanation)
3. Can this question be answered using BOTH chunks? (yes/no + explanation)
4. Does the question avoid exact keywords from the source? (yes/no)
5. Is the answer factually correct? (yes/no)

Output JSON:
{
  "answerable_from_a_only": false,
  "answerable_from_b_only": false,
  "answerable_from_both": true,
  "avoids_keywords": true,
  "factually_correct": true,
  "validation_passed": true,
  "issues": []
}
```

---

## 8. Expected Results Format

### 8.1 Main Results Table

| Method | Recall@5 | Recall@10 | Pair Recall@5 | Pair Recall@10 | MRR |
|--------|----------|-----------|---------------|----------------|-----|
| BM25 | - | - | - | - | - |
| Naive Chunking | - | - | - | - | - |
| Overlapping | - | - | - | - | - |
| Sentence Window | - | - | - | - | - |
| Hybrid | - | - | - | - | - |
| **DecayRAG** | - | - | - | - | - |

### 8.2 Ablation Results

| DecayRAG Variant | Pair Recall@5 | Pair Recall@10 |
|------------------|---------------|----------------|
| Exponential (τ=N/10) | - | - |
| Gaussian (σ=N/10) | - | - |
| Power-law (α=1.0) | - | - |
| No Blend (decay only) | - | - |
| Full (decay + blend) | - | - |

### 8.3 Distance Analysis

| Chunk Pair Distance | DecayRAG Pair Recall@10 | Baseline Pair Recall@10 | Δ |
|---------------------|-------------------------|-------------------------|---|
| 2-3 (within window) | - | - | - |
| 4-5 | - | - | - |
| 6-10 (outside window) | - | - | - |

---

## 9. Statistical Analysis

- **Significance Testing**: Paired t-test or Wilcoxon signed-rank test
- **Effect Size**: Cohen's d for practical significance
- **Confidence Intervals**: 95% CI via bootstrap resampling
- **Multiple Comparisons**: Bonferroni correction

---

## 10. Timeline

| Week | Tasks |
|------|-------|
| 1 | Dataset construction (Wikipedia + chunking) |
| 1 | Question generation pipeline |
| 2 | Baseline implementations |
| 2 | Evaluation harness |
| 3 | Run experiments |
| 3 | Analysis & visualization |

---

## 11. Success Criteria

The evaluation will be considered successful if:

1. **Dataset Quality**: >80% of generated questions pass validation
2. **Statistical Significance**: p < 0.05 for DecayRAG vs. best baseline on Pair Recall
3. **Practical Significance**: >10% relative improvement on Pair Recall@10
4. **Hypothesis Support**: Larger gains for within-window chunk pairs

---

## 12. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Generated questions too easy | Strict validation + manual review |
| Keyword matching shortcuts | Aggressive synonym replacement |
| Insufficient data | Increase to 10 articles if needed |
| API costs | Batch processing, caching |

---

## Appendix A: Wikipedia Article Candidates

| Article | Domain | Est. Length | Why Selected |
|---------|--------|-------------|--------------|
| Artificial_intelligence | Tech | ~15k words | Rich interconnections |
| World_War_II | History | ~20k words | Temporal reasoning |
| Amazon_rainforest | Geography | ~8k words | Spatial relationships |
| Albert_Einstein | Biography | ~12k words | Causal chains |
| Climate_change | Science | ~18k words | Multi-factor reasoning |

---

## Appendix B: Evaluation Code Skeleton

```python
# run_evaluation.py

from eval.dataset import load_multihop_dataset
from eval.baselines import BASELINE_METHODS
from eval.metrics import compute_all_metrics
from decayrag import retrieve

def main():
    # Load dataset
    dataset = load_multihop_dataset("data/multihop_qa.json")

    # Methods to evaluate
    methods = {
        "BM25": bm25_retrieve,
        "Naive": naive_retrieve,
        "Overlapping": overlap_retrieve,
        "SentenceWindow": sentence_window_retrieve,
        "Hybrid": hybrid_retrieve,
        "DecayRAG": decayrag_retrieve,
    }

    # Run evaluation
    results = {}
    for name, method in methods.items():
        results[name] = evaluate_method(method, dataset)

    # Compute statistics & output
    analyze_and_report(results)

if __name__ == "__main__":
    main()
```
