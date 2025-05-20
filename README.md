# DecayRAG

*A distance‑decay, order‑aware retrieval‑augmented generation pipeline for long, hierarchical documents.*

---

## 1 Objective

Large documents must be split into chunks before vector search, but this often breaks the logical flow—pronouns, bridging anaphora, or implicit references get separated from their antecedents. **DecayRAG** rescues those missing links by letting each chunk 'borrow' relevance from its neighbours with a **continuous distance‑decay kernel** that automatically scales with document length. The resulting scores/embeddings are blended with a document‑level vector so the retriever sees both local detail *and* global context.

---

## 2 Why DecayRAG is Different

* **Continuous decay, not fixed windows** – neighbouring influence fades smoothly (exponential/gaussian/power), instead of copying a hard ±N‑sentence window.
* **Length‑adaptive** – decay parameters are derived from *N* chunks, so short docs get a tight window, long docs a broader one.
* **Embedding‑level fusion** – we mix raw, neighbour‑smoothed, and document embeddings (α, β, γ) *before* similarity search, enabling learned rerankers or ANN pruning.
* **Hierarchy‑aware ingestion** – chunking respects book → chapter → section boundaries; metadata survives the trip to the vector store.

---

## 3 Related Work

| Cluster             | Paper / Toolkit                                      | Relation                                                         |
| ------------------- | ---------------------------------------------------- | ---------------------------------------------------------------- |
| Fixed windows       | Sentence‑Window Retriever (LlamaIndex, 2024)         | uniform ±N context, no decay                                     |
| Better chunking     | *Late Chunking* (Jina AI blog, 2025)                 | embeds whole doc *then* slices; no neighbour weighting           |
| Graph hops          | *Chunk‑Interaction Graph* (Zhang et al., EMNLP 2024) | discrete edges, not continuous decay                             |
| Rewriting           | Craddock et al., *Decontextualization* (ICLR 2025)   | rewrites chunks; DecayRAG leaves text untouched                  |
| Iterative retrieval | Jain et al., **RICHES** (EMNLP 2024)                 | multi‑hop LLM retrieval; DecayRAG is single‑shot but order‑aware |



---

## 4 Method in Detail

The details will be added soon.

### 4.1 Decay kernels

*Exponential* `w(d)=exp(−d/τ)` | *Gaussian* `exp(−d²/2σ²)` | *Power‑law* `1/(1+d)^α`

`τ, σ, α` default to *N / 10* so the effective window widens in longer docs.

### 4.2 Embedding fusion

$**e**ᵢ′ = α·**e**ᵢ + β·**\~e**ᵢ + γ·**E**₍doc₎$

* **e**ᵢ – raw chunk embedding 
* **\~e**ᵢ – neighbour‑smoothed embedding 
* **E**₍doc₎ – global document vector (mean / TF‑IDF‑weighted) 

---

## 5 Project Structure & Tasks

### 5.1 Current Folder Structure

```
decayrag/
│
├─ decayrag/
│   ├─ __init__.py        # Package initialization
│   ├─ ingest.py          # Document ingestion
│   ├─ pooling.py         # Embedding pooling helpers
│   └─ retrieval.py       # Query-time retrieval
│
├─ examples/
│   └─ quickstart.py      # End-to-end example
│
├─ tests/
│   └─ test_ingest.py     # pytest unit tests
│
└─ requirements.txt       # Dependencies
```

### 5.2 Implementation Status

| Phase | Milestone / Function                                                      | Status |
| ----- | ------------------------------------------------------------------------- | ------ |
| **0** | Vector‑store config (`config.yaml`)                                       | ⬜️     |
| **1** | `parse_document`, `chunk_nodes`, `embed_chunks`, `upsert_embeddings`      | ⬜️     |
| **2** | `embed_query`, `compute_chunk_similarities`, `top_k_chunks`, `retrieve()` | ⬜️     |
| **3** | `assemble_context`, `generate_answer`                                     | ⬜️     |
| **4** | Eval harness, baselines, sweeps                                           | ⬜️     |
| **5** | Docs, packaging, CI, license                                              | ⬜️     |

*(Live Kanban coming soon.)*

---

## 6 Quick Start

```bash
# install from source (PyPI package coming soon)
git clone https://github.com/mustafaakben/DecayRAG.git
pip install -e ./DecayRAG

# python ≥ 3.9
python examples/quickstart.py --input docs/example.pdf --query "When was the first self‑driving demo?"
```
Set the `OPENAI_API_KEY` environment variable when using OpenAI embeddings.

---

## 7 Requirements

* Python 3.9+
* Access to an embedding model (OpenAI, Cohere, GTE, BGE, etc.)
* Alternatively, you can call sentence-transformer models (e.g. all-MiniLM-L6-v2)
* All other dependencies are listed in `requirements.txt`

---

## 8 Running Tests

Install the dependencies listed in `requirements.txt` and run `pytest` from the
project root:

```bash
pip install -r requirements.txt
pytest
```

---

## 9 License

DecayRAG will be released under the **MIT License**. The `LICENSE` file is not yet included but will be added later.

---

## 10 Citation

```bibtex
@misc{decayrag2025,
  title   = {DecayRAG: Distance-Decay, Order-Aware Retrieval for Long Documents},
  author  = {Akben, Mustafa},
  year    = {2025},
  howpublished = {\url{https://github.com/mustafaakben/DecayRAG}}
}
```

---

## 11 Contributing & Support

Open an issue or pull request on GitHub. Join the discussion tab for questions and roadmap chatter.

### **DecayRAG – end-to-end roadmap**

| Phase                           | Milestone / Task                                                                                     | Module / Function                                     | Status                |
| ------------------------------- | ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------- | --------------------- |
| **0  Infrastructure**           | Pick vector store (FAISS / LanceDB) & config file schema                                             | `config.yaml`                                         | ⬜️                    |
| **1  Document ingestion**       | **1.1 Hierarchy-aware parser** – walk files, capture *book → chapter → section → paragraph* metadata | `parse_document(path) → List[Node]`                   | ⬜️                    |
|                                 | **1.2 Chunker** – break text into tokens ≤ max\_len while respecting hierarchy & sentence boundaries | `chunk_nodes(nodes, max_tokens, overlap=0)`           | ⬜️                    |
|                                 | **1.3 Chunk embedding** – batch-call chosen model (OpenAI, GTE, etc.,sentence-transformer)           | `embed_chunks(chunks, model_id) → np.ndarray`         | ⬜️                    |
|                                 | **1.4 Document-level pooling** – already built                                                       | `compute_global_embedding(...)`                       | ✅                     |
|                                 | **1.5 TF-IDF weight calculator** – already built                                                     | `compute_tfidf_weights(...)`                          | ✅                     |
|                                 | **1.6 Vector store writer** – persist `{doc_id, chunk_id, text, meta, embedding}`                    | `upsert_embeddings(store, records)`                   | ⬜️                    |
| **2  Query-time pipeline**      | **2.1 Query embedding**                                                                              | `embed_query(text, model_id)`                         | ⬜️                    |
|                                 | **2.2 Similarity calc** – query vs. chunk vectors                                                    | `compute_chunk_similarities(query_vec, chunk_embeds)` | ⬜️                    |
|                                 | **2.3 Neighbour decay (scores &/or embeddings)** – already built                                     | `apply_neighbor_decay_*`                              | ✅                     |
|                                 | **2.4 Blending layer** – combine chunk, decayed, doc embeddings                                      | `blend_embeddings(...)`                               | ✅                     |
|                                 | **2.5 Final score builder** – if you choose score-space blending instead                             | `blend_scores(...)`                                   | ✅                     |
|                                 | **2.6 Top-k selector**                                                                               | `top_k_chunks(scores, k)`                             | ⬜️                    |
|                                 | **2.7 Retrieval wrapper** – orchestrates 2.1 → 2.6                                                   | `retrieve(query, k, settings) → List[Chunk]`          | ⬜️                    |
| **3  Post-retrieval**           | **3.1 Context assembly** – stitch neighbouring text, respect hierarchy                               | `assemble_context(chunks, window)`                    | ⬜️                    |
|                                 | **3.2 Generator call** – feed assembled context to LLM                                               | `generate_answer(context, query)`                     | ⬜️                    |
| **4  Evaluation & Experiments** | **4.1 Gap-challenge dataset builder**                                                                | `build_gap_set(corpus)`                               | ⬜️                    |
|                                 | **4.2 Metrics harness** – Recall\@k, MRR, EM                                                         | `evaluate(run_cfg)`                                   | ⬜️                    |
|                                 | **4.3 Baseline runners** – Sentence Window, Late-Chunk, etc.                                         | scripts                                               | ⬜️                    |
|                                 | **4.4 Parameter sweep** – α/β/γ & decay params                                                       | `sweep.py`                                            | ⬜️                    |
| **5  Tooling & Release**        | **5.1 Unit tests / CI**                                                                              | `tests/` + GH Actions                                 | ⬜️                    |
|                                 | **5.2 Docs & README** – quick-start, API, diagram                                                    | `docs/`                                               | ⬜️                    |
|                                 | **5.3 Packaging** – `setup.cfg`, upload to PyPI `decayrag`                                           | v0.1.0                                                | ⬜️                    |
|                                 | **5.4 License & paper draft**                                                                        | MIT/Apache-2.0 + arXiv template                       | ⬜️                    |

**Legend:** ✅ done | ⬜️ open

