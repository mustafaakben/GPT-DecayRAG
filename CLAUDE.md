# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DecayRAG is a distance-decay, order-aware retrieval-augmented generation (RAG) pipeline for long, hierarchical documents. The core innovation uses a continuous distance-decay kernel that allows chunks to "borrow" relevance from neighboring chunks, automatically scaling with document length.

## Commands

### Installation
```bash
pip install -r requirements.txt
pip install -e ./decayrag
```

### Running Tests
```bash
pytest                          # Run all unit tests
pytest -v                       # Verbose output
pytest tests/test_retrieval.py  # Run specific test file
pytest -m integration           # Run integration tests (requires OPENAI_API_KEY + RUN_OPENAI_INTEGRATION=1)
```

### Example Usage
```bash
# Ingest documents
python examples/quickstart.py docs/ data/index.faiss --config config.yaml

# Generate answers
python examples/generate_answer.py "Your question" --index_path data/index.faiss --config config.yaml
```

## Architecture

### Three-Phase Pipeline

1. **Ingestion** (`ingest.py`): Parse documents → chunk by tokens → embed via OpenAI → store in FAISS
2. **Retrieval** (`retrieval.py`): Embed query → compute similarities → apply decay kernels → blend scores → return top-k
3. **Post-retrieval** (`post.py`): Expand chunks with neighbors → generate answer via LLM

### Core Modules (in `decayrag/decayrag/`)

| Module | Purpose |
|--------|---------|
| `ingest.py` | Document parsing (TXT/MD/PDF), token-aware chunking, OpenAI embeddings, FAISS persistence |
| `retrieval.py` | Query embedding, similarity computation, decay-based retrieval |
| `pooling.py` | Decay kernels (exponential/gaussian/power-law), embedding blending, TF-IDF pooling |
| `post.py` | Context assembly with neighbor expansion, LLM answer generation |

### Key Data Structures

Each chunk carries metadata:
- `doc_id`, `chunk_id`: Document and position identifiers
- `position`: Global position for decay calculations
- `level`: Hierarchy path (e.g., `["Chapter 1", "Section 2"]`)
- `text`: Chunk content

Index files:
- `*.faiss`: Vector index
- `*.faiss.meta`: JSON lines metadata
- `*.faiss.npy`: NumPy embeddings array for decay/blending

### Decay Parameters

Default to `N / 10` where N = number of chunks:
- **Exponential** (τ): `w(d) = exp(-d/τ)`
- **Gaussian** (σ): `w(d) = exp(-d²/2σ²)`
- **Power-law** (α): `w(d) = 1/(1+d)^α`

## Configuration

`config.yaml` controls all settings. CLI arguments override config values.

Key settings:
- `model`: Embedding model (default: `text-embedding-3-small`)
- `max_tokens`, `overlap`: Chunking parameters
- `top_k`, `window`: Retrieval parameters
- `answer_model`, `answer_temperature`, `answer_max_tokens`: Generation settings

## Environment Variables

```bash
OPENAI_API_KEY=sk-...           # Required for embeddings and chat
OPENAI_EMBED_BATCH_SIZE=100     # Optional: API batch size
OPENAI_TIMEOUT=30               # Optional: API timeout
RUN_OPENAI_INTEGRATION=1        # Enable integration tests
```

## Testing Notes

- Tests mock OpenAI API at `_api_embed()` boundary
- Use `tmp_path` fixture for temporary FAISS indexes
- Integration tests require `RUN_OPENAI_INTEGRATION=1` and valid API key
