# DecayRAG Overview

DecayRAG is a distance-decay, order-aware retrieval-augmented generation pipeline designed for long, hierarchical documents.

## The Core Problem

When documents are split into chunks for vector search, logical flow breaks. Pronouns, bridging anaphora, and implicit references get separated from their antecedents. Traditional RAG systems struggle with this because each chunk is treated independently.

## The Solution

DecayRAG rescues missing links by letting each chunk "borrow" relevance from its neighbors using a continuous distance-decay kernel. This approach automatically scales with document length.

### Key Features

1. **Continuous Decay**: Neighboring influence fades smoothly using exponential, gaussian, or power-law functions, instead of using a hard fixed window.

2. **Length-Adaptive**: Decay parameters are derived from the number of chunks, so short documents get a tight window while long documents get a broader one.

3. **Embedding-Level Fusion**: The system mixes raw, neighbor-smoothed, and document embeddings before similarity search.

4. **Hierarchy-Aware**: Chunking respects book, chapter, and section boundaries. Metadata survives through the vector store.

## How It Works

The pipeline has three main phases:

1. **Ingestion Phase**: Documents are parsed, chunked by tokens, embedded via OpenAI, and stored in FAISS.

2. **Retrieval Phase**: Queries are embedded, similarities computed, decay kernels applied, and scores blended to find the best chunks.

3. **Post-Retrieval Phase**: Top chunks are expanded with neighbors and passed to an LLM for answer generation.
