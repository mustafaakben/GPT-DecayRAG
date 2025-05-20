"""DecayRAG - document ingestion and retrieval utilities."""

__version__ = "0.1.0"

from .ingest import (
    parse_document,
    chunk_nodes,
    embed_chunks,
    upsert_embeddings,
    batch_ingest,
)
from .retrieval import (
    embed_query,
    compute_chunk_similarities,
    blend_scores,
    top_k_chunks,
    retrieve,
)

__all__ = [
    "parse_document",
    "chunk_nodes",
    "embed_chunks",
    "upsert_embeddings",
    "batch_ingest",
    "embed_query",
    "compute_chunk_similarities",
    "blend_scores",
    "top_k_chunks",
    "retrieve",
]

