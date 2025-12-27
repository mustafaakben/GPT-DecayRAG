"""DecayRAG - document ingestion and retrieval utilities."""

__version__ = "0.1.0"

from .ingest import (
    load_config,
    resolve_ingest_settings,
    parse_document,
    chunk_nodes,
    embed_chunks,
    upsert_embeddings,
    batch_ingest,
    batch_ingest_from_config,
)
from .retrieval import (
    embed_query,
    compute_chunk_similarities,
    blend_scores,
    top_k_chunks,
    retrieve,
)
from .post import assemble_context, generate_answer

__all__ = [
    "load_config",
    "resolve_ingest_settings",
    "parse_document",
    "chunk_nodes",
    "embed_chunks",
    "upsert_embeddings",
    "batch_ingest",
    "batch_ingest_from_config",
    "embed_query",
    "compute_chunk_similarities",
    "blend_scores",
    "top_k_chunks",
    "retrieve",
    "assemble_context",
    "generate_answer",
]
