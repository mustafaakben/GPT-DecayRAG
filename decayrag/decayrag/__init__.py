"""DecayRAG - document ingestion and retrieval utilities."""

__version__ = "0.1.0"

from .ingest import (
    parse_document,
    chunk_nodes,
    embed_chunks,
    upsert_embeddings,
    batch_ingest,
)
from .retrieval import retrieve

__all__ = [
    "parse_document",
    "chunk_nodes",
    "embed_chunks",
    "upsert_embeddings",
    "batch_ingest",
    "retrieve",
]

