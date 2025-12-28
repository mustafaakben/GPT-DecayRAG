"""Document ingestion utilities for DecayRAG."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List

import faiss
import numpy as np
import yaml
from pdfminer.high_level import extract_text
import tiktoken

__all__ = [
    "load_config",
    "resolve_ingest_settings",
    "parse_document",
    "chunk_nodes",
    "embed_chunks",
    "embed_chunks_async",
    "_api_embed_async",
    "upsert_embeddings",
    "batch_ingest",
    "batch_ingest_from_config",
]


def load_config(path: str | None = None) -> dict:
    """Load a YAML configuration file.

    Parameters
    ----------
    path:
        Optional path to the YAML file. Defaults to ``config.yaml`` in the
        current working directory.
    """
    config_path = Path(path or "config.yaml")
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _coerce_int(value: int | str | None, name: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def resolve_ingest_settings(
    config_path: str | None = None,
    *,
    index_path: str | None = None,
    model_name: str | None = None,
    max_tokens: int | str | None = None,
    overlap: int | str | None = None,
) -> dict:
    """Resolve ingestion settings from config and overrides."""
    cfg = load_config(config_path) if config_path else {}
    resolved_index = index_path or cfg.get("index_path")
    if not resolved_index:
        raise ValueError("index_path is required (provide CLI arg or config.yaml)")
    resolved_model = model_name or cfg.get("model") or "text-embedding-3-small"
    resolved_max_tokens = (
        _coerce_int(max_tokens, "max_tokens")
        if max_tokens is not None
        else _coerce_int(cfg.get("max_tokens", 200), "max_tokens")
    )
    resolved_overlap = (
        _coerce_int(overlap, "overlap")
        if overlap is not None
        else _coerce_int(cfg.get("overlap", 0), "overlap")
    )
    return {
        "index_path": resolved_index,
        "model_name": resolved_model,
        "max_tokens": resolved_max_tokens,
        "overlap": resolved_overlap,
    }

# ---------------------------------------------------------------------------
# 1. Document parsing
# ---------------------------------------------------------------------------

def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_markdown(text: str, doc_id: str) -> List[dict]:
    lines = text.splitlines()
    nodes: List[dict] = []
    current: List[str] = []
    buffer: List[str] = []

    def flush() -> None:
        if buffer:
            para = "\n".join(buffer).strip()
            if para:
                nodes.append({"doc_id": doc_id, "level": current.copy(), "text": para})
            buffer.clear()

    heading_re = re.compile(r"^(#{1,6})\s+(.*)")
    for line in lines:
        match = heading_re.match(line)
        if match:
            flush()
            hashes, title = match.groups()
            level = len(hashes) - 1
            current = current[:level] + [title.strip()]
        elif line.strip() == "":
            flush()
        else:
            buffer.append(line)
    flush()
    return nodes


def parse_document(path: str) -> List[dict]:
    """Read a document and return a list of nodes with hierarchy metadata."""
    file = Path(path)
    doc_id = file.stem
    ext = file.suffix.lower()

    if ext == ".txt":
        text = _read_txt(file)
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return [{"doc_id": doc_id, "level": [], "text": p} for p in parts]

    if ext == ".md":
        text = _read_txt(file)
        return _parse_markdown(text, doc_id)

    if ext == ".pdf":
        try:
            text = extract_text(str(file))
        except Exception:
            text = ""
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return [{"doc_id": doc_id, "level": [], "text": p} for p in parts]

    raise ValueError(f"Unsupported file type: {ext}")


# ---------------------------------------------------------------------------
# 2. Chunking
# ---------------------------------------------------------------------------

def _split_text(text: str, max_tokens: int, overlap: int) -> List[str]:
    if max_tokens <= 0:
        return [text] if text else []
    overlap = max(0, min(overlap, max_tokens - 1)) if max_tokens > 1 else 0
    step = max_tokens - overlap if max_tokens > overlap else max_tokens
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        if not tokens:
            return []
        chunks: List[str] = []
        for start in range(0, len(tokens), step):
            chunk_tokens = tokens[start:start + max_tokens]
            if chunk_tokens:
                chunks.append(encoding.decode(chunk_tokens))
        return chunks
    except Exception:
        words = text.split()
        if not words:
            return []
        chunks = []
        for start in range(0, len(words), step):
            chunk_words = words[start:start + max_tokens]
            if chunk_words:
                chunks.append(" ".join(chunk_words))
        return chunks


def chunk_nodes(nodes: List[dict], max_tokens: int, overlap: int = 0) -> List[dict]:
    """Split node texts into token-bounded chunks preserving order."""

    chunks: List[dict] = []
    pos = 0
    for node in nodes:
        for piece in _split_text(node["text"], max_tokens, overlap):
            chunks.append(
                {
                    "doc_id": node["doc_id"],
                    "chunk_id": pos,
                    "text": piece,
                    "level": node.get("level", []),
                    "position": pos,
                }
            )
            pos += 1
    return chunks


# ---------------------------------------------------------------------------
# 3. Embedding
# ---------------------------------------------------------------------------

def _api_embed(
    texts: List[str],
    model_name: str,
    *,
    batch_size: int | None = None,
    timeout: float | None = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> np.ndarray:
    """Embed *texts* via the OpenAI API using *model_name*."""
    import time

    import openai

    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Set the environment variable to call the embeddings API."
        )

    env_batch_size = os.getenv("OPENAI_EMBED_BATCH_SIZE")
    resolved_batch_size = (
        _coerce_int(batch_size, "batch_size")
        if batch_size is not None
        else _coerce_int(env_batch_size, "OPENAI_EMBED_BATCH_SIZE")
        if env_batch_size
        else len(texts)
    )
    if resolved_batch_size is None or resolved_batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    resolved_timeout = timeout
    if resolved_timeout is None:
        env_timeout = os.getenv("OPENAI_TIMEOUT")
        if env_timeout is not None:
            try:
                resolved_timeout = float(env_timeout)
            except ValueError as exc:
                raise ValueError("OPENAI_TIMEOUT must be a number") from exc

    client = openai.OpenAI(api_key=api_key, timeout=resolved_timeout)
    transient_errors = tuple(
        err
        for err in (
            getattr(openai, "RateLimitError", None),
            getattr(openai, "APIConnectionError", None),
            getattr(openai, "APITimeoutError", None),
            getattr(openai, "InternalServerError", None),
            getattr(openai, "APIError", None),
        )
        if err is not None
    )

    vectors: List[List[float]] = []
    for start in range(0, len(texts), resolved_batch_size):
        batch = texts[start:start + resolved_batch_size]
        attempt = 0
        while True:
            try:
                response = client.embeddings.create(model=model_name, input=batch)
                vectors.extend([item.embedding for item in response.data])
                break
            except transient_errors as exc:
                if attempt >= max_retries:
                    raise RuntimeError(
                        "OpenAI embeddings request failed after retries."
                    ) from exc
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                attempt += 1
            except Exception as exc:
                raise RuntimeError("OpenAI embeddings request failed.") from exc

    return np.asarray(vectors, dtype=np.float32)


async def _api_embed_async(
    texts: List[str],
    model_name: str,
    *,
    batch_size: int | None = None,
    timeout: float | None = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
    concurrency_limit: int = 50,
) -> np.ndarray:
    """Async version of _api_embed using AsyncOpenAI client.

    Parameters
    ----------
    texts : List[str]
        Texts to embed
    model_name : str
        OpenAI embedding model name
    batch_size : int, optional
        Number of texts per API call
    timeout : float, optional
        Request timeout in seconds
    max_retries : int
        Maximum retry attempts for transient errors
    base_delay : float
        Base delay for exponential backoff
    concurrency_limit : int
        Maximum concurrent API requests

    Returns
    -------
    np.ndarray
        Embeddings matrix (n_texts, embedding_dim)
    """
    import asyncio
    import openai
    from openai import AsyncOpenAI

    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Set the environment variable to call the embeddings API."
        )

    env_batch_size = os.getenv("OPENAI_EMBED_BATCH_SIZE")
    resolved_batch_size = (
        _coerce_int(batch_size, "batch_size")
        if batch_size is not None
        else _coerce_int(env_batch_size, "OPENAI_EMBED_BATCH_SIZE")
        if env_batch_size
        else 100  # Default batch size for async
    )
    if resolved_batch_size is None or resolved_batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    resolved_timeout = timeout
    if resolved_timeout is None:
        env_timeout = os.getenv("OPENAI_TIMEOUT")
        if env_timeout is not None:
            try:
                resolved_timeout = float(env_timeout)
            except ValueError as exc:
                raise ValueError("OPENAI_TIMEOUT must be a number") from exc

    client = AsyncOpenAI(api_key=api_key, timeout=resolved_timeout)
    semaphore = asyncio.Semaphore(concurrency_limit)

    transient_errors = tuple(
        err
        for err in (
            getattr(openai, "RateLimitError", None),
            getattr(openai, "APIConnectionError", None),
            getattr(openai, "APITimeoutError", None),
            getattr(openai, "InternalServerError", None),
            getattr(openai, "APIError", None),
        )
        if err is not None
    )

    async def embed_batch(batch: List[str], batch_idx: int) -> tuple[int, List[List[float]]]:
        """Embed a single batch with retry logic."""
        async with semaphore:
            for attempt in range(max_retries):
                try:
                    response = await client.embeddings.create(model=model_name, input=batch)
                    return batch_idx, [item.embedding for item in response.data]
                except transient_errors as exc:
                    if attempt >= max_retries - 1:
                        raise RuntimeError(
                            "OpenAI embeddings request failed after retries."
                        ) from exc
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                except Exception as exc:
                    raise RuntimeError("OpenAI embeddings request failed.") from exc
        return batch_idx, []  # Should not reach here

    # Create batches
    batches = []
    for i in range(0, len(texts), resolved_batch_size):
        batches.append(texts[i:i + resolved_batch_size])

    # Execute all batches concurrently
    tasks = [embed_batch(batch, idx) for idx, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks)

    # Sort by batch index and flatten
    results.sort(key=lambda x: x[0])
    vectors = []
    for _, batch_vectors in results:
        vectors.extend(batch_vectors)

    return np.asarray(vectors, dtype=np.float32)


async def embed_chunks_async(chunks: List[dict], model_name: str) -> np.ndarray:
    """Async version of embed_chunks using AsyncOpenAI client.

    Parameters
    ----------
    chunks : List[dict]
        Chunks to embed (must have 'text' key)
    model_name : str
        OpenAI embedding model name

    Returns
    -------
    np.ndarray
        Normalized embeddings matrix
    """
    texts = [c["text"] for c in chunks]
    embeds = await _api_embed_async(texts, model_name)
    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    np.divide(embeds, norms, out=embeds, where=norms != 0)
    return embeds


def embed_chunks(chunks: List[dict], model_name: str) -> np.ndarray:
    """Embed chunk texts using the OpenAI embeddings API."""
    texts = [c["text"] for c in chunks]
    embeds = _api_embed(texts, model_name)
    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    np.divide(embeds, norms, out=embeds, where=norms != 0)
    return embeds


# ---------------------------------------------------------------------------
# 4. Upsert embeddings
# ---------------------------------------------------------------------------

def upsert_embeddings(index_path: str, chunks: List[dict], embeds: np.ndarray) -> None:
    """Store embeddings and metadata into a FAISS index on disk."""
    if len(chunks) == 0 or embeds.size == 0:
        return

    index_file = Path(index_path)
    meta_file = Path(index_path + ".meta")
    embed_file = Path(index_path + ".npy")

    if index_file.exists():
        index = faiss.read_index(str(index_file))
        start = index.ntotal
    else:
        dim = embeds.shape[1]
        index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        start = 0

    ids = np.arange(start, start + len(chunks)).astype(np.int64)
    index.add_with_ids(embeds, ids)
    faiss.write_index(index, str(index_file))

    with meta_file.open("a", encoding="utf-8") as fh:
        for meta in chunks:
            record = {
                "doc_id": meta.get("doc_id"),
                "chunk_id": int(meta.get("chunk_id", 0)),
                "level": meta.get("level", []),
                "position": int(meta.get("position", 0)),
                "text": meta.get("text", ""),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    if embed_file.exists():
        existing = np.load(embed_file)
        if existing.ndim != 2 or existing.shape[1] != embeds.shape[1]:
            raise ValueError("Embedding store shape does not match new embeddings")
        combined = np.vstack([existing, embeds])
    else:
        combined = embeds
    np.save(embed_file, combined)


# ---------------------------------------------------------------------------
# 5. Batch ingestion
# ---------------------------------------------------------------------------

def batch_ingest(
    input_folder: str,
    index_path: str,
    model_name: str,
    max_tokens: int,
    overlap: int = 0,
) -> None:
    """Process all supported files in *input_folder* end-to-end."""
    folder = Path(input_folder)
    for file in folder.iterdir():
        if file.suffix.lower() not in {".txt", ".md", ".pdf"}:
            continue
        try:
            nodes = parse_document(str(file))
            chunks = chunk_nodes(nodes, max_tokens, overlap)
            if not chunks:
                continue
            embeds = embed_chunks(chunks, model_name)
            if embeds.size == 0:
                continue
            upsert_embeddings(index_path, chunks, embeds)
        except Exception as exc:  # pragma: no cover
            print(f"Warning: failed to ingest {file}: {exc}")
            continue


def batch_ingest_from_config(
    input_folder: str,
    config_path: str | None = None,
    *,
    index_path: str | None = None,
    model_name: str | None = None,
    max_tokens: int | str | None = None,
    overlap: int | str | None = None,
) -> dict:
    """Run batch ingestion using config defaults with optional overrides."""
    settings = resolve_ingest_settings(
        config_path,
        index_path=index_path,
        model_name=model_name,
        max_tokens=max_tokens,
        overlap=overlap,
    )
    batch_ingest(
        input_folder,
        settings["index_path"],
        settings["model_name"],
        settings["max_tokens"],
        settings["overlap"],
    )
    return settings


def _cli_main() -> None:
    """CLI entry point for batch ingestion."""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest documents into a FAISS index")
    parser.add_argument("input", help="Input folder containing documents")
    parser.add_argument("index", nargs="?", default=None, help="Output FAISS index path")
    parser.add_argument("--model", default=None, help="Embedding model name")
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--config", help="Path to config.yaml", default=None)
    args = parser.parse_args()

    settings = resolve_ingest_settings(
        args.config,
        index_path=args.index,
        model_name=args.model,
        max_tokens=args.max_tokens,
        overlap=args.overlap,
    )

    Path(settings["index_path"]).parent.mkdir(parents=True, exist_ok=True)
    batch_ingest(
        args.input,
        settings["index_path"],
        settings["model_name"],
        settings["max_tokens"],
        settings["overlap"],
    )
    print(f"Index written to {settings['index_path']}")


if __name__ == "__main__":
    _cli_main()
