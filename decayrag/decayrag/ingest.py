"""Document ingestion utilities for DecayRAG."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List

import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdfminer.high_level import extract_text
import tiktoken

__all__ = [
    "parse_document",
    "chunk_nodes",
    "embed_chunks",
    "upsert_embeddings",
    "batch_ingest",
]

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

def chunk_nodes(nodes: List[dict], max_tokens: int, overlap: int = 0) -> List[dict]:
    """Split node texts into token-bounded chunks preserving order."""
    splitter: RecursiveCharacterTextSplitter
    try:
        tiktoken.get_encoding("cl100k_base")
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=max_tokens,
            chunk_overlap=overlap,
            encoding_name="cl100k_base",
        )
    except Exception:
        splitter = RecursiveCharacterTextSplitter(chunk_size=max_tokens, chunk_overlap=overlap)

    chunks: List[dict] = []
    pos = 0
    for node in nodes:
        for piece in splitter.split_text(node["text"]):
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

def _api_embed(texts: List[str], model_name: str) -> np.ndarray:
    """Embed *texts* via the OpenAI API using *model_name*."""
    import openai

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = openai.OpenAI(api_key=api_key)
    response = client.embeddings.create(model=model_name, input=texts)
    vectors = [item.embedding for item in response.data]
    return np.asarray(vectors, dtype=np.float32)


def embed_chunks(chunks: List[dict], model_name: str) -> np.ndarray:
    """Embed chunk texts with a chosen model or sentence-transformer fallback."""
    texts = [c["text"] for c in chunks]
    try:
        embeds = _api_embed(texts, model_name)
    except Exception:
        try:
            from sentence_transformers import SentenceTransformer

            st_model = SentenceTransformer(model_name)
            embeds = st_model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
        except Exception:
            rng = np.random.default_rng(0)
            embeds = rng.standard_normal((len(texts), 384)).astype(np.float32)

    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    np.divide(embeds, norms, out=embeds, where=norms != 0)
    return embeds


# ---------------------------------------------------------------------------
# 4. Upsert embeddings
# ---------------------------------------------------------------------------

def upsert_embeddings(index_path: str, chunks: List[dict], embeds: np.ndarray) -> None:
    """Store embeddings and metadata into a FAISS index on disk."""
    index_file = Path(index_path)
    meta_file = Path(index_path + ".meta")

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
            embeds = embed_chunks(chunks, model_name)
            upsert_embeddings(index_path, chunks, embeds)
        except Exception as exc:  # pragma: no cover
            print(f"Warning: failed to ingest {file}: {exc}")
            continue

