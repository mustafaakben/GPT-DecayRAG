import os
from pathlib import Path

import numpy as np
import faiss
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decayrag import (
    parse_document,
    chunk_nodes,
    embed_chunks,
    upsert_embeddings,
    batch_ingest,
)


def test_parse_document_txt(tmp_path: Path) -> None:
    txt = tmp_path / "doc.txt"
    txt.write_text("para1\n\npara2")
    nodes = parse_document(str(txt))
    assert len(nodes) == 2
    assert nodes[0]["text"] == "para1"
    assert nodes[0]["level"] == []


def test_parse_document_md(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text("# Title\n\nIntro text\n\n## Section\n\nMore text")
    nodes = parse_document(str(md))
    assert nodes[0]["level"] == ["Title"]
    assert nodes[1]["level"] == ["Title", "Section"]


def test_chunk_nodes_and_embed(tmp_path: Path) -> None:
    nodes = [{"doc_id": "x", "level": [], "text": "word " * 50}]
    chunks = chunk_nodes(nodes, max_tokens=20, overlap=5)
    assert len(chunks) > 0
    embeds = embed_chunks(chunks, "text-embedding-3-small")
    assert embeds.shape[0] == len(chunks)
    assert embeds.shape[1] > 0
    norms = np.linalg.norm(embeds, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_upsert_and_batch_ingest(tmp_path: Path) -> None:
    index_path = tmp_path / "index.faiss"
    chunks = [
        {"doc_id": "d", "chunk_id": 0, "text": "hello", "level": [], "position": 0}
    ]
    embeds = np.random.random((1, 384)).astype(np.float32)
    upsert_embeddings(str(index_path), chunks, embeds)
    assert index_path.exists()
    meta = Path(str(index_path) + ".meta")
    assert meta.exists()
    lines = meta.read_text().strip().splitlines()
    assert len(lines) == 1
    # ingest via batch_ingest
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.txt").write_text("text")
    batch_ingest(str(folder), str(index_path), "text-embedding-3-small", 10)
    meta_lines = Path(str(index_path) + ".meta").read_text().strip().splitlines()
    idx = faiss.read_index(str(index_path))
    assert idx.ntotal == len(meta_lines)

