"""Post-retrieval utilities for assembling context and generating answers."""

from __future__ import annotations

import os
import argparse
import json
from pathlib import Path
from typing import List

import openai

from .ingest import load_config
from .retrieval import retrieve

__all__ = ["assemble_context", "generate_answer"]


def assemble_context(chunks: List[dict], window: int, all_chunks: List[dict] | None = None) -> str:
    """Expand chunks with neighbouring context and join their text.

    Parameters
    ----------
    chunks:
        List of chunk dictionaries containing ``text``, ``position``, and
        ``doc_id`` keys for the top retrieval results.
    window:
        Number of neighbouring positions to include on each side of a result.
    all_chunks:
        Optional list of all chunk dictionaries to source neighbours from. When
        omitted, neighbours are searched only within ``chunks``.
    """
    if not chunks:
        return ""

    candidates = all_chunks if all_chunks is not None else chunks
    doc_lookup: dict[str, dict[int, dict]] = {}
    for candidate in candidates:
        doc_id = str(candidate.get("doc_id", ""))
        position = int(candidate.get("position", 0))
        doc_lookup.setdefault(doc_id, {})[position] = candidate

    expanded: dict[tuple[str, int], dict] = {}
    for chunk in chunks:
        doc_id = str(chunk.get("doc_id", ""))
        position = int(chunk.get("position", 0))
        for neighbor_pos in range(position - window, position + window + 1):
            neighbor = doc_lookup.get(doc_id, {}).get(neighbor_pos)
            if neighbor is not None:
                expanded[(doc_id, neighbor_pos)] = neighbor

    ordered = sorted(
        expanded.values(),
        key=lambda c: (str(c.get("doc_id", "")), int(c.get("position", 0))),
    )
    texts = [c.get("text", "") for c in ordered]
    return "\n".join(texts)


def generate_answer(
    context: str,
    query: str,
    model: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 512,
    system_prompt: str = "You are a helpful assistant.",
) -> str:
    """Call an LLM to produce an answer based on *context* and *query*."""
    if not context or not context.strip():
        raise ValueError("Context is empty; provide retrieved context before generating an answer.")
    if not query or not query.strip():
        raise ValueError("Query is empty; provide a question before generating an answer.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:",
            },
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def _load_metadata(index_path: str) -> List[dict]:
    meta_path = Path(index_path + ".meta")
    with meta_path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _run_cli() -> None:
    parser = argparse.ArgumentParser(description="Retrieve context and generate an answer")
    parser.add_argument("query", help="Query string to answer")
    parser.add_argument("--index_path", default=None, help="Path to FAISS index")
    parser.add_argument("--top_k", type=int, default=None, help="Number of chunks to retrieve")
    parser.add_argument("--window", type=int, default=None, help="Neighbour window size")
    parser.add_argument("--model", default=None, help="Chat model name")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--system_prompt", default=None)
    parser.add_argument("--config", help="Path to config.yaml", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    index_path = args.index_path or cfg.get("index_path")
    if not index_path:
        raise ValueError("index_path is required (provide --index_path or config.yaml)")

    embedding_model = cfg.get("model", "text-embedding-3-small")
    top_k = args.top_k if args.top_k is not None else int(cfg.get("top_k", 5))
    window = args.window if args.window is not None else int(cfg.get("window", 1))
    model = args.model or cfg.get("answer_model", "gpt-4o-mini")
    temperature = (
        args.temperature if args.temperature is not None else float(cfg.get("answer_temperature", 0.2))
    )
    max_tokens = args.max_tokens if args.max_tokens is not None else int(cfg.get("answer_max_tokens", 512))
    system_prompt = args.system_prompt or cfg.get(
        "answer_system_prompt",
        "You are a helpful assistant.",
    )

    chunks = retrieve(
        args.query,
        index_path,
        model=embedding_model,
        top_k=top_k,
    )
    all_chunks = _load_metadata(index_path)
    context = assemble_context(chunks, window, all_chunks=all_chunks)

    answer = generate_answer(
        context,
        args.query,
        model,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )
    print(answer)


if __name__ == "__main__":
    _run_cli()
