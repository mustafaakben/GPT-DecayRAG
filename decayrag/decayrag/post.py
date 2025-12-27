"""Post-retrieval utilities for assembling context and generating answers."""

from __future__ import annotations

import os
from typing import List

import openai

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
