"""Post-retrieval utilities for assembling context and generating answers."""

from __future__ import annotations

import os
from typing import List

import openai

__all__ = ["assemble_context", "generate_answer"]


def assemble_context(chunks: List[dict], window: int) -> str:
    """Order chunks by position and join their text.

    Parameters
    ----------
    chunks:
        List of chunk dictionaries containing ``text`` and ``position`` keys.
    window:
        Unused placeholder for future neighbour-expansion; kept for API
        compatibility.
    """
    if not chunks:
        return ""
    ordered = sorted(chunks, key=lambda c: int(c.get("position", 0)))
    texts = [c.get("text", "") for c in ordered]
    return "\n".join(texts)


def generate_answer(context: str, query: str, model: str) -> str:
    """Call an LLM to produce an answer based on *context* and *query*."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:",
            },
        ],
    )
    return response.choices[0].message.content.strip()
