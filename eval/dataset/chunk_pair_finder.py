"""Find bridge chunk pairs suitable for multi-hop questions."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

from openai import OpenAI


@dataclass
class ChunkPair:
    """A pair of chunks suitable for multi-hop reasoning."""

    doc_id: str
    chunk_a_pos: int
    chunk_b_pos: int
    chunk_a_text: str
    chunk_b_text: str
    connecting_entity: str
    relationship: str
    distance: int = 0

    def __post_init__(self):
        self.distance = abs(self.chunk_b_pos - self.chunk_a_pos)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ChunkPair":
        return cls(**data)


def _extract_entities_simple(text: str) -> List[str]:
    """Simple entity extraction using capitalized words."""
    # Find capitalized multi-word phrases
    pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
    entities = re.findall(pattern, text)

    # Filter common words
    stopwords = {'The', 'This', 'That', 'These', 'Those', 'However', 'Although',
                 'Because', 'While', 'When', 'Where', 'Which', 'What', 'After',
                 'Before', 'During', 'Since', 'Until', 'Also', 'Many', 'Some',
                 'Most', 'Other', 'Such', 'Each', 'Both', 'All', 'Any', 'New'}

    entities = [e for e in entities if e not in stopwords and len(e) > 2]
    return list(set(entities))


def find_bridge_chunk_pairs_simple(
    chunks: List[dict],
    doc_id: str,
    min_distance: int = 2,
    max_distance: int = 8,
    max_pairs: int = 20,
) -> List[ChunkPair]:
    """Find chunk pairs with shared entities using simple heuristics.

    Parameters
    ----------
    chunks : List[dict]
        List of chunk dictionaries with 'text' and 'position' keys
    doc_id : str
        Document identifier
    min_distance : int
        Minimum distance between chunks (default 2)
    max_distance : int
        Maximum distance between chunks (default 8)
    max_pairs : int
        Maximum number of pairs to return (default 20)

    Returns
    -------
    List[ChunkPair]
        Chunk pairs suitable for multi-hop questions
    """
    pairs = []

    # Extract entities for each chunk
    chunk_entities = []
    for chunk in chunks:
        entities = _extract_entities_simple(chunk.get("text", ""))
        chunk_entities.append(set(entities))

    # Find pairs with shared entities
    for i, chunk_a in enumerate(chunks):
        for j, chunk_b in enumerate(chunks):
            if i >= j:
                continue

            distance = abs(j - i)
            if distance < min_distance or distance > max_distance:
                continue

            # Find shared entities
            shared = chunk_entities[i] & chunk_entities[j]
            if not shared:
                continue

            # Pick the most specific shared entity (longest)
            connecting = max(shared, key=len)

            pair = ChunkPair(
                doc_id=doc_id,
                chunk_a_pos=chunk_a.get("position", i),
                chunk_b_pos=chunk_b.get("position", j),
                chunk_a_text=chunk_a.get("text", ""),
                chunk_b_text=chunk_b.get("text", ""),
                connecting_entity=connecting,
                relationship="shared_entity",
            )
            pairs.append(pair)

            if len(pairs) >= max_pairs * 2:
                break

        if len(pairs) >= max_pairs * 2:
            break

    # Sort by distance (prefer moderate distances) and limit
    pairs.sort(key=lambda p: abs(p.distance - 4))  # Prefer distance ~4
    return pairs[:max_pairs]


def find_bridge_chunk_pairs_llm(
    chunks: List[dict],
    doc_id: str,
    doc_title: str,
    min_distance: int = 2,
    max_distance: int = 8,
    max_pairs: int = 10,
    model: str = "gpt-5.2",
) -> List[ChunkPair]:
    """Find chunk pairs using LLM analysis.

    Parameters
    ----------
    chunks : List[dict]
        List of chunk dictionaries
    doc_id : str
        Document identifier
    doc_title : str
        Document title
    min_distance : int
        Minimum distance between chunks
    max_distance : int
        Maximum distance between chunks
    max_pairs : int
        Maximum number of pairs to return
    model : str
        Model to use for analysis

    Returns
    -------
    List[ChunkPair]
        Chunk pairs identified by LLM
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI()

    # Prepare chunk summaries (limit to avoid token limits)
    chunk_summaries = []
    for i, chunk in enumerate(chunks[:50]):  # Limit chunks
        text = chunk.get("text", "")[:300]  # Truncate
        pos = chunk.get("position", i)
        chunk_summaries.append(f"[{pos}] {text}")

    chunks_text = "\n\n".join(chunk_summaries)

    prompt = f"""Analyze these chunks from the Wikipedia article "{doc_title}" to find pairs suitable for multi-hop reasoning questions.

CHUNKS:
{chunks_text}

Find pairs of chunks (distance {min_distance}-{max_distance} apart) where:
1. They share a connecting entity, concept, or causal relationship
2. Information from BOTH is needed to answer a non-trivial question
3. Neither chunk alone is sufficient

Return a JSON array with up to {max_pairs} pairs:
[
  {{
    "chunk_a_pos": <int>,
    "chunk_b_pos": <int>,
    "connecting_entity": "<string>",
    "relationship": "<brief description>"
  }}
]

Only return valid JSON array, no other text."""

    try:
        # Retry up to 3 times for API failures
        result_text = ""
        for attempt in range(3):
            response = client.responses.create(
                model=model,
                input=prompt,
                reasoning={"effort": "medium"},
                max_output_tokens=2048,
            )
            result_text = response.output_text.strip()
            if result_text:
                break
            import time
            time.sleep(1)

        if not result_text:
            print(f"Empty response after 3 attempts in chunk pair finder")
            return []

        # Extract JSON from response
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]

        pairs_data = json.loads(result_text)

        pairs = []
        for p in pairs_data:
            pos_a = int(p["chunk_a_pos"])
            pos_b = int(p["chunk_b_pos"])

            # Validate positions
            if pos_a >= len(chunks) or pos_b >= len(chunks):
                continue

            distance = abs(pos_b - pos_a)
            if distance < min_distance or distance > max_distance:
                continue

            pair = ChunkPair(
                doc_id=doc_id,
                chunk_a_pos=pos_a,
                chunk_b_pos=pos_b,
                chunk_a_text=chunks[pos_a].get("text", ""),
                chunk_b_text=chunks[pos_b].get("text", ""),
                connecting_entity=p.get("connecting_entity", ""),
                relationship=p.get("relationship", ""),
            )
            pairs.append(pair)

        return pairs[:max_pairs]

    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        return []
    except Exception as e:
        print(f"LLM pair finding error: {e}")
        return []


def find_bridge_chunk_pairs(
    chunks: List[dict],
    doc_id: str,
    doc_title: str = "",
    min_distance: int = 2,
    max_distance: int = 8,
    max_pairs: int = 15,
    use_llm: bool = True,
    model: str = "gpt-5.2",
) -> List[ChunkPair]:
    """Find bridge chunk pairs using hybrid approach.

    Uses simple heuristics first, then optionally validates/enriches with LLM.

    Parameters
    ----------
    chunks : List[dict]
        List of chunk dictionaries
    doc_id : str
        Document identifier
    doc_title : str
        Document title (for LLM context)
    min_distance : int
        Minimum distance between chunks
    max_distance : int
        Maximum distance between chunks
    max_pairs : int
        Maximum number of pairs to return
    use_llm : bool
        Whether to use LLM for pair finding
    model : str
        Model to use for LLM analysis

    Returns
    -------
    List[ChunkPair]
        Chunk pairs suitable for multi-hop questions
    """
    if use_llm and os.getenv("OPENAI_API_KEY"):
        # Use LLM for better quality pairs
        llm_pairs = find_bridge_chunk_pairs_llm(
            chunks, doc_id, doc_title,
            min_distance, max_distance, max_pairs, model
        )
        if llm_pairs:
            return llm_pairs

    # Fallback to simple heuristics
    return find_bridge_chunk_pairs_simple(
        chunks, doc_id, min_distance, max_distance, max_pairs
    )


if __name__ == "__main__":
    # Test with sample chunks
    sample_chunks = [
        {"position": 0, "text": "Albert Einstein was born in Ulm, Germany in 1879."},
        {"position": 1, "text": "He showed early aptitude for mathematics and physics."},
        {"position": 2, "text": "Einstein moved to Switzerland in 1895."},
        {"position": 3, "text": "He worked at the Swiss Patent Office while developing his theories."},
        {"position": 4, "text": "In 1905, he published four groundbreaking papers."},
        {"position": 5, "text": "The special theory of relativity was introduced that year."},
        {"position": 6, "text": "Einstein received the Nobel Prize in 1921 for the photoelectric effect."},
        {"position": 7, "text": "He emigrated to the United States in 1933."},
        {"position": 8, "text": "Einstein spent his later years at Princeton University."},
        {"position": 9, "text": "He died in 1955 in New Jersey."},
    ]

    # Test simple method
    pairs = find_bridge_chunk_pairs_simple(sample_chunks, "einstein", min_distance=2, max_distance=5)
    print(f"Found {len(pairs)} pairs using simple method:")
    for p in pairs[:3]:
        print(f"  [{p.chunk_a_pos}] <-> [{p.chunk_b_pos}]: {p.connecting_entity}")
