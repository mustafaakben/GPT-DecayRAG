from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decayrag import assemble_context


def test_assemble_context_orders_by_doc_and_position():
    all_chunks = [
        {"doc_id": "doc-1", "text": "A1", "position": 0},
        {"doc_id": "doc-1", "text": "A2", "position": 1},
        {"doc_id": "doc-2", "text": "B1", "position": 0},
        {"doc_id": "doc-2", "text": "B2", "position": 1},
    ]
    chunks = [
        {"doc_id": "doc-2", "text": "B1", "position": 0},
        {"doc_id": "doc-1", "text": "A2", "position": 1},
    ]

    ctx = assemble_context(chunks, window=1, all_chunks=all_chunks)

    assert ctx.splitlines() == ["A1", "A2", "B1", "B2"]
