from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decayrag import assemble_context, generate_answer
import openai


def test_assemble_context_orders_windowed_neighbors():
    all_chunks = [
        {"doc_id": "doc-1", "text": "A0", "position": 0},
        {"doc_id": "doc-1", "text": "A1", "position": 1},
        {"doc_id": "doc-1", "text": "A2", "position": 2},
        {"doc_id": "doc-2", "text": "B0", "position": 0},
        {"doc_id": "doc-2", "text": "B1", "position": 1},
        {"doc_id": "doc-2", "text": "B2", "position": 2},
    ]
    chunks = [
        {"doc_id": "doc-2", "text": "B1", "position": 1},
        {"doc_id": "doc-1", "text": "A1", "position": 1},
    ]
    ctx = assemble_context(chunks, window=1, all_chunks=all_chunks)
    assert ctx.splitlines() == ["A0", "A1", "A2", "B0", "B1", "B2"]


def test_assemble_context_empty_input_returns_empty_string():
    assert assemble_context([], window=1, all_chunks=[]) == ""


def test_generate_answer(monkeypatch):
    calls = {}

    class FakeCompletions:
        def create(self, model, messages, temperature, max_tokens):
            calls["model"] = model
            calls["messages"] = messages
            calls["temperature"] = temperature
            calls["max_tokens"] = max_tokens
            class Res:
                choices = [
                    type("obj", (object,), {"message": type("m", (object,), {"content": "Answer"})()})
                ]
            return Res()

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self, api_key):
            calls["api_key"] = api_key
            self.chat = FakeChat()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    out = generate_answer("ctx", "q", "gpt-test")
    assert out == "Answer"
    assert calls["model"] == "gpt-test"
    assert calls["temperature"] == 0.2
    assert calls["max_tokens"] == 512
    assert calls["messages"] == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Context:\nctx\n\nQuestion: q\nAnswer:"},
    ]


def test_generate_answer_requires_context_and_query():
    try:
        generate_answer("", "q", "gpt-test")
    except ValueError as exc:
        assert "Context is empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty context")

    try:
        generate_answer("ctx", "  ", "gpt-test")
    except ValueError as exc:
        assert "Query is empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty query")


def test_assemble_context_window_expansion():
    all_chunks = [
        {"doc_id": "doc-1", "text": "A", "position": 0},
        {"doc_id": "doc-1", "text": "B", "position": 1},
        {"doc_id": "doc-1", "text": "C", "position": 2},
        {"doc_id": "doc-1", "text": "D", "position": 3},
        {"doc_id": "doc-1", "text": "E", "position": 4},
        {"doc_id": "doc-2", "text": "X", "position": 0},
    ]
    chunks = [
        {"doc_id": "doc-1", "text": "C", "position": 2},
    ]
    ctx = assemble_context(chunks, window=2, all_chunks=all_chunks)
    assert ctx.splitlines() == ["A", "B", "C", "D", "E"]
