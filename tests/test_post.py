from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decayrag import assemble_context, generate_answer
import openai


def test_assemble_context_sorting():
    chunks = [
        {"text": "B", "position": 1},
        {"text": "A", "position": 0},
        {"text": "C", "position": 2},
    ]
    ctx = assemble_context(chunks, window=1)
    assert ctx.splitlines() == ["A", "B", "C"]


def test_generate_answer(monkeypatch):
    calls = {}

    class FakeCompletions:
        def create(self, model, messages):
            calls["model"] = model
            calls["messages"] = messages
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
