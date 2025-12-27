"""Generate an answer by retrieving context with DecayRAG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from decayrag import assemble_context, generate_answer, load_config, retrieve


def _load_metadata(index_path: str) -> list[dict]:
    meta_path = Path(index_path + ".meta")
    with meta_path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
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
    main()
