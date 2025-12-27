"""Generate an answer from context using DecayRAG."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from decayrag import generate_answer, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an answer from context text")
    parser.add_argument("context", help="Context text to ground the answer")
    parser.add_argument("query", help="Question to answer")
    parser.add_argument("--model", default="gpt-4o-mini", help="Chat model name")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--system_prompt", default="You are a helpful assistant.")
    parser.add_argument("--config", help="Path to config.yaml", default=None)
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
        temperature = float(cfg.get("answer_temperature", args.temperature))
        max_tokens = int(cfg.get("answer_max_tokens", args.max_tokens))
        system_prompt = str(cfg.get("answer_system_prompt", args.system_prompt))
    else:
        temperature = args.temperature
        max_tokens = args.max_tokens
        system_prompt = args.system_prompt

    answer = generate_answer(
        args.context,
        args.query,
        args.model,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )
    print(answer)


if __name__ == "__main__":
    main()
