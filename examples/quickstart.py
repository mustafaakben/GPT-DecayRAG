"""Simple ingestion example for DecayRAG."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from decayrag import batch_ingest, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a folder of documents")
    parser.add_argument("input", help="Input folder containing documents")
    parser.add_argument("index", help="Output FAISS index path")
    parser.add_argument("--model", default="text-embedding-3-small", help="Embedding model name")
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--config", help="Path to config.yaml", default=None)
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
        model = cfg.get("model", args.model)
        max_tokens = int(cfg.get("max_tokens", args.max_tokens))
        overlap = int(cfg.get("overlap", args.overlap))
    else:
        model = args.model
        max_tokens = args.max_tokens
        overlap = args.overlap

    Path(args.index).parent.mkdir(parents=True, exist_ok=True)
    batch_ingest(args.input, args.index, model, max_tokens, overlap)
    print(f"Index written to {args.index}")


if __name__ == "__main__":
    main()


