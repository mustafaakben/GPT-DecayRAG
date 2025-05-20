"""Simple ingestion example for DecayRAG."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from decayrag import batch_ingest


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a folder of documents")
    parser.add_argument("input", help="Input folder containing documents")
    parser.add_argument("index", help="Output FAISS index path")
    parser.add_argument("--model", default="text-embedding-3-small", help="Embedding model name")
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--overlap", type=int, default=0)
    args = parser.parse_args()

    Path(args.index).parent.mkdir(parents=True, exist_ok=True)
    batch_ingest(args.input, args.index, args.model, args.max_tokens, args.overlap)
    print(f"Index written to {args.index}")


if __name__ == "__main__":
    main()


