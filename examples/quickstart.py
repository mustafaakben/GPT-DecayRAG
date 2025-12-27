"""Simple ingestion example for DecayRAG."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from decayrag import batch_ingest, resolve_ingest_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a folder of documents")
    parser.add_argument("input", help="Input folder containing documents")
    parser.add_argument("index", nargs="?", default=None, help="Output FAISS index path")
    parser.add_argument("--model", default=None, help="Embedding model name")
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--config", help="Path to config.yaml", default=None)
    args = parser.parse_args()

    settings = resolve_ingest_settings(
        args.config,
        index_path=args.index,
        model_name=args.model,
        max_tokens=args.max_tokens,
        overlap=args.overlap,
    )

    Path(settings["index_path"]).parent.mkdir(parents=True, exist_ok=True)
    batch_ingest(
        args.input,
        settings["index_path"],
        settings["model_name"],
        settings["max_tokens"],
        settings["overlap"],
    )
    print(f"Index written to {settings['index_path']}")


if __name__ == "__main__":
    main()

