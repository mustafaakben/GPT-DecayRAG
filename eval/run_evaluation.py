"""Main evaluation harness for DecayRAG multi-hop retrieval benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Callable

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from decayrag import (
    chunk_nodes,
    parse_document,
    retrieve,
    embed_chunks,
    upsert_embeddings,
)
from decayrag.decayrag.ingest import embed_chunks_async

from eval.dataset.wikipedia_loader import (
    download_wikipedia_articles,
    download_wikipedia_articles_async,
    save_articles_as_text,
    WikipediaArticle,
)
from eval.dataset.chunk_pair_finder import (
    find_bridge_chunk_pairs,
    find_all_bridge_chunk_pairs_async,
    ChunkPair,
)
from eval.dataset.question_generator import (
    generate_multihop_questions,
    generate_multihop_questions_async,
    save_questions,
    load_questions,
    MultiHopQuestion,
)
from eval.baselines import (
    NaiveRetriever,
    BM25Retriever,
    SentenceWindowRetriever,
    HybridRetriever,
)
from eval.metrics import recall_at_k, pair_recall_at_k, mrr, ndcg_at_k


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    # Dataset
    articles: List[str] = None
    min_word_count: int = 3000
    max_tokens: int = 200
    overlap: int = 20

    # Question generation
    questions_per_article: int = 5
    min_chunk_distance: int = 2
    max_chunk_distance: int = 8
    validate_questions: bool = True

    # Retrieval
    top_k_values: List[int] = None
    embedding_model: str = "text-embedding-3-small"

    # Paths
    cache_dir: str = "data/eval_cache"
    output_dir: str = "data/eval_results"

    def __post_init__(self):
        if self.articles is None:
            self.articles = [
                "Artificial_intelligence",
                "Albert_Einstein",
                "Climate_change",
            ]
        if self.top_k_values is None:
            self.top_k_values = [3, 5, 10, 20]


@dataclass
class EvalResult:
    """Results for a single question."""

    question_id: int
    question: str
    doc_id: str
    target_chunks: List[int]
    method: str
    retrieved_positions: List[int]
    metrics: Dict[str, float]


def prepare_dataset(config: EvalConfig) -> tuple:
    """Prepare evaluation dataset.

    Returns
    -------
    tuple
        (articles, all_chunks, questions)
    """
    print("=" * 60)
    print("PHASE 1: Dataset Preparation")
    print("=" * 60)

    # Download articles
    print("\n1.1 Downloading Wikipedia articles...")
    articles = download_wikipedia_articles(
        titles=config.articles,
        min_word_count=config.min_word_count,
        cache_dir=config.cache_dir,
    )

    if not articles:
        raise RuntimeError("No articles downloaded!")

    print(f"  Downloaded {len(articles)} articles")

    # Save as text files
    docs_dir = Path(config.cache_dir) / "docs"
    save_articles_as_text(articles, str(docs_dir))

    # Parse and chunk
    print("\n1.2 Parsing and chunking documents...")
    all_chunks = []
    doc_chunks_map = {}

    for article in articles:
        # Create text file path
        safe_title = article.title.replace(' ', '_')
        doc_path = docs_dir / f"{safe_title}.txt"

        if not doc_path.exists():
            print(f"  Warning: {doc_path} not found, skipping")
            continue

        # Parse and chunk
        nodes = parse_document(str(doc_path))
        chunks = chunk_nodes(nodes, config.max_tokens, config.overlap)

        # Add doc_id
        for chunk in chunks:
            chunk["doc_id"] = article.title

        doc_chunks_map[article.title] = chunks
        all_chunks.extend(chunks)
        print(f"  {article.title}: {len(chunks)} chunks")

    print(f"  Total: {len(all_chunks)} chunks")

    # Find chunk pairs and generate questions
    print("\n1.3 Finding chunk pairs and generating questions...")
    all_pairs = []
    for doc_id, chunks in doc_chunks_map.items():
        pairs = find_bridge_chunk_pairs(
            chunks=chunks,
            doc_id=doc_id,
            doc_title=doc_id,
            min_distance=config.min_chunk_distance,
            max_distance=config.max_chunk_distance,
            max_pairs=config.questions_per_article * 2,
            use_llm=True,
        )
        all_pairs.extend(pairs)
        print(f"  {doc_id}: {len(pairs)} chunk pairs")

    # Generate questions
    print("\n1.4 Generating multi-hop questions...")
    questions = generate_multihop_questions(
        chunk_pairs=all_pairs,
        model="gpt-5.2",
        validate=config.validate_questions,
        max_questions=config.questions_per_article * len(articles),
    )

    # Save questions
    questions_path = Path(config.cache_dir) / "questions.json"
    save_questions(questions, str(questions_path))

    print(f"\n  Generated {len(questions)} validated questions")

    return articles, all_chunks, questions, doc_chunks_map


def build_retrievers(
    all_chunks: List[dict],
    config: EvalConfig,
) -> Dict[str, object]:
    """Build all retriever instances.

    Returns
    -------
    Dict[str, retriever]
        Dictionary mapping method names to retriever instances
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Building Retrievers")
    print("=" * 60)

    retrievers = {}

    # BM25 (no API calls needed)
    print("\n2.1 Building BM25 retriever...")
    bm25 = BM25Retriever()
    bm25.index_chunks(all_chunks)
    retrievers["BM25"] = bm25
    print("  ✓ BM25 ready")

    # Naive (dense, no decay)
    print("\n2.2 Building Naive retriever...")
    naive = NaiveRetriever(model=config.embedding_model)
    naive.index_chunks(all_chunks)
    retrievers["Naive"] = naive
    print("  ✓ Naive ready")

    # Sentence Window
    print("\n2.3 Building Sentence Window retriever...")
    sentence_window = SentenceWindowRetriever(
        model=config.embedding_model,
        window_size=1,
    )
    sentence_window.chunks = all_chunks
    sentence_window.embeddings = naive.embeddings  # Reuse embeddings
    sentence_window.index = naive.index
    # Build doc_chunks
    sentence_window.doc_chunks = {}
    for chunk in all_chunks:
        doc_id = chunk.get("doc_id", "")
        if doc_id not in sentence_window.doc_chunks:
            sentence_window.doc_chunks[doc_id] = []
        sentence_window.doc_chunks[doc_id].append(chunk)
    for doc_id in sentence_window.doc_chunks:
        sentence_window.doc_chunks[doc_id].sort(key=lambda x: x.get("position", 0))
    retrievers["SentenceWindow"] = sentence_window
    print("  ✓ Sentence Window ready")

    # Hybrid
    print("\n2.4 Building Hybrid retriever...")
    hybrid = HybridRetriever(model=config.embedding_model)
    hybrid.chunks = all_chunks
    hybrid.dense_retriever = naive
    hybrid.sparse_retriever = bm25
    retrievers["Hybrid"] = hybrid
    print("  ✓ Hybrid ready")

    # DecayRAG will use the main retrieve function
    retrievers["DecayRAG"] = None  # Placeholder, handled specially

    return retrievers


def run_evaluation(
    questions: List[MultiHopQuestion],
    all_chunks: List[dict],
    retrievers: Dict[str, object],
    config: EvalConfig,
    index_path: str,
) -> List[EvalResult]:
    """Run evaluation on all questions.

    Returns
    -------
    List[EvalResult]
        Results for all question-method combinations
    """
    print("\n" + "=" * 60)
    print("PHASE 3: Running Evaluation")
    print("=" * 60)

    results = []
    max_k = max(config.top_k_values)

    for q_idx, question in enumerate(questions):
        print(f"\n[{q_idx + 1}/{len(questions)}] {question.question[:60]}...")
        targets = question.target_chunks

        for method_name, retriever in retrievers.items():
            # Get retrieved chunks
            if method_name == "DecayRAG":
                # Use main DecayRAG retrieve function
                retrieved = retrieve(
                    question.question,
                    index_path,
                    model=config.embedding_model,
                    top_k=max_k,
                    decay=True,
                    blend=True,
                )
            elif method_name == "SentenceWindow":
                # Use base retrieval without expansion for fair comparison
                retrieved = retriever.retrieve_base_only(question.question, top_k=max_k)
            else:
                retrieved = retriever.retrieve(question.question, top_k=max_k)

            # Extract positions
            retrieved_positions = [r.get("position", -1) for r in retrieved]

            # Compute metrics
            metrics = {}
            for k in config.top_k_values:
                metrics[f"recall@{k}"] = recall_at_k(retrieved, targets, k)
                metrics[f"pair_recall@{k}"] = pair_recall_at_k(
                    retrieved, targets[0], targets[1], k
                )
                metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved, targets, k)

            metrics["mrr"] = mrr(retrieved, targets)

            result = EvalResult(
                question_id=q_idx,
                question=question.question,
                doc_id=question.doc_id,
                target_chunks=targets,
                method=method_name,
                retrieved_positions=retrieved_positions[:10],
                metrics=metrics,
            )
            results.append(result)

        # Print progress
        print(f"  Targets: {targets}")
        for method_name in retrievers.keys():
            method_results = [r for r in results
                            if r.question_id == q_idx and r.method == method_name]
            if method_results:
                pr5 = method_results[0].metrics.get("pair_recall@5", 0)
                print(f"    {method_name}: Pair Recall@5 = {pr5:.2f}")

    return results


def aggregate_results(results: List[EvalResult], config: EvalConfig) -> Dict:
    """Aggregate results by method.

    Returns
    -------
    Dict
        Aggregated metrics per method
    """
    print("\n" + "=" * 60)
    print("PHASE 4: Aggregating Results")
    print("=" * 60)

    # Group by method
    method_results = {}
    for result in results:
        if result.method not in method_results:
            method_results[result.method] = []
        method_results[result.method].append(result)

    # Compute averages
    aggregated = {}
    for method, method_result_list in method_results.items():
        metrics_sum = {}
        for result in method_result_list:
            for metric_name, value in result.metrics.items():
                if metric_name not in metrics_sum:
                    metrics_sum[metric_name] = 0.0
                metrics_sum[metric_name] += value

        n = len(method_result_list)
        aggregated[method] = {
            metric: value / n for metric, value in metrics_sum.items()
        }
        aggregated[method]["n_questions"] = n

    return aggregated


def print_results_table(aggregated: Dict, config: EvalConfig) -> None:
    """Print results as a formatted table."""
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    # Header
    k_vals = config.top_k_values[:3]  # Show top 3 K values
    header = f"{'Method':<18}"
    for k in k_vals:
        header += f" {'Recall@' + str(k):<10} {'PairR@' + str(k):<10}"
    header += f" {'MRR':<8}"
    print(header)
    print("-" * 80)

    # Rows
    methods = ["BM25", "Naive", "SentenceWindow", "Hybrid", "DecayRAG"]
    for method in methods:
        if method not in aggregated:
            continue
        metrics = aggregated[method]
        row = f"{method:<18}"
        for k in k_vals:
            recall = metrics.get(f"recall@{k}", 0)
            pair_recall = metrics.get(f"pair_recall@{k}", 0)
            row += f" {recall:<10.3f} {pair_recall:<10.3f}"
        row += f" {metrics.get('mrr', 0):<8.3f}"
        print(row)

    print("-" * 80)

    # Best method analysis
    print("\nBest Methods by Pair Recall@5:")
    pair_recall_5 = [(m, aggregated[m].get("pair_recall@5", 0))
                     for m in aggregated.keys()]
    pair_recall_5.sort(key=lambda x: -x[1])
    for method, score in pair_recall_5:
        print(f"  {method}: {score:.3f}")


def save_results(
    results: List[EvalResult],
    aggregated: Dict,
    config: EvalConfig,
) -> str:
    """Save results to JSON file."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"eval_results_{timestamp}.json"

    output_data = {
        "config": asdict(config) if hasattr(config, '__dataclass_fields__') else vars(config),
        "aggregated": aggregated,
        "detailed_results": [asdict(r) for r in results],
        "timestamp": timestamp,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
    return str(output_file)


# ---------------------------------------------------------------------------
# Async versions of the evaluation functions
# ---------------------------------------------------------------------------


async def prepare_dataset_async(config: EvalConfig) -> tuple:
    """Async version of prepare_dataset.

    Downloads articles and generates questions concurrently.

    Returns
    -------
    tuple
        (articles, all_chunks, questions, doc_chunks_map)
    """
    print("=" * 60)
    print("PHASE 1: Dataset Preparation (Async)")
    print("=" * 60)

    # Download articles concurrently
    print("\n1.1 Downloading Wikipedia articles...")
    articles = await download_wikipedia_articles_async(
        titles=config.articles,
        min_word_count=config.min_word_count,
        cache_dir=config.cache_dir,
    )

    if not articles:
        raise RuntimeError("No articles downloaded!")

    print(f"  Downloaded {len(articles)} articles")

    # Save as text files (sync is fine for file I/O)
    docs_dir = Path(config.cache_dir) / "docs"
    save_articles_as_text(articles, str(docs_dir))

    # Parse and chunk (CPU-bound, keep sync)
    print("\n1.2 Parsing and chunking documents...")
    all_chunks = []
    doc_chunks_map = {}

    for article in articles:
        safe_title = article.title.replace(' ', '_')
        doc_path = docs_dir / f"{safe_title}.txt"

        if not doc_path.exists():
            print(f"  Warning: {doc_path} not found, skipping")
            continue

        nodes = parse_document(str(doc_path))
        chunks = chunk_nodes(nodes, config.max_tokens, config.overlap)

        for chunk in chunks:
            chunk["doc_id"] = article.title

        doc_chunks_map[article.title] = chunks
        all_chunks.extend(chunks)
        print(f"  {article.title}: {len(chunks)} chunks")

    print(f"  Total: {len(all_chunks)} chunks")

    # Find chunk pairs concurrently
    print("\n1.3 Finding chunk pairs...")
    all_pairs_dict = await find_all_bridge_chunk_pairs_async(
        doc_chunks_map,
        min_distance=config.min_chunk_distance,
        max_distance=config.max_chunk_distance,
        max_pairs=config.questions_per_article * 2,
    )

    # Flatten pairs
    all_pairs = []
    for doc_id, pairs in all_pairs_dict.items():
        all_pairs.extend(pairs)
        print(f"  {doc_id}: {len(pairs)} chunk pairs")

    # Generate questions concurrently
    print("\n1.4 Generating multi-hop questions...")
    questions = await generate_multihop_questions_async(
        chunk_pairs=all_pairs,
        model="gpt-5.2",
        validate=config.validate_questions,
        max_questions=config.questions_per_article * len(articles),
    )

    # Save questions
    questions_path = Path(config.cache_dir) / "questions.json"
    save_questions(questions, str(questions_path))

    print(f"\n  Generated {len(questions)} validated questions")

    return articles, all_chunks, questions, doc_chunks_map


async def build_retrievers_async(
    all_chunks: List[dict],
    config: EvalConfig,
) -> Dict[str, object]:
    """Async version of build_retrievers.

    Returns
    -------
    Dict[str, retriever]
        Dictionary mapping method names to retriever instances
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Building Retrievers (Async)")
    print("=" * 60)

    retrievers = {}

    # BM25 (no API calls needed, keep sync)
    print("\n2.1 Building BM25 retriever...")
    bm25 = BM25Retriever()
    bm25.index_chunks(all_chunks)
    retrievers["BM25"] = bm25
    print("  ✓ BM25 ready")

    # Naive (dense, no decay) - async embeddings
    print("\n2.2 Building Naive retriever (async)...")
    naive = NaiveRetriever(model=config.embedding_model)
    await naive.index_chunks_async(all_chunks)
    retrievers["Naive"] = naive
    print("  ✓ Naive ready")

    # Sentence Window - reuse embeddings from Naive
    print("\n2.3 Building Sentence Window retriever...")
    sentence_window = SentenceWindowRetriever(
        model=config.embedding_model,
        window_size=1,
    )
    sentence_window.chunks = all_chunks
    sentence_window.embeddings = naive.embeddings  # Reuse embeddings
    sentence_window.index = naive.index
    # Build doc_chunks
    sentence_window.doc_chunks = {}
    for chunk in all_chunks:
        doc_id = chunk.get("doc_id", "")
        if doc_id not in sentence_window.doc_chunks:
            sentence_window.doc_chunks[doc_id] = []
        sentence_window.doc_chunks[doc_id].append(chunk)
    for doc_id in sentence_window.doc_chunks:
        sentence_window.doc_chunks[doc_id].sort(key=lambda x: x.get("position", 0))
    retrievers["SentenceWindow"] = sentence_window
    print("  ✓ Sentence Window ready")

    # Hybrid - reuse both dense and sparse
    print("\n2.4 Building Hybrid retriever...")
    hybrid = HybridRetriever(model=config.embedding_model)
    hybrid.chunks = all_chunks
    hybrid.dense_retriever = naive
    hybrid.sparse_retriever = bm25
    retrievers["Hybrid"] = hybrid
    print("  ✓ Hybrid ready")

    # DecayRAG will use the main retrieve function
    retrievers["DecayRAG"] = None  # Placeholder, handled specially

    return retrievers


async def main_async(config: Optional[EvalConfig] = None) -> Dict:
    """Async version of main evaluation pipeline.

    Parameters
    ----------
    config : EvalConfig, optional
        Evaluation configuration

    Returns
    -------
    Dict
        Aggregated results
    """
    if config is None:
        config = EvalConfig()

    print("\n" + "=" * 60)
    print("DecayRAG Multi-Hop Retrieval Evaluation (Async)")
    print("=" * 60)
    print(f"Articles: {config.articles}")
    print(f"Questions per article: {config.questions_per_article}")
    print(f"Evaluation K values: {config.top_k_values}")

    # Phase 1: Prepare dataset (async)
    articles, all_chunks, questions, doc_chunks_map = await prepare_dataset_async(config)

    if not questions:
        print("ERROR: No questions generated!")
        return {}

    # Build DecayRAG index
    print("\n2.5 Building DecayRAG index (async)...")
    index_path = str(Path(config.cache_dir) / "decayrag_index.faiss")

    # Use async embedding
    embeds = await embed_chunks_async(all_chunks, config.embedding_model)
    upsert_embeddings(index_path, all_chunks, embeds)
    print("  ✓ DecayRAG index ready")

    # Phase 2: Build retrievers (async)
    retrievers = await build_retrievers_async(all_chunks, config)

    # Phase 3: Run evaluation (sync - FAISS search is CPU-bound)
    results = run_evaluation(questions, all_chunks, retrievers, config, index_path)

    # Phase 4: Aggregate and report (sync)
    aggregated = aggregate_results(results, config)
    print_results_table(aggregated, config)

    # Save results
    save_results(results, aggregated, config)

    return aggregated


def main(config: Optional[EvalConfig] = None) -> Dict:
    """Run the full evaluation pipeline.

    Parameters
    ----------
    config : EvalConfig, optional
        Evaluation configuration

    Returns
    -------
    Dict
        Aggregated results
    """
    if config is None:
        config = EvalConfig()

    print("\n" + "=" * 60)
    print("DecayRAG Multi-Hop Retrieval Evaluation")
    print("=" * 60)
    print(f"Articles: {config.articles}")
    print(f"Questions per article: {config.questions_per_article}")
    print(f"Evaluation K values: {config.top_k_values}")

    # Phase 1: Prepare dataset
    articles, all_chunks, questions, doc_chunks_map = prepare_dataset(config)

    if not questions:
        print("ERROR: No questions generated!")
        return {}

    # Build DecayRAG index for comparison
    print("\n2.5 Building DecayRAG index...")
    from decayrag import upsert_embeddings, embed_chunks

    index_path = str(Path(config.cache_dir) / "decayrag_index.faiss")

    # Check if we can reuse embeddings from naive retriever
    embeds = embed_chunks(all_chunks, config.embedding_model)
    upsert_embeddings(index_path, all_chunks, embeds)
    print("  ✓ DecayRAG index ready")

    # Phase 2: Build retrievers
    retrievers = build_retrievers(all_chunks, config)

    # Phase 3: Run evaluation
    results = run_evaluation(questions, all_chunks, retrievers, config, index_path)

    # Phase 4: Aggregate and report
    aggregated = aggregate_results(results, config)
    print_results_table(aggregated, config)

    # Save results
    save_results(results, aggregated, config)

    return aggregated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DecayRAG evaluation")
    parser.add_argument("--articles", nargs="+", default=None,
                       help="Wikipedia articles to use")
    parser.add_argument("--questions-per-article", type=int, default=3,
                       help="Questions per article")
    parser.add_argument("--no-validate", action="store_true",
                       help="Skip question validation")
    parser.add_argument("--async", dest="use_async", action="store_true",
                       help="Use async/concurrent execution (faster)")
    parser.add_argument("--sync", dest="use_async", action="store_false",
                       help="Use synchronous execution (default)")
    parser.set_defaults(use_async=True)  # Default to async for better performance
    args = parser.parse_args()

    config = EvalConfig(
        articles=args.articles,
        questions_per_article=args.questions_per_article,
        validate_questions=not args.no_validate,
    )

    if args.use_async:
        asyncio.run(main_async(config))
    else:
        main(config)
