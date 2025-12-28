"""Test individual components of the evaluation framework."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()


def test_wikipedia_loader():
    """Test Wikipedia article download."""
    print("\n" + "=" * 50)
    print("Testing Wikipedia Loader")
    print("=" * 50)

    from eval.dataset.wikipedia_loader import download_wikipedia_article

    article = download_wikipedia_article("Albert_Einstein")
    if article:
        print(f"  ✓ Downloaded: {article.title}")
        print(f"    Words: {article.word_count}")
        print(f"    Preview: {article.content[:100]}...")
        return True
    else:
        print("  ✗ Failed to download article")
        return False


def test_chunk_pair_finder():
    """Test chunk pair identification."""
    print("\n" + "=" * 50)
    print("Testing Chunk Pair Finder")
    print("=" * 50)

    from eval.dataset.chunk_pair_finder import find_bridge_chunk_pairs_simple

    chunks = [
        {"position": 0, "text": "Albert Einstein was born in Ulm, Germany in 1879."},
        {"position": 1, "text": "He showed early aptitude for mathematics."},
        {"position": 2, "text": "Einstein moved to Switzerland in 1895."},
        {"position": 3, "text": "He worked at the Swiss Patent Office."},
        {"position": 4, "text": "In 1905, Einstein published groundbreaking papers."},
        {"position": 5, "text": "The theory of relativity changed physics."},
    ]

    pairs = find_bridge_chunk_pairs_simple(chunks, "einstein", min_distance=2, max_distance=5)

    if pairs:
        print(f"  ✓ Found {len(pairs)} chunk pairs")
        for p in pairs[:2]:
            print(f"    [{p.chunk_a_pos}] <-> [{p.chunk_b_pos}]: {p.connecting_entity}")
        return True
    else:
        print("  ✗ No pairs found")
        return False


def test_metrics():
    """Test evaluation metrics."""
    print("\n" + "=" * 50)
    print("Testing Evaluation Metrics")
    print("=" * 50)

    from eval.metrics import recall_at_k, pair_recall_at_k, mrr, ndcg_at_k

    retrieved = [
        {"position": 5, "score": 0.9},
        {"position": 2, "score": 0.8},
        {"position": 7, "score": 0.7},
        {"position": 1, "score": 0.6},
    ]
    targets = [2, 7]

    recall = recall_at_k(retrieved, targets, k=5)
    pair_recall = pair_recall_at_k(retrieved, targets[0], targets[1], k=5)
    mrr_score = mrr(retrieved, targets)

    print(f"  Recall@5: {recall:.3f}")
    print(f"  Pair Recall@5: {pair_recall:.3f}")
    print(f"  MRR: {mrr_score:.3f}")

    if recall == 1.0 and pair_recall == 1.0:
        print("  ✓ Metrics working correctly")
        return True
    else:
        print("  ✗ Unexpected metric values")
        return False


def test_bm25_retriever():
    """Test BM25 retriever."""
    print("\n" + "=" * 50)
    print("Testing BM25 Retriever")
    print("=" * 50)

    from eval.baselines import BM25Retriever

    chunks = [
        {"doc_id": "test", "position": 0, "text": "Python is a programming language."},
        {"doc_id": "test", "position": 1, "text": "Machine learning uses algorithms."},
        {"doc_id": "test", "position": 2, "text": "Python is great for data science."},
    ]

    retriever = BM25Retriever()
    retriever.index_chunks(chunks)
    results = retriever.retrieve("Python programming", top_k=2)

    if results and results[0]["position"] in [0, 2]:
        print(f"  ✓ BM25 working, retrieved {len(results)} chunks")
        for r in results:
            print(f"    [{r['position']}] {r['score']:.3f}: {r['text'][:40]}")
        return True
    else:
        print("  ✗ BM25 not working correctly")
        return False


def test_naive_retriever():
    """Test Naive (dense) retriever."""
    print("\n" + "=" * 50)
    print("Testing Naive Retriever")
    print("=" * 50)

    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("  ⚠ Skipping (no API key)")
        return True

    from eval.baselines import NaiveRetriever

    chunks = [
        {"doc_id": "test", "position": 0, "text": "Python is a programming language."},
        {"doc_id": "test", "position": 1, "text": "Cooking recipes for dinner."},
        {"doc_id": "test", "position": 2, "text": "Software development with Python."},
    ]

    retriever = NaiveRetriever()
    retriever.index_chunks(chunks)
    results = retriever.retrieve("Python coding", top_k=2)

    if results:
        print(f"  ✓ Naive retriever working, retrieved {len(results)} chunks")
        for r in results:
            print(f"    [{r['position']}] {r['score']:.3f}: {r['text'][:40]}")
        return True
    else:
        print("  ✗ Naive retriever not working")
        return False


def test_question_generator():
    """Test question generation with GPT-5.2."""
    print("\n" + "=" * 50)
    print("Testing Question Generator (GPT-5.2)")
    print("=" * 50)

    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("  ⚠ Skipping (no API key)")
        return True

    from eval.dataset.chunk_pair_finder import ChunkPair
    from eval.dataset.question_generator import generate_question_for_pair

    pair = ChunkPair(
        doc_id="Einstein",
        chunk_a_pos=0,
        chunk_b_pos=4,
        chunk_a_text="Albert Einstein was born in Ulm, Germany in 1879. He showed early aptitude for mathematics and physics.",
        chunk_b_text="In 1905, Einstein published four groundbreaking papers while working at the Swiss Patent Office. This year became known as his 'miracle year'.",
        connecting_entity="Einstein",
        relationship="early_life_to_achievement",
    )

    question = generate_question_for_pair(pair, model="gpt-5.2", validate=False)

    if question:
        print(f"  ✓ Question generated:")
        print(f"    Q: {question.question}")
        print(f"    A: {question.answer}")
        return True
    else:
        print("  ✗ Question generation failed")
        return False


def run_all_tests():
    """Run all component tests."""
    print("\n" + "=" * 60)
    print("DecayRAG Evaluation Framework - Component Tests")
    print("=" * 60)

    tests = [
        ("Wikipedia Loader", test_wikipedia_loader),
        ("Chunk Pair Finder", test_chunk_pair_finder),
        ("Evaluation Metrics", test_metrics),
        ("BM25 Retriever", test_bm25_retriever),
        ("Naive Retriever", test_naive_retriever),
        ("Question Generator", test_question_generator),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for _, p in results if p)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    for name, p in results:
        status = "✓" if p else "✗"
        print(f"  {status} {name}")

    return all(p for _, p in results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
