"""Tests for async evaluation framework."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

import numpy as np
import pytest

from eval.utils.rate_limiter import RateLimiter, RateLimiterConfig, reset_rate_limiter
from eval.dataset.chunk_pair_finder import ChunkPair
from eval.dataset.question_generator import MultiHopQuestion


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_chunk_pair():
    """Create a sample ChunkPair for testing."""
    return ChunkPair(
        doc_id="test_doc",
        chunk_a_pos=0,
        chunk_b_pos=3,
        chunk_a_text="Albert Einstein was born in Germany in 1879.",
        chunk_b_text="He developed the theory of relativity in 1905.",
        connecting_entity="Einstein",
        relationship="biography",
    )


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        {"doc_id": "test", "position": 0, "text": "Python is a programming language."},
        {"doc_id": "test", "position": 1, "text": "It was created by Guido van Rossum."},
        {"doc_id": "test", "position": 2, "text": "Python is used in data science."},
        {"doc_id": "test", "position": 3, "text": "Machine learning is popular."},
    ]


@pytest.fixture(autouse=True)
def reset_global_rate_limiter():
    """Reset global rate limiter before each test."""
    reset_rate_limiter()
    yield
    reset_rate_limiter()


# ---------------------------------------------------------------------------
# Rate Limiter Tests
# ---------------------------------------------------------------------------


class TestRateLimiter:
    """Tests for rate limiting functionality."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RateLimiterConfig()
        assert config.openai_concurrency == 20
        assert config.wikipedia_concurrency == 10
        assert config.embedding_concurrency == 50

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RateLimiterConfig(
            openai_concurrency=5,
            wikipedia_concurrency=3,
            embedding_concurrency=10,
        )
        limiter = RateLimiter(config)
        assert limiter.config.openai_concurrency == 5
        assert limiter.config.wikipedia_concurrency == 3
        assert limiter.config.embedding_concurrency == 10

    def test_semaphore_creation(self):
        """Test that semaphores are created lazily."""
        limiter = RateLimiter()
        assert limiter._openai_semaphore is None

        # Access the property to create the semaphore
        sem = limiter.openai
        assert sem is not None
        assert isinstance(sem, asyncio.Semaphore)

        # Should return same instance
        assert limiter.openai is sem

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """Test that semaphore properly limits concurrent operations."""
        active_count = 0
        max_active = 0
        lock = asyncio.Lock()

        async def tracked_operation(semaphore):
            nonlocal active_count, max_active
            async with semaphore:
                async with lock:
                    active_count += 1
                    max_active = max(max_active, active_count)
                await asyncio.sleep(0.01)
                async with lock:
                    active_count -= 1

        semaphore = asyncio.Semaphore(3)
        tasks = [tracked_operation(semaphore) for _ in range(10)]
        await asyncio.gather(*tasks)

        assert max_active <= 3

    def test_reset(self):
        """Test reset clears semaphores."""
        limiter = RateLimiter()
        _ = limiter.openai  # Create semaphore
        assert limiter._openai_semaphore is not None

        limiter.reset()
        assert limiter._openai_semaphore is None


# ---------------------------------------------------------------------------
# Question Generator Async Tests
# ---------------------------------------------------------------------------


class TestQuestionGeneratorAsync:
    """Tests for async question generation."""

    @pytest.mark.asyncio
    async def test_generate_question_for_pair_async_mock(self, sample_chunk_pair):
        """Test async question generation with mocked OpenAI."""
        from eval.dataset.question_generator import generate_question_for_pair_async

        mock_response = MagicMock()
        mock_response.output_text = json.dumps({
            "question": "Test question?",
            "answer": "Test answer",
            "reasoning": "Test reasoning",
            "chunk_a_contribution": "Info A",
            "chunk_b_contribution": "Info B",
            "difficulty": "medium"
        })

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)

        semaphore = asyncio.Semaphore(10)

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            result = await generate_question_for_pair_async(
                sample_chunk_pair,
                mock_client,
                semaphore,
                model="gpt-4",
                validate=False,
            )

        assert result is not None
        assert result.question == "Test question?"
        assert result.answer == "Test answer"
        mock_client.responses.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generate_multihop_questions_async_handles_errors(self, sample_chunk_pair):
        """Test that async generation handles errors gracefully."""
        from eval.dataset.question_generator import generate_multihop_questions_async

        mock_response = MagicMock()
        mock_response.output_text = json.dumps({
            "question": "Test question?",
            "answer": "Test answer",
            "reasoning": "Test reasoning",
            "chunk_a_contribution": "Info A",
            "chunk_b_contribution": "Info B",
            "difficulty": "medium"
        })

        async def mock_create(*args, **kwargs):
            return mock_response

        with patch('openai.AsyncOpenAI') as MockClient:
            mock_instance = AsyncMock()
            mock_instance.responses.create = mock_create
            MockClient.return_value = mock_instance

            with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
                pairs = [sample_chunk_pair]
                results = await generate_multihop_questions_async(
                    pairs,
                    model="gpt-4",
                    validate=False,
                    concurrency_limit=2,
                )

        # Should return results even with some errors
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Wikipedia Loader Async Tests
# ---------------------------------------------------------------------------


class TestWikipediaLoaderAsync:
    """Tests for async Wikipedia loading."""

    @pytest.mark.asyncio
    async def test_download_wikipedia_article_async_success(self):
        """Test successful async article download."""
        from eval.dataset.wikipedia_loader import download_wikipedia_article_async
        import aiohttp

        mock_response_data = {
            "query": {
                "pages": {
                    "123": {
                        "title": "Test Article",
                        "extract": "This is test content. " * 100,
                        "fullurl": "https://en.wikipedia.org/wiki/Test",
                        "categories": [{"title": "Category:Test"}],
                    }
                }
            }
        }

        class MockResponse:
            status = 200
            async def json(self):
                return mock_response_data
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass

        class MockSession:
            def get(self, url):
                return MockResponse()

        semaphore = asyncio.Semaphore(10)
        result = await download_wikipedia_article_async(
            "Test_Article",
            MockSession(),
            semaphore,
        )

        assert result is not None
        assert result.title == "Test Article"

    @pytest.mark.asyncio
    async def test_download_wikipedia_article_async_not_found(self):
        """Test handling of missing article."""
        from eval.dataset.wikipedia_loader import download_wikipedia_article_async

        mock_response_data = {
            "query": {
                "pages": {
                    "-1": {
                        "missing": True,
                    }
                }
            }
        }

        class MockResponse:
            status = 200
            async def json(self):
                return mock_response_data
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass

        class MockSession:
            def get(self, url):
                return MockResponse()

        semaphore = asyncio.Semaphore(10)
        result = await download_wikipedia_article_async(
            "Nonexistent_Article",
            MockSession(),
            semaphore,
        )

        assert result is None


# ---------------------------------------------------------------------------
# Chunk Pair Finder Async Tests
# ---------------------------------------------------------------------------


class TestChunkPairFinderAsync:
    """Tests for async chunk pair finding."""

    @pytest.mark.asyncio
    async def test_find_all_bridge_chunk_pairs_async_fallback(self, sample_chunks):
        """Test that async finder falls back to simple method without API key."""
        from eval.dataset.chunk_pair_finder import find_all_bridge_chunk_pairs_async

        # Without API key, should use simple method
        with patch.dict('os.environ', {}, clear=True):
            doc_chunks_map = {"test_doc": sample_chunks}
            result = await find_all_bridge_chunk_pairs_async(
                doc_chunks_map,
                min_distance=1,
                max_distance=3,
            )

        assert isinstance(result, dict)
        assert "test_doc" in result
        # Simple method may or may not find pairs depending on shared entities


# ---------------------------------------------------------------------------
# Naive Retriever Async Tests
# ---------------------------------------------------------------------------


class TestNaiveRetrieverAsync:
    """Tests for async naive retriever."""

    @pytest.mark.asyncio
    async def test_index_chunks_async_mock(self, sample_chunks):
        """Test async indexing with mocked embeddings."""
        from eval.baselines import naive_retrieval

        fake_embeddings = np.random.randn(len(sample_chunks), 1536).astype(np.float32)

        async def fake_api_embed_async(texts, model, **kwargs):
            return fake_embeddings[:len(texts)]

        # Patch at the module where it's imported
        with patch.object(naive_retrieval, '_api_embed_async', fake_api_embed_async):
            retriever = naive_retrieval.NaiveRetriever()
            await retriever.index_chunks_async(sample_chunks)

        assert retriever.index is not None
        assert len(retriever.chunks) == len(sample_chunks)
        assert retriever.embeddings is not None


# ---------------------------------------------------------------------------
# Embedding Async Tests
# ---------------------------------------------------------------------------


class TestEmbeddingAsync:
    """Tests for async embedding functions."""

    @pytest.mark.asyncio
    async def test_api_embed_async_empty_input(self):
        """Test async embedding with empty input."""
        from decayrag.decayrag.ingest import _api_embed_async

        result = await _api_embed_async([], "text-embedding-3-small")
        assert result.shape == (0, 0)

    @pytest.mark.asyncio
    async def test_api_embed_async_mock(self):
        """Test async embedding with mocked client."""
        from decayrag.decayrag.ingest import _api_embed_async

        mock_embedding = [0.1] * 1536

        class MockResponse:
            data = [MagicMock(embedding=mock_embedding)]

        async def mock_create(*args, **kwargs):
            return MockResponse()

        with patch('openai.AsyncOpenAI') as MockClient:
            mock_instance = AsyncMock()
            mock_instance.embeddings.create = mock_create
            MockClient.return_value = mock_instance

            with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
                result = await _api_embed_async(
                    ["test text"],
                    "text-embedding-3-small",
                )

        assert result.shape == (1, 1536)


# ---------------------------------------------------------------------------
# Integration Tests (require API key)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAsyncIntegration:
    """Integration tests requiring API key."""

    @pytest.mark.asyncio
    async def test_full_async_pipeline(self, tmp_path):
        """Test full async pipeline with real API calls."""
        import os
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        if os.getenv("RUN_OPENAI_INTEGRATION") != "1":
            pytest.skip("RUN_OPENAI_INTEGRATION not enabled")

        from eval.run_evaluation import main_async, EvalConfig

        config = EvalConfig(
            articles=["Albert_Einstein"],
            questions_per_article=1,
            validate_questions=False,
            cache_dir=str(tmp_path / "cache"),
            output_dir=str(tmp_path / "results"),
        )

        result = await main_async(config)

        assert isinstance(result, dict)
        # Should have results for each method
        assert len(result) > 0
