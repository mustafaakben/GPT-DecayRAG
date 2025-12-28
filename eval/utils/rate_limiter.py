"""Rate limiting utilities for async operations."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional


@dataclass
class RateLimiterConfig:
    """Configuration for rate limiting.

    Attributes
    ----------
    openai_concurrency : int
        Max concurrent OpenAI API requests (default: 20)
    wikipedia_concurrency : int
        Max concurrent Wikipedia API requests (default: 10)
    embedding_concurrency : int
        Max concurrent embedding batch requests (default: 50)
    """

    openai_concurrency: int = 20
    wikipedia_concurrency: int = 10
    embedding_concurrency: int = 50


class RateLimiter:
    """Centralized rate limiter for different services.

    Uses asyncio.Semaphore to limit concurrent requests per service type.
    Semaphores are created lazily on first access.

    Examples
    --------
    >>> limiter = RateLimiter()
    >>> async with limiter.openai:
    ...     response = await client.chat.completions.create(...)
    """

    def __init__(self, config: Optional[RateLimiterConfig] = None):
        """Initialize rate limiter with optional config.

        Parameters
        ----------
        config : RateLimiterConfig, optional
            Configuration for concurrency limits
        """
        self.config = config or RateLimiterConfig()
        self._openai_semaphore: Optional[asyncio.Semaphore] = None
        self._wikipedia_semaphore: Optional[asyncio.Semaphore] = None
        self._embedding_semaphore: Optional[asyncio.Semaphore] = None

    @property
    def openai(self) -> asyncio.Semaphore:
        """Get semaphore for OpenAI API requests."""
        if self._openai_semaphore is None:
            self._openai_semaphore = asyncio.Semaphore(
                self.config.openai_concurrency
            )
        return self._openai_semaphore

    @property
    def wikipedia(self) -> asyncio.Semaphore:
        """Get semaphore for Wikipedia API requests."""
        if self._wikipedia_semaphore is None:
            self._wikipedia_semaphore = asyncio.Semaphore(
                self.config.wikipedia_concurrency
            )
        return self._wikipedia_semaphore

    @property
    def embedding(self) -> asyncio.Semaphore:
        """Get semaphore for embedding requests."""
        if self._embedding_semaphore is None:
            self._embedding_semaphore = asyncio.Semaphore(
                self.config.embedding_concurrency
            )
        return self._embedding_semaphore

    def reset(self) -> None:
        """Reset all semaphores (useful for testing)."""
        self._openai_semaphore = None
        self._wikipedia_semaphore = None
        self._embedding_semaphore = None


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter(config: Optional[RateLimiterConfig] = None) -> RateLimiter:
    """Get or create the global rate limiter.

    Parameters
    ----------
    config : RateLimiterConfig, optional
        Configuration to use if creating new limiter

    Returns
    -------
    RateLimiter
        The global rate limiter instance
    """
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(config)
    return _rate_limiter


def reset_rate_limiter() -> None:
    """Reset the global rate limiter (useful for testing)."""
    global _rate_limiter
    _rate_limiter = None
