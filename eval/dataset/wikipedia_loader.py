"""Wikipedia article downloader and processor."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote

import aiohttp
import wikipedia

# Default concurrency limit for Wikipedia API calls
WIKIPEDIA_CONCURRENCY_LIMIT = 10


@dataclass
class WikipediaArticle:
    """Represents a downloaded Wikipedia article."""

    title: str
    content: str
    url: str
    categories: List[str] = field(default_factory=list)
    word_count: int = 0

    def __post_init__(self):
        if self.word_count == 0:
            self.word_count = len(self.content.split())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "WikipediaArticle":
        return cls(**data)


# Default articles for evaluation (long, information-rich)
DEFAULT_ARTICLES = [
    "Artificial_intelligence",
    "World_War_II",
    "Amazon_rainforest",
    "Albert_Einstein",
    "Climate_change",
]


def download_wikipedia_article(title: str) -> Optional[WikipediaArticle]:
    """Download a single Wikipedia article.

    Parameters
    ----------
    title : str
        Wikipedia article title (use underscores for spaces)

    Returns
    -------
    WikipediaArticle or None
        The downloaded article, or None if download failed
    """
    try:
        # Handle underscore-formatted titles
        search_title = title.replace("_", " ")

        # Get the page
        page = wikipedia.page(search_title, auto_suggest=False)

        # Clean content - remove reference markers like [1], [2], etc.
        content = re.sub(r'\[\d+\]', '', page.content)

        # Remove excessive whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)

        return WikipediaArticle(
            title=page.title,
            content=content,
            url=page.url,
            categories=page.categories[:10],  # Limit categories
        )
    except wikipedia.exceptions.DisambiguationError as e:
        # If disambiguation, try the first option
        if e.options:
            try:
                page = wikipedia.page(e.options[0], auto_suggest=False)
                content = re.sub(r'\[\d+\]', '', page.content)
                content = re.sub(r'\n{3,}', '\n\n', content)
                return WikipediaArticle(
                    title=page.title,
                    content=content,
                    url=page.url,
                    categories=page.categories[:10],
                )
            except Exception:
                pass
        print(f"Warning: Disambiguation error for '{title}': {e.options[:3]}")
        return None
    except wikipedia.exceptions.PageError:
        print(f"Warning: Page not found: '{title}'")
        return None
    except Exception as e:
        print(f"Warning: Error downloading '{title}': {e}")
        return None


def download_wikipedia_articles(
    titles: Optional[List[str]] = None,
    min_word_count: int = 3000,
    cache_dir: Optional[str] = None,
) -> List[WikipediaArticle]:
    """Download multiple Wikipedia articles.

    Parameters
    ----------
    titles : List[str], optional
        List of article titles. Uses DEFAULT_ARTICLES if None.
    min_word_count : int
        Minimum word count to include article (default 3000)
    cache_dir : str, optional
        Directory to cache downloaded articles

    Returns
    -------
    List[WikipediaArticle]
        Successfully downloaded articles meeting criteria
    """
    titles = titles or DEFAULT_ARTICLES
    articles = []

    # Check cache first
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_file = cache_path / "wikipedia_articles.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                cached_articles = [WikipediaArticle.from_dict(a) for a in cached]
                cached_titles = {a.title for a in cached_articles}

                # Return cached if all titles present
                if all(t.replace("_", " ") in cached_titles or t in cached_titles
                       for t in titles):
                    print(f"Loaded {len(cached_articles)} articles from cache")
                    return [a for a in cached_articles if a.word_count >= min_word_count]
            except Exception as e:
                print(f"Cache read error: {e}")

    # Download articles
    for title in titles:
        print(f"Downloading: {title}...")
        article = download_wikipedia_article(title)

        if article:
            if article.word_count >= min_word_count:
                articles.append(article)
                print(f"  ✓ {article.title}: {article.word_count} words")
            else:
                print(f"  ✗ {article.title}: only {article.word_count} words (min: {min_word_count})")
        else:
            print(f"  ✗ Failed to download: {title}")

    # Cache results
    if cache_dir and articles:
        cache_file = Path(cache_dir) / "wikipedia_articles.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump([a.to_dict() for a in articles], f, indent=2)
        print(f"Cached {len(articles)} articles to {cache_file}")

    return articles


async def download_wikipedia_article_async(
    title: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
) -> Optional[WikipediaArticle]:
    """Async version of download_wikipedia_article using Wikipedia API.

    Parameters
    ----------
    title : str
        Wikipedia article title (use underscores for spaces)
    session : aiohttp.ClientSession
        Aiohttp session for making requests
    semaphore : asyncio.Semaphore
        Semaphore for rate limiting

    Returns
    -------
    WikipediaArticle or None
        The downloaded article, or None if download failed
    """
    search_title = title.replace("_", " ")

    # Wikipedia API endpoint for getting article content
    api_url = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=query"
        f"&titles={quote(search_title)}"
        f"&prop=extracts|categories|info"
        f"&explaintext=1"
        f"&format=json"
        f"&inprop=url"
        f"&cllimit=10"
    )

    try:
        async with semaphore:
            async with session.get(api_url) as response:
                if response.status != 200:
                    print(f"Warning: HTTP {response.status} for '{title}'")
                    return None

                data = await response.json()

        pages = data.get("query", {}).get("pages", {})
        if not pages:
            print(f"Warning: No pages found for '{title}'")
            return None

        # Get the first (and should be only) page
        page_id = next(iter(pages.keys()))
        page = pages[page_id]

        # Check for missing page
        if page_id == "-1" or "missing" in page:
            print(f"Warning: Page not found: '{title}'")
            return None

        content = page.get("extract", "")
        if not content:
            print(f"Warning: No content for '{title}'")
            return None

        # Clean content - remove excessive whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)

        # Extract categories
        categories = [
            cat.get("title", "").replace("Category:", "")
            for cat in page.get("categories", [])
        ]

        return WikipediaArticle(
            title=page.get("title", search_title),
            content=content,
            url=page.get("fullurl", f"https://en.wikipedia.org/wiki/{quote(search_title)}"),
            categories=categories[:10],
        )

    except asyncio.TimeoutError:
        print(f"Warning: Timeout downloading '{title}'")
        return None
    except Exception as e:
        print(f"Warning: Error downloading '{title}': {e}")
        return None


async def download_wikipedia_articles_async(
    titles: Optional[List[str]] = None,
    min_word_count: int = 3000,
    cache_dir: Optional[str] = None,
    concurrency_limit: int = WIKIPEDIA_CONCURRENCY_LIMIT,
) -> List[WikipediaArticle]:
    """Async version of download_wikipedia_articles.

    Downloads multiple Wikipedia articles concurrently.

    Parameters
    ----------
    titles : List[str], optional
        List of article titles. Uses DEFAULT_ARTICLES if None.
    min_word_count : int
        Minimum word count to include article (default 3000)
    cache_dir : str, optional
        Directory to cache downloaded articles
    concurrency_limit : int
        Maximum concurrent downloads

    Returns
    -------
    List[WikipediaArticle]
        Successfully downloaded articles meeting criteria
    """
    titles = titles or DEFAULT_ARTICLES

    # Check cache first (sync file I/O is fine for small cache check)
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_file = cache_path / "wikipedia_articles.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                cached_articles = [WikipediaArticle.from_dict(a) for a in cached]
                cached_titles = {a.title for a in cached_articles}

                # Return cached if all titles present
                if all(t.replace("_", " ") in cached_titles or t in cached_titles
                       for t in titles):
                    print(f"Loaded {len(cached_articles)} articles from cache")
                    return [a for a in cached_articles if a.word_count >= min_word_count]
            except Exception as e:
                print(f"Cache read error: {e}")

    # Download articles concurrently
    print(f"Downloading {len(titles)} articles concurrently...")
    semaphore = asyncio.Semaphore(concurrency_limit)

    timeout = aiohttp.ClientTimeout(total=30)
    connector = aiohttp.TCPConnector(limit=concurrency_limit)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [
            download_wikipedia_article_async(title, session, semaphore)
            for title in titles
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter successful results
    articles = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"  Failed: {titles[i]} - {result}")
        elif result is not None:
            if result.word_count >= min_word_count:
                articles.append(result)
                print(f"  {result.title}: {result.word_count} words")
            else:
                print(f"  {result.title}: only {result.word_count} words (min: {min_word_count})")

    # Cache results
    if cache_dir and articles:
        cache_file = Path(cache_dir) / "wikipedia_articles.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump([a.to_dict() for a in articles], f, indent=2)
        print(f"Cached {len(articles)} articles to {cache_file}")

    return articles


def save_articles_as_text(articles: List[WikipediaArticle], output_dir: str) -> List[Path]:
    """Save articles as text files for ingestion.

    Parameters
    ----------
    articles : List[WikipediaArticle]
        Articles to save
    output_dir : str
        Directory to save text files

    Returns
    -------
    List[Path]
        Paths to saved files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for article in articles:
        # Create safe filename
        safe_title = re.sub(r'[^\w\s-]', '', article.title).replace(' ', '_')
        file_path = output_path / f"{safe_title}.txt"

        # Write with title header
        content = f"# {article.title}\n\n{article.content}"
        file_path.write_text(content, encoding="utf-8")
        saved_files.append(file_path)
        print(f"Saved: {file_path}")

    return saved_files


if __name__ == "__main__":
    # Test the loader
    articles = download_wikipedia_articles(
        titles=["Artificial_intelligence", "Albert_Einstein"],
        min_word_count=1000,
        cache_dir="data/eval_cache",
    )

    for article in articles:
        print(f"\n{article.title}:")
        print(f"  Words: {article.word_count}")
        print(f"  URL: {article.url}")
        print(f"  Preview: {article.content[:200]}...")
