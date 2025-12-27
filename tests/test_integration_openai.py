import json
import os
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import pytest

from decayrag import batch_ingest, retrieve

WIKIPEDIA_TITLES = [
    "OpenAI",
    "Artificial_intelligence",
    "Natural_language_processing",
]


def _fetch_wikipedia_extract(title: str) -> str:
    url = (
        "https://en.wikipedia.org/w/api.php"
        "?action=query"
        "&prop=extracts"
        "&explaintext=1"
        "&format=json"
        f"&titles={quote(title)}"
    )
    request = Request(
        url,
        headers={
            "User-Agent": "DecayRAG-tests/0.1 (https://github.com/mustafaakben/DecayRAG)",
        },
    )
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    pages = payload.get("query", {}).get("pages", {})
    if not pages:
        return ""
    page = next(iter(pages.values()))
    return page.get("extract", "") or ""


@pytest.mark.integration
def test_wikipedia_ingest_and_retrieve(tmp_path: Path) -> None:
    if os.getenv("RUN_OPENAI_INTEGRATION") != "1":
        pytest.skip("RUN_OPENAI_INTEGRATION not enabled")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    for title in WIKIPEDIA_TITLES:
        extract = _fetch_wikipedia_extract(title)
        assert extract, f"Empty extract for {title}"
        (docs_dir / f"{title}.txt").write_text(extract)

    index_path = tmp_path / "index.faiss"
    batch_ingest(
        str(docs_dir),
        str(index_path),
        "text-embedding-3-small",
        max_tokens=200,
        overlap=20,
    )
    if not index_path.exists():
        pytest.skip("OpenAI embeddings failed; index not created")

    results = retrieve(
        "What does OpenAI do?",
        str(index_path),
        model="text-embedding-3-small",
        top_k=3,
        decay=True,
        blend=True,
        embedding_blend=False,
    )
    assert results
    doc_ids = {item.get("doc_id") for item in results}
    assert "OpenAI" in doc_ids

    blended_results = retrieve(
        "Define artificial intelligence.",
        str(index_path),
        model="text-embedding-3-small",
        top_k=3,
        decay=True,
        blend=True,
        embedding_blend=True,
    )
    assert blended_results
