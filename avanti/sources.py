"""Find papers online and fetch their PDFs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import feedparser
import requests

ARXIV_API = "https://export.arxiv.org/api/query"
USER_AGENT = "AVANTI/1.0 (research paper analyzer)"
TIMEOUT = 25


@dataclass
class SearchResult:
    """One paper found by a search."""

    title: str
    authors: List[str]
    published: str
    summary: str
    abstract_url: str
    pdf_url: str

    @property
    def author_line(self) -> str:
        if not self.authors:
            return "Unknown"
        if len(self.authors) <= 3:
            return ", ".join(self.authors)
        return f"{', '.join(self.authors[:3])} +{len(self.authors) - 3} more"


def _pdf_url(entry) -> str:
    """Build a clean PDF URL from an arXiv feed entry."""
    for link in getattr(entry, "links", []):
        if getattr(link, "type", "") == "application/pdf":
            return link.href
    entry_id = getattr(entry, "id", "")
    if "/abs/" in entry_id:
        return f"https://arxiv.org/pdf/{entry_id.split('/abs/')[-1]}"
    return entry_id


def search_arxiv(query: str, limit: int = 8) -> List[SearchResult]:
    """Search arXiv. Raises requests.RequestException on network failure."""
    query = (query or "").strip()
    if not query:
        return []

    # Params go through `params` so that a query containing &, +, : or
    # spaces is encoded rather than corrupting the URL.
    response = requests.get(
        ARXIV_API,
        params={
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": int(limit),
            "sortBy": "relevance",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    feed = feedparser.parse(response.content)

    results: List[SearchResult] = []
    for entry in feed.entries:
        results.append(
            SearchResult(
                title=" ".join(getattr(entry, "title", "").split()),
                authors=[a.name for a in getattr(entry, "authors", [])],
                published=getattr(entry, "published", "")[:10],
                summary=" ".join(getattr(entry, "summary", "").split()),
                abstract_url=getattr(entry, "id", ""),
                pdf_url=_pdf_url(entry),
            )
        )
    return results


def fetch_pdf(url: str, max_bytes: int = 40 * 1024 * 1024) -> Optional[bytes]:
    """Download a PDF. Returns None if it is not a PDF or is too large."""
    response = requests.get(
        url, headers={"User-Agent": USER_AGENT}, timeout=60, stream=True
    )
    response.raise_for_status()

    chunks, total = [], 0
    for chunk in response.iter_content(chunk_size=65536):
        total += len(chunk)
        if total > max_bytes:
            return None
        chunks.append(chunk)

    data = b"".join(chunks)
    # A PDF always starts with %PDF; anything else is an error page.
    return data if data[:4] == b"%PDF" else None
