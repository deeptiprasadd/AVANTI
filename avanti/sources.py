"""Find papers online and fetch their PDFs.

Three open sources, all free and keyless:

  arXiv       preprints in physics, maths, CS, statistics, biology, finance
  OpenAlex    a catalogue of works across every field; only results with an
              open-access PDF are kept, since a paper has to be read to be
              analyzed
  Europe PMC  biomedical and life-science literature, open-access subset

`search(query, source="all")` queries them together and merges the results.
"""

from __future__ import annotations

import html
import ipaddress
import re
import socket
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import feedparser
import requests

from . import __version__

ARXIV_API = "https://export.arxiv.org/api/query"
OPENALEX_API = "https://api.openalex.org/works"
EUROPEPMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

USER_AGENT = (
    f"AVANTI/{__version__} (research paper analyzer; "
    "https://github.com/deeptiprasadd/AVANTI)"
)
TIMEOUT = 25

# Most results one search may ask for. Each result is a card on the page,
# and the slowest source decides how long a search takes.
MAX_RESULTS = 50

SOURCES = {"arxiv": "arXiv", "openalex": "OpenAlex", "europepmc": "Europe PMC"}


@dataclass
class SearchResult:
    """One paper found by a search."""

    title: str
    authors: List[str]
    published: str
    summary: str
    abstract_url: str
    pdf_url: str
    source: str = "arXiv"

    @property
    def author_line(self) -> str:
        if not self.authors:
            return "Unknown"
        if len(self.authors) <= 3:
            return ", ".join(self.authors)
        return f"{', '.join(self.authors[:3])} +{len(self.authors) - 3} more"


@dataclass
class SearchResponse:
    """Merged results, plus the sources that could not be reached."""

    results: List[SearchResult] = field(default_factory=list)
    failed: Dict[str, str] = field(default_factory=dict)  # source -> error


def _clean(text: str) -> str:
    """Strip markup and collapse whitespace (Europe PMC titles carry <i>)."""
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", text or "")).split())


def _get(url: str, params: dict) -> requests.Response:
    response = requests.get(
        url, params=params, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT
    )
    response.raise_for_status()
    return response


def _limit(limit: int) -> int:
    return max(1, min(int(limit), MAX_RESULTS))


# ------------------------------------------------------------------ arXiv


def _pdf_url(entry) -> str:
    """Build a clean PDF URL from an arXiv feed entry."""
    for link in getattr(entry, "links", []):
        if getattr(link, "type", "") == "application/pdf":
            return link.href
    entry_id = getattr(entry, "id", "")
    if "/abs/" in entry_id:
        return f"https://arxiv.org/pdf/{entry_id.split('/abs/')[-1]}"
    return entry_id


_ARXIV_STOP = {
    "a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "with",
    "is", "are", "by", "from", "at", "as", "via", "using", "how", "what",
}


def _arxiv_query(query: str) -> str:
    """Require every meaningful word.

    arXiv treats "all:graph neural networks drug discovery" as one required
    word and four optional ones, so its top hit for that query was a paper
    on polynomial chaos. ANDing the words returns GNN drug-discovery papers.
    """
    words = [w for w in re.findall(r"[\w\-]+", query)
             if w.lower() not in _ARXIV_STOP]
    return " AND ".join(f"all:{w}" for w in words) or f"all:{query}"


def search_arxiv(query: str, limit: int = 10) -> List[SearchResult]:
    """Search arXiv. Raises requests.RequestException on network failure."""
    query = (query or "").strip()
    if not query:
        return []

    def fetch(search_query: str):
        # Params go through `params` so that a query containing &, +, : or
        # spaces is encoded rather than corrupting the URL.
        response = _get(
            ARXIV_API,
            {
                "search_query": search_query,
                "start": 0,
                "max_results": _limit(limit),
                "sortBy": "relevance",
            },
        )
        return feedparser.parse(response.content)

    feed = fetch(_arxiv_query(query))
    if not feed.entries:
        # Every word together matched nothing; settle for any of them.
        feed = fetch(f"all:{query}")

    return [
        SearchResult(
            title=" ".join(getattr(entry, "title", "").split()),
            authors=[a.name for a in getattr(entry, "authors", [])],
            published=getattr(entry, "published", "")[:10],
            summary=" ".join(getattr(entry, "summary", "").split()),
            abstract_url=getattr(entry, "id", ""),
            pdf_url=_pdf_url(entry),
            source=SOURCES["arxiv"],
        )
        for entry in feed.entries
    ]


# --------------------------------------------------------------- OpenAlex


def _openalex_abstract(inverted: Optional[dict]) -> str:
    """OpenAlex ships abstracts as {word: [positions]}; put them back in order."""
    if not inverted:
        return ""
    slots = {}
    for word, positions in inverted.items():
        for p in positions:
            slots[p] = word
    return " ".join(slots[i] for i in sorted(slots))


def _openalex_pdf(work: dict) -> str:
    """The open PDF most likely to actually download.

    An arXiv copy wins when there is one. OpenAlex's own "best" location is
    not always reachable: for "Attention Is All You Need" it pointed at a
    mirror that answered 404, while an arXiv copy sat further down the list.
    """
    best = work.get("best_oa_location") or {}
    candidates = [best.get("pdf_url")] + [
        loc.get("pdf_url") for loc in work.get("locations") or [] if loc
    ]
    candidates = [url for url in candidates if url]
    for url in candidates:
        if urlparse(url).hostname in ("arxiv.org", "export.arxiv.org"):
            return url
    return candidates[0] if candidates else ""


def search_openalex(query: str, limit: int = 10) -> List[SearchResult]:
    """Search OpenAlex for open-access works that have a PDF."""
    query = (query or "").strip()
    if not query:
        return []

    want = _limit(limit)
    response = _get(
        OPENALEX_API,
        {
            "search": query,
            "filter": "open_access.is_oa:true",
            # Many open works only have a landing page, not a PDF; ask for
            # extra so enough survive the filter below.
            "per-page": min(200, want * 3),
            "select": "id,doi,title,publication_date,authorships,"
                      "best_oa_location,locations,abstract_inverted_index",
        },
    )

    results: List[SearchResult] = []
    for work in response.json().get("results", []):
        pdf = _openalex_pdf(work)
        if not pdf or not work.get("title"):
            continue
        results.append(
            SearchResult(
                title=_clean(work["title"]),
                authors=[
                    a["author"]["display_name"]
                    for a in work.get("authorships") or []
                    if a.get("author", {}).get("display_name")
                ],
                published=work.get("publication_date") or "",
                summary=_openalex_abstract(work.get("abstract_inverted_index")),
                abstract_url=work.get("doi") or work.get("id") or pdf,
                pdf_url=pdf,
                source=SOURCES["openalex"],
            )
        )
        if len(results) >= want:
            break
    return results


# ------------------------------------------------------------- Europe PMC


def _europepmc_pdf(record: dict) -> str:
    urls = (record.get("fullTextUrlList") or {}).get("fullTextUrl") or []
    pdfs = [u.get("url", "") for u in urls if u.get("documentStyle") == "pdf"]
    # Europe PMC's own render link serves the PDF directly; NCBI's answers a
    # script with a bot check.
    for url in pdfs:
        if "europepmc.org" in url:
            return url
    return pdfs[0] if pdfs else ""


def search_europepmc(query: str, limit: int = 10) -> List[SearchResult]:
    """Search Europe PMC's open-access, PDF-bearing subset."""
    query = (query or "").strip()
    if not query:
        return []

    want = _limit(limit)
    response = _get(
        EUROPEPMC_API,
        {
            "query": f"({query}) AND OPEN_ACCESS:y AND HAS_PDF:y",
            "format": "json",
            "resultType": "core",
            "pageSize": min(100, want * 2),
        },
    )

    results: List[SearchResult] = []
    for record in response.json().get("resultList", {}).get("result", []):
        pdf = _europepmc_pdf(record)
        if not pdf or not record.get("title"):
            continue
        pmcid = record.get("pmcid", "")
        authors = [a.strip() for a in record.get("authorString", "").split(",")]
        results.append(
            SearchResult(
                title=_clean(record["title"]).rstrip("."),
                authors=[a.rstrip(".") for a in authors if a],
                published=record.get("firstPublicationDate")
                or str(record.get("pubYear") or ""),
                summary=_clean(record.get("abstractText", "")),
                abstract_url=f"https://europepmc.org/articles/{pmcid}"
                if pmcid else pdf,
                pdf_url=pdf,
                source=SOURCES["europepmc"],
            )
        )
        if len(results) >= want:
            break
    return results


# ------------------------------------------------------------------ merged

_SEARCHERS: Dict[str, Callable[[str, int], List[SearchResult]]] = {
    "arxiv": search_arxiv,
    "openalex": search_openalex,
    "europepmc": search_europepmc,
}


def _title_key(title: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", title.lower()))


def search(query: str, limit: int = 10, source: str = "all") -> SearchResponse:
    """Search one source, or all of them at once.

    With "all", each source is asked for `limit` results in parallel and the
    lists are interleaved -- first from each, then second from each -- so no
    one source crowds out the others, and a paper listed by two sources
    appears once. A source that fails is reported in `failed` rather than
    sinking the whole search; only if every source fails is that an error.
    """
    if source != "all" and source not in _SEARCHERS:
        raise ValueError(f"Unknown source {source!r}.")
    want = _limit(limit)
    names = list(_SEARCHERS) if source == "all" else [source]

    found: Dict[str, List[SearchResult]] = {}
    out = SearchResponse()
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        futures = {n: pool.submit(_SEARCHERS[n], query, want) for n in names}
        for name, future in futures.items():
            try:
                found[name] = future.result()
            except (requests.RequestException, ValueError) as exc:
                out.failed[SOURCES[name]] = str(exc)

    if not found:
        raise requests.RequestException(
            "; ".join(f"{k}: {v}" for k, v in out.failed.items())
        )

    seen = set()
    for rank in range(want):
        for name in names:
            batch = found.get(name, [])
            if rank < len(batch):
                key = _title_key(batch[rank].title)
                if key and key not in seen:
                    seen.add(key)
                    out.results.append(batch[rank])
    out.results = out.results[:want]
    return out


# --------------------------------------------------------------- download


class UnsafeURL(ValueError):
    """A URL that points somewhere a server must not fetch from."""


def check_public_url(url: str) -> None:
    """Refuse URLs that resolve to private, loopback or link-local addresses.

    The server downloads PDFs from URLs that search results -- and callers --
    supply. Without this, a URL like http://169.254.169.254/ would have the
    server fetch its own cloud metadata, or probe services on its private
    network, on someone else's behalf.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise UnsafeURL("URL must be http or https.")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or 443)
    except socket.gaierror as exc:
        raise UnsafeURL(f"Could not resolve {parsed.hostname}.") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0].split("%")[0])
        if not address.is_global:
            raise UnsafeURL("That address is not on the public internet.")


def fetch_pdf(url: str, max_bytes: int = 40 * 1024 * 1024) -> Optional[bytes]:
    """Download a PDF. Returns None if it is not a PDF or is too large.

    Raises UnsafeURL for a non-public address, and requests.RequestException
    on network failure. Redirects are followed by hand so that every hop is
    checked, not just the first.
    """
    for _ in range(6):
        check_public_url(url)
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=60,
            stream=True,
            allow_redirects=False,
        )
        if response.is_redirect:
            url = urljoin(url, response.headers.get("Location", ""))
            response.close()
            continue
        break
    else:
        raise requests.TooManyRedirects("Too many redirects.")

    response.raise_for_status()

    chunks, total = [], 0
    for chunk in response.iter_content(chunk_size=65536):
        total += len(chunk)
        if total > max_bytes:
            response.close()
            return None
        chunks.append(chunk)

    data = b"".join(chunks)
    # A PDF always starts with %PDF; anything else is an error page.
    return data if data[:4] == b"%PDF" else None
