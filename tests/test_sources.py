"""Tests for paper search and safe PDF download.

No network: each source's HTTP call is replaced with a canned response in
the shape that source really returns.
"""

import socket

import pytest
import requests

from avanti import sources

ARXIV_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <published>2017-06-12T17:57:34Z</published>
    <title>Attention Is All
      You Need</title>
    <summary>The dominant sequence transduction models...</summary>
    <author><name>Ashish Vaswani</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/1706.03762v7" rel="related" type="application/pdf"/>
  </entry>
</feed>"""

OPENALEX = {
    "results": [
        {   # no PDF anywhere: must be dropped
            "id": "https://openalex.org/W1", "title": "Landing page only",
            "best_oa_location": {"pdf_url": None}, "locations": [],
        },
        {
            "id": "https://openalex.org/W2", "doi": "https://doi.org/10.1/abc",
            "title": "Attention Is All You Need",
            "publication_date": "2017-06-12",
            "authorships": [{"author": {"display_name": "Ashish Vaswani"}}],
            "best_oa_location": None,
            "locations": [{"pdf_url": None}, {"pdf_url": "https://example.org/a.pdf"}],
            "abstract_inverted_index": {"Attention": [0], "works": [1], "well": [2]},
        },
        {
            "id": "https://openalex.org/W3", "title": "Graph <i>neural</i> networks",
            "best_oa_location": {"pdf_url": "https://example.org/g.pdf"},
        },
    ]
}

EUROPEPMC = {
    "resultList": {
        "result": [
            {
                "title": "Deep learning for <i>drug</i> discovery.",
                "authorString": "Smith J, Doe A.",
                "firstPublicationDate": "2024-02-01",
                "abstractText": "<h4>Background</h4>Drugs are hard.",
                "pmcid": "PMC123",
                "fullTextUrlList": {"fullTextUrl": [
                    {"documentStyle": "pdf", "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123/pdf/"},
                    {"documentStyle": "pdf", "url": "https://europepmc.org/articles/PMC123?pdf=render"},
                ]},
            },
            {"title": "No PDF here", "fullTextUrlList": {"fullTextUrl": []}},
        ]
    }
}


class FakeResponse:
    def __init__(self, payload=None, content=b"", status=200, headers=None):
        self._payload, self.content = payload, content
        self.status_code, self.headers = status, headers or {}

    @property
    def is_redirect(self):
        return self.status_code in (301, 302, 303, 307, 308)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def iter_content(self, chunk_size=1):
        yield self.content

    def close(self):
        pass


@pytest.fixture
def fake_http(monkeypatch):
    calls = []

    def get(url, params=None, **kwargs):
        calls.append((url, params))
        if url == sources.ARXIV_API:
            return FakeResponse(content=ARXIV_FEED)
        if url == sources.OPENALEX_API:
            return FakeResponse(OPENALEX)
        if url == sources.EUROPEPMC_API:
            return FakeResponse(EUROPEPMC)
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(sources.requests, "get", get)
    return calls


def test_arxiv_requires_every_meaningful_word(fake_http):
    results = sources.search_arxiv("graph neural networks for drug discovery")
    query = fake_http[0][1]["search_query"]
    assert query == ("all:graph AND all:neural AND all:networks AND all:drug "
                     "AND all:discovery")
    assert results[0].title == "Attention Is All You Need"
    assert results[0].pdf_url == "http://arxiv.org/pdf/1706.03762v7"
    assert results[0].source == "arXiv"


def test_openalex_keeps_only_works_with_a_pdf(fake_http):
    results = sources.search_openalex("attention", limit=5)
    assert [r.title for r in results] == ["Attention Is All You Need",
                                          "Graph neural networks"]
    first = results[0]
    assert first.pdf_url == "https://example.org/a.pdf"   # found in locations
    assert first.summary == "Attention works well"         # inverted index rebuilt
    assert first.abstract_url == "https://doi.org/10.1/abc"


def test_openalex_prefers_an_arxiv_copy():
    """OpenAlex's "best" copy of a paper can be a dead mirror."""
    work = {
        "best_oa_location": {"pdf_url": "https://mirror.example/download/10/108"},
        "locations": [{"pdf_url": "https://mirror.example/download/10/108"},
                      {"pdf_url": "https://arxiv.org/pdf/1706.03762"}],
    }
    assert sources._openalex_pdf(work) == "https://arxiv.org/pdf/1706.03762"


def test_europepmc_prefers_its_own_pdf_link_and_strips_markup(fake_http):
    results = sources.search_europepmc("drug discovery")
    assert len(results) == 1
    r = results[0]
    assert r.title == "Deep learning for drug discovery"
    assert r.pdf_url == "https://europepmc.org/articles/PMC123?pdf=render"
    assert r.summary == "Background Drugs are hard."
    assert r.authors == ["Smith J", "Doe A"]
    assert "OPEN_ACCESS:y" in fake_http[0][1]["query"]


def test_all_sources_interleave_and_drop_duplicates(fake_http):
    found = sources.search("attention", limit=10)
    titles = [(r.source, r.title) for r in found.results]
    # Round-robin by rank; OpenAlex's copy of the arXiv paper is dropped.
    assert titles == [
        ("arXiv", "Attention Is All You Need"),
        ("Europe PMC", "Deep learning for drug discovery"),
        ("OpenAlex", "Graph neural networks"),
    ]
    assert found.failed == {}


def test_limit_caps_merged_results(fake_http):
    assert len(sources.search("attention", limit=2).results) == 2


def test_one_failing_source_does_not_sink_the_search(monkeypatch):
    def get(url, params=None, **kwargs):
        if url == sources.OPENALEX_API:
            raise requests.ConnectionError("down")
        return FakeResponse(EUROPEPMC) if url == sources.EUROPEPMC_API \
            else FakeResponse(content=ARXIV_FEED)

    monkeypatch.setattr(sources.requests, "get", get)
    found = sources.search("attention")
    assert "OpenAlex" in found.failed
    assert {r.source for r in found.results} == {"arXiv", "Europe PMC"}


def test_every_source_failing_is_an_error(monkeypatch):
    def get(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(sources.requests, "get", get)
    with pytest.raises(requests.RequestException):
        sources.search("attention")


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError):
        sources.search("attention", source="google")


# --------------------------------------------------------------- download


def _resolve_to(monkeypatch, address):
    monkeypatch.setattr(
        socket, "getaddrinfo",
        lambda host, port: [(socket.AF_INET, 0, 0, "", (address, port))],
    )


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.5", "169.254.169.254",
                                     "192.168.1.1", "::1"])
def test_private_addresses_are_refused(monkeypatch, address):
    _resolve_to(monkeypatch, address)
    with pytest.raises(sources.UnsafeURL):
        sources.check_public_url("http://looks-innocent.example/paper.pdf")


def test_non_http_schemes_are_refused():
    with pytest.raises(sources.UnsafeURL):
        sources.check_public_url("file:///etc/passwd")


def test_a_redirect_to_a_private_address_is_refused(monkeypatch):
    """Checking only the first URL is not enough: every hop is checked."""
    def resolve(host, port):
        address = "10.0.0.1" if host == "internal.example" else "93.184.216.34"
        return [(socket.AF_INET, 0, 0, "", (address, port))]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(
        sources.requests, "get",
        lambda url, **kw: FakeResponse(
            status=302, headers={"Location": "http://internal.example/secret"}),
    )
    with pytest.raises(sources.UnsafeURL):
        sources.fetch_pdf("https://public.example/paper.pdf")


def test_fetch_returns_a_pdf_and_rejects_anything_else(monkeypatch):
    _resolve_to(monkeypatch, "93.184.216.34")
    monkeypatch.setattr(sources.requests, "get",
                        lambda url, **kw: FakeResponse(content=b"%PDF-1.7 body"))
    assert sources.fetch_pdf("https://public.example/a.pdf") == b"%PDF-1.7 body"

    monkeypatch.setattr(sources.requests, "get",
                        lambda url, **kw: FakeResponse(content=b"<html>login</html>"))
    assert sources.fetch_pdf("https://public.example/a.pdf") is None
