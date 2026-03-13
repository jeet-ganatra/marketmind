"""
Phase 3A tests -- Document ingestion pipeline.

Offline tests (no network): chunking, vector store, news filtering.
Network tests (marked with pytest.mark.network): EdgarTools, RSS feeds.
"""

import pytest

import chromadb

from marketmind.db.vector_store import VectorStore, chunk_text


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def ephemeral_vs(tmp_path):
    """VectorStore backed by an in-memory ChromaDB client for test isolation."""
    client = chromadb.EphemeralClient()
    # Clean up any leftover collections from previous tests
    # (EphemeralClient may share state within the same process)
    for name in ["sec_filings", "news"]:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    return VectorStore(data_dir=tmp_path, client=client)


# ──────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────


class TestChunking:
    def test_basic_chunking(self):
        """Text longer than chunk_size should be split into multiple chunks."""
        text = " ".join(f"word{i}" for i in range(1000))
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) >= 2
        # Each chunk should be around 500 words (except possibly the last)
        for chunk in chunks[:-1]:
            assert len(chunk.split()) == 500

    def test_overlap(self):
        """Consecutive chunks should share overlapping words."""
        text = " ".join(f"word{i}" for i in range(600))
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) == 2

        first_words = chunks[0].split()
        second_words = chunks[1].split()
        # Last 50 words of first chunk should appear at start of second chunk
        assert first_words[-50:] == second_words[:50]

    def test_short_text_no_chunking(self):
        """Text within chunk_size should return as single chunk."""
        text = " ".join(f"word{i}" for i in range(200))
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text(self):
        """Empty or whitespace-only text should return empty list."""
        assert chunk_text("") == []
        assert chunk_text("   ") == []
        assert chunk_text(None) == []

    def test_minimum_chunk_size(self):
        """Chunks under 50 words should be skipped."""
        text = " ".join(f"word{i}" for i in range(60))
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) == 1

    def test_tiny_text_below_minimum(self):
        """Text under 50 words should return empty list."""
        text = " ".join(f"word{i}" for i in range(30))
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) == 0

    def test_trailing_tiny_chunk_skipped(self):
        """A tiny trailing chunk (<50 words) should be dropped."""
        # 520 words with chunk_size=500, overlap=50: first chunk = 500 words,
        # second chunk starts at word 450, gets 70 words (520-450).
        # 70 >= 50 so it should be kept.
        text = " ".join(f"word{i}" for i in range(520))
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) == 2

        # Now 510 words: second chunk starts at 450, gets 60 words. 60 >= 50, kept.
        text = " ".join(f"word{i}" for i in range(510))
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) == 2

        # 480 words: second chunk starts at 450, gets 30 words. 30 < 50, dropped.
        text = " ".join(f"word{i}" for i in range(480))
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        # 480 < 500, so single chunk returned
        assert len(chunks) == 1


# ──────────────────────────────────────────────
# Vector Store
# ──────────────────────────────────────────────


class TestVectorStore:
    def test_add_and_search(self, ephemeral_vs):
        """Documents added to the store should be retrievable via search."""
        docs = [
            {
                "text": "Apple reported strong Q4 earnings with revenue growth in services.",
                "metadata": {
                    "ticker": "AAPL",
                    "doc_type": "10-K",
                    "section": "mdna",
                    "date": "2024-11-01",
                    "source": "sec_edgar",
                },
            },
            {
                "text": "Microsoft Azure cloud revenue grew 30% year over year.",
                "metadata": {
                    "ticker": "MSFT",
                    "doc_type": "10-Q",
                    "section": "mdna",
                    "date": "2024-10-15",
                    "source": "sec_edgar",
                },
            },
        ]

        count = ephemeral_vs.add_documents(docs, collection="sec_filings")
        assert count == 2

        results = ephemeral_vs.search("Apple earnings", collection="sec_filings", n_results=2)
        assert len(results) > 0
        assert results[0]["metadata"]["ticker"] == "AAPL"

    def test_search_with_metadata_filter(self, ephemeral_vs):
        """Search should respect metadata filters."""
        docs = [
            {
                "text": "Apple revenue grew significantly in the fiscal year.",
                "metadata": {"ticker": "AAPL", "doc_type": "10-K", "section": "mdna",
                             "date": "2024-11-01", "source": "sec_edgar"},
            },
            {
                "text": "Google revenue grew significantly in the fiscal year.",
                "metadata": {"ticker": "GOOGL", "doc_type": "10-K", "section": "mdna",
                             "date": "2024-11-01", "source": "sec_edgar"},
            },
        ]
        ephemeral_vs.add_documents(docs, collection="sec_filings")

        results = ephemeral_vs.search(
            "revenue growth", collection="sec_filings",
            n_results=5, filters={"ticker": "GOOGL"},
        )
        assert len(results) > 0
        assert all(r["metadata"]["ticker"] == "GOOGL" for r in results)

    def test_has_document(self, ephemeral_vs):
        """has_document should detect existing documents."""
        docs = [
            {
                "text": "Test content for duplicate detection.",
                "metadata": {"ticker": "AAPL", "doc_type": "10-K", "section": "mdna",
                             "date": "2024-01-01", "source": "sec_edgar"},
            },
        ]
        ephemeral_vs.add_documents(docs, collection="sec_filings")

        assert ephemeral_vs.has_document("AAPL", "10-K", "2024-01-01", "sec_filings") is True
        assert ephemeral_vs.has_document("AAPL", "10-K", "2025-01-01", "sec_filings") is False
        assert ephemeral_vs.has_document("MSFT", "10-K", "2024-01-01", "sec_filings") is False

    def test_get_document_count(self, ephemeral_vs):
        """get_document_count should return correct counts."""
        docs = [
            {"text": "Apple doc one.", "metadata": {"ticker": "AAPL", "doc_type": "10-K",
                                                     "section": "mdna", "date": "2024-01-01",
                                                     "source": "sec_edgar"}},
            {"text": "Apple doc two.", "metadata": {"ticker": "AAPL", "doc_type": "10-K",
                                                     "section": "risk_factors", "date": "2024-01-01",
                                                     "source": "sec_edgar"}},
            {"text": "Microsoft doc.", "metadata": {"ticker": "MSFT", "doc_type": "10-Q",
                                                     "section": "mdna", "date": "2024-01-01",
                                                     "source": "sec_edgar"}},
        ]
        ephemeral_vs.add_documents(docs, collection="sec_filings")

        assert ephemeral_vs.get_document_count("sec_filings") == 3
        assert ephemeral_vs.get_document_count("sec_filings", ticker="AAPL") == 2
        assert ephemeral_vs.get_document_count("sec_filings", ticker="MSFT") == 1
        assert ephemeral_vs.get_document_count("sec_filings", ticker="NVDA") == 0

    def test_delete_documents(self, ephemeral_vs):
        """delete_documents should remove all documents for a ticker."""
        docs = [
            {"text": "Apple doc.", "metadata": {"ticker": "AAPL", "doc_type": "10-K",
                                                 "section": "mdna", "date": "2024-01-01",
                                                 "source": "sec_edgar"}},
            {"text": "Microsoft doc.", "metadata": {"ticker": "MSFT", "doc_type": "10-Q",
                                                     "section": "mdna", "date": "2024-01-01",
                                                     "source": "sec_edgar"}},
        ]
        ephemeral_vs.add_documents(docs, collection="sec_filings")
        assert ephemeral_vs.get_document_count("sec_filings") == 2

        ephemeral_vs.delete_documents("AAPL", collection="sec_filings")
        assert ephemeral_vs.get_document_count("sec_filings") == 1
        assert ephemeral_vs.get_document_count("sec_filings", ticker="AAPL") == 0
        assert ephemeral_vs.get_document_count("sec_filings", ticker="MSFT") == 1

    def test_upsert_idempotency(self, ephemeral_vs):
        """Adding the same documents twice should not create duplicates."""
        docs = [
            {"text": "Same content.", "metadata": {"ticker": "AAPL", "doc_type": "10-K",
                                                    "section": "mdna", "date": "2024-01-01",
                                                    "source": "sec_edgar"}},
        ]
        ephemeral_vs.add_documents(docs, collection="sec_filings")
        ephemeral_vs.add_documents(docs, collection="sec_filings")

        assert ephemeral_vs.get_document_count("sec_filings") == 1

    def test_empty_documents(self, ephemeral_vs):
        """Adding an empty list should return 0."""
        assert ephemeral_vs.add_documents([], collection="sec_filings") == 0


# ──────────────────────────────────────────────
# News Filtering
# ──────────────────────────────────────────────


class TestNewsFiltering:
    def test_article_matches_ticker(self):
        """Articles containing ticker symbols should match."""
        from marketmind.tools.news_feeds import _article_matches_tickers

        article = {"title": "AAPL hits all-time high", "summary": "Apple stock surges."}
        assert _article_matches_tickers(article, ["AAPL"], {"AAPL": "Apple Inc."}) is True
        assert _article_matches_tickers(article, ["MSFT"], {"MSFT": "Microsoft"}) is False

    def test_article_matches_company_name(self):
        """Articles containing company names should match."""
        from marketmind.tools.news_feeds import _article_matches_tickers

        article = {"title": "Tech rally continues", "summary": "Apple Inc. led the gains."}
        assert _article_matches_tickers(article, ["AAPL"], {"AAPL": "Apple Inc."}) is True

    def test_article_no_match(self):
        """Articles not mentioning any ticker or name should not match."""
        from marketmind.tools.news_feeds import _article_matches_tickers

        article = {"title": "Oil prices surge", "summary": "Crude oil hit $80 per barrel."}
        assert _article_matches_tickers(article, ["AAPL", "MSFT"], {"AAPL": "Apple", "MSFT": "Microsoft"}) is False


# ──────────────────────────────────────────────
# Network Tests (require internet access)
# ──────────────────────────────────────────────


@pytest.mark.network
class TestEdgarToolsIntegration:
    def test_fetch_aapl_10k(self):
        """EdgarTools should be able to fetch AAPL's latest 10-K."""
        from edgar import Company, set_identity
        set_identity("MarketMind test@example.com")

        company = Company("AAPL")
        assert company.name is not None

        filings = company.get_filings(form="10-K").head(1)
        assert len(list(filings)) > 0

    def test_extract_mdna_section(self):
        """Should be able to extract MD&A section from a 10-K."""
        from edgar import Company, set_identity
        set_identity("MarketMind test@example.com")

        company = Company("AAPL")
        filing = company.get_filings(form="10-K").latest()
        tenk = filing.obj()

        mdna = getattr(tenk, "management_discussion", None)
        assert mdna is not None
        text = str(mdna)
        assert len(text) > 100


@pytest.mark.network
class TestRSSFeeds:
    def test_fetch_news_returns_articles(self):
        """At least one RSS feed should return articles."""
        from marketmind.tools.news_feeds import fetch_news

        articles = fetch_news()
        # At least some feeds should work; we just need >0 articles
        assert len(articles) > 0

    def test_fetch_news_has_required_fields(self):
        """Articles should have title and source fields."""
        from marketmind.tools.news_feeds import fetch_news

        articles = fetch_news()
        if articles:
            article = articles[0]
            assert "title" in article
            assert "source" in article
            assert len(article["title"]) > 0
