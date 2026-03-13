"""
Phase 3B tests -- RAG retrieval and context formatting.

Offline tests (no network, no API keys): retrieval, formatting, token estimation,
and prompt integration.
"""

import chromadb
import pytest

from marketmind.agent.rag import RAGRetriever, _estimate_tokens
from marketmind.db.vector_store import VectorStore
from marketmind.prompts.templates import (
    COMPREHENSIVE_ANALYSIS_PROMPT,
    HOLDING_ANALYSIS_PROMPT,
    PORTFOLIO_ANALYSIS_PROMPT,
    POTENTIAL_BUY_PROMPT,
    RAG_CONTEXT_BLOCK,
    RAG_PORTFOLIO_CONTEXT_BLOCK,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def ephemeral_vs(tmp_path):
    """VectorStore backed by an in-memory ChromaDB client for test isolation."""
    client = chromadb.EphemeralClient()
    for name in ["sec_filings", "news"]:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    return VectorStore(data_dir=tmp_path, client=client)


@pytest.fixture
def populated_vs(ephemeral_vs):
    """VectorStore with sample documents for AAPL and MSFT."""
    # SEC filings: MD&A
    mdna_docs = [
        {
            "text": "Apple reported record revenue of $124 billion in Q1 2024, "
                    "driven by strong iPhone sales and services growth.",
            "metadata": {
                "ticker": "AAPL", "doc_type": "10-K", "section": "mdna",
                "date": "2024-11-01", "source": "sec_edgar",
            },
        },
        {
            "text": "Services revenue reached $24 billion, representing 20% year-over-year "
                    "growth. Management expects continued momentum in subscriptions.",
            "metadata": {
                "ticker": "AAPL", "doc_type": "10-K", "section": "mdna",
                "date": "2024-11-01", "source": "sec_edgar",
            },
        },
        {
            "text": "Microsoft Azure revenue grew 30% year over year. Cloud segment "
                    "is now the largest revenue driver for the company.",
            "metadata": {
                "ticker": "MSFT", "doc_type": "10-K", "section": "mdna",
                "date": "2024-10-15", "source": "sec_edgar",
            },
        },
    ]

    # SEC filings: Risk factors
    risk_docs = [
        {
            "text": "Apple faces supply chain concentration risk. A significant portion "
                    "of manufacturing is concentrated in China and Southeast Asia.",
            "metadata": {
                "ticker": "AAPL", "doc_type": "10-K", "section": "risk_factors",
                "date": "2024-11-01", "source": "sec_edgar",
            },
        },
    ]

    # SEC filings: Business overview
    biz_docs = [
        {
            "text": "Apple designs, manufactures, and markets smartphones, personal "
                    "computers, tablets, wearables, and accessories worldwide.",
            "metadata": {
                "ticker": "AAPL", "doc_type": "10-K", "section": "business_overview",
                "date": "2024-11-01", "source": "sec_edgar",
            },
        },
    ]

    # SEC filings: Material events
    event_docs = [
        {
            "text": "Apple announced a $110 billion share buyback program, the largest "
                    "in corporate history.",
            "metadata": {
                "ticker": "AAPL", "doc_type": "8-K", "section": "material_event",
                "date": "2024-05-02", "source": "sec_edgar",
            },
        },
    ]

    # SEC filings: Insider trades
    insider_docs = [
        {
            "text": "Tim Cook sold 50,000 shares at $185 per share under a 10b5-1 plan.",
            "metadata": {
                "ticker": "AAPL", "doc_type": "Form4", "section": "insider_trade",
                "date": "2024-08-15", "source": "sec_edgar",
            },
        },
    ]

    ephemeral_vs.add_documents(
        mdna_docs + risk_docs + biz_docs + event_docs + insider_docs,
        collection="sec_filings",
    )

    # News articles
    news_docs = [
        {
            "text": "Apple stock hits all-time high as iPhone 16 demand exceeds expectations.",
            "metadata": {
                "ticker": "AAPL", "doc_type": "news", "section": "news_article",
                "date": "2024-12-01", "source": "rss_yahoo_headline_AAPL",
            },
        },
        {
            "text": "Apple expands AI features in iOS 18.2, analysts see significant upside.",
            "metadata": {
                "ticker": "AAPL", "doc_type": "news", "section": "news_article",
                "date": "2024-12-05", "source": "rss_google_news_AAPL",
            },
        },
        {
            "text": "Microsoft announces new Copilot features for enterprise customers.",
            "metadata": {
                "ticker": "MSFT", "doc_type": "news", "section": "news_article",
                "date": "2024-12-03", "source": "rss_yahoo_headline_MSFT",
            },
        },
    ]
    ephemeral_vs.add_documents(news_docs, collection="news")

    return ephemeral_vs


# ──────────────────────────────────────────────
# RAGRetriever Tests
# ──────────────────────────────────────────────


class TestRAGRetriever:
    def test_retrieve_filters_by_ticker(self, populated_vs):
        """Retrieval should only return documents for the requested ticker."""
        retriever = RAGRetriever(populated_vs)
        result = retriever.retrieve_for_stock_analysis("AAPL")

        # Should have AAPL MD&A content, not MSFT
        mdna_text = " ".join(result["sec_filings"]["mdna"])
        assert "Apple" in mdna_text or "iPhone" in mdna_text or "Services" in mdna_text
        assert "Azure" not in mdna_text

    def test_retrieve_separates_sections(self, populated_vs):
        """Retrieved context should be organized by section."""
        retriever = RAGRetriever(populated_vs)
        result = retriever.retrieve_for_stock_analysis("AAPL")

        assert "mdna" in result["sec_filings"]
        assert "risk_factors" in result["sec_filings"]
        assert "business_overview" in result["sec_filings"]
        assert "material_events" in result["sec_filings"]
        assert "insider_trades" in result["sec_filings"]

        # MD&A should have content
        assert len(result["sec_filings"]["mdna"]) > 0

        # Risk factors should mention supply chain
        if result["sec_filings"]["risk_factors"]:
            risk_text = " ".join(result["sec_filings"]["risk_factors"])
            assert "supply chain" in risk_text.lower() or "China" in risk_text

    def test_retrieve_includes_news(self, populated_vs):
        """Retrieval should include news articles for the ticker."""
        retriever = RAGRetriever(populated_vs)
        result = retriever.retrieve_for_stock_analysis("AAPL")

        assert len(result["news"]) > 0
        news_text = " ".join(result["news"])
        assert "Apple" in news_text or "iPhone" in news_text

    def test_retrieve_with_query(self, populated_vs):
        """Query-specific retrieval should return relevant results."""
        retriever = RAGRetriever(populated_vs)
        result = retriever.retrieve_for_stock_analysis("AAPL", query="services revenue growth")

        assert len(result["query_specific"]) > 0

    def test_retrieve_metadata(self, populated_vs):
        """Metadata should include chunk counts and dates."""
        retriever = RAGRetriever(populated_vs)
        result = retriever.retrieve_for_stock_analysis("AAPL")

        meta = result["metadata"]
        assert meta["total_chunks_retrieved"] > 0
        assert isinstance(meta["sec_filing_dates"], list)
        assert isinstance(meta["news_dates"], list)

    def test_retrieve_empty_ticker(self, populated_vs):
        """Retrieval for a ticker with no documents should return empty gracefully."""
        retriever = RAGRetriever(populated_vs)
        result = retriever.retrieve_for_stock_analysis("NVDA")

        assert result["sec_filings"]["mdna"] == []
        assert result["sec_filings"]["risk_factors"] == []
        assert result["news"] == []
        assert result["metadata"]["total_chunks_retrieved"] == 0

    def test_retrieve_portfolio(self, populated_vs):
        """Portfolio retrieval should return per-ticker results."""
        retriever = RAGRetriever(populated_vs)
        result = retriever.retrieve_for_portfolio_analysis(["AAPL", "MSFT"])

        assert "AAPL" in result
        assert "MSFT" in result
        assert "mdna" in result["AAPL"]
        assert "news" in result["AAPL"]
        assert len(result["AAPL"]["mdna"]) > 0

    def test_retrieve_portfolio_empty_ticker(self, populated_vs):
        """Portfolio retrieval for tickers without documents should return empty lists."""
        retriever = RAGRetriever(populated_vs)
        result = retriever.retrieve_for_portfolio_analysis(["AAPL", "NVDA"])

        assert len(result["AAPL"]["mdna"]) > 0
        assert result["NVDA"]["mdna"] == []
        assert result["NVDA"]["news"] == []


# ──────────────────────────────────────────────
# Context Formatting Tests
# ──────────────────────────────────────────────


class TestContextFormatting:
    def test_format_produces_readable_output(self, populated_vs):
        """Formatted context should be readable with section headings."""
        retriever = RAGRetriever(populated_vs)
        retrieved = retriever.retrieve_for_stock_analysis("AAPL")
        formatted = retriever.format_context_for_prompt(retrieved)

        assert "### Management Discussion" in formatted
        assert len(formatted) > 100

    def test_format_respects_max_words(self, populated_vs):
        """Formatted output should not exceed the word limit."""
        retriever = RAGRetriever(populated_vs)
        retrieved = retriever.retrieve_for_stock_analysis("AAPL")
        formatted = retriever.format_context_for_prompt(retrieved, max_words=50)

        word_count = len(formatted.split())
        # Allow some headroom for headings
        assert word_count < 80

    def test_format_empty_returns_empty_string(self, ephemeral_vs):
        """No documents should produce an empty string, not empty headings."""
        retriever = RAGRetriever(ephemeral_vs)
        retrieved = retriever.retrieve_for_stock_analysis("NVDA")
        formatted = retriever.format_context_for_prompt(retrieved)

        assert formatted == ""

    def test_format_portfolio_context(self, populated_vs):
        """Portfolio context should include per-ticker sections."""
        retriever = RAGRetriever(populated_vs)
        retrieved = retriever.retrieve_for_portfolio_analysis(["AAPL", "MSFT"])
        formatted = retriever.format_portfolio_context(retrieved)

        assert "**AAPL**" in formatted
        assert "**MSFT**" in formatted

    def test_format_portfolio_empty(self, ephemeral_vs):
        """Empty portfolio retrieval should produce empty string."""
        retriever = RAGRetriever(ephemeral_vs)
        retrieved = retriever.retrieve_for_portfolio_analysis(["NVDA"])
        formatted = retriever.format_portfolio_context(retrieved)

        assert formatted == ""


# ──────────────────────────────────────────────
# Token Estimation Tests
# ──────────────────────────────────────────────


class TestTokenEstimation:
    def test_empty_text(self):
        """Empty text should return 0 tokens."""
        assert _estimate_tokens("") == 0
        assert _estimate_tokens(None) == 0

    def test_reasonable_estimate(self):
        """Token estimate should be roughly 1.3x word count."""
        text = "The quick brown fox jumps over the lazy dog"  # 9 words
        estimate = _estimate_tokens(text)
        assert 10 <= estimate <= 15  # 9 * 1.3 = 11.7

    def test_longer_text(self):
        """Longer text estimates should scale proportionally."""
        text = " ".join(f"word{i}" for i in range(100))
        estimate = _estimate_tokens(text)
        assert 120 <= estimate <= 140  # 100 * 1.3 = 130


# ──────────────────────────────────────────────
# Prompt Integration Tests
# ──────────────────────────────────────────────


class TestPromptIntegration:
    def test_comprehensive_prompt_includes_rag(self, populated_vs):
        """Comprehensive analysis prompt should include RAG context when provided."""
        retriever = RAGRetriever(populated_vs)
        retrieved = retriever.retrieve_for_stock_analysis("AAPL")
        raw_context = retriever.format_context_for_prompt(retrieved)
        rag_block = RAG_CONTEXT_BLOCK.format(rag_content=raw_context)

        prompt = COMPREHENSIVE_ANALYSIS_PROMPT.format(
            ticker="AAPL",
            price_data="Price: $195",
            fundamental_data="P/E: 30",
            history_period="3 months",
            history_data="Up 10%",
            analyst_data="Buy consensus",
            rag_context=rag_block,
        )

        assert "SEC filings" in prompt
        assert "trust the context" in prompt
        assert "Apple" in prompt

    def test_comprehensive_prompt_omits_rag_when_empty(self):
        """Prompt should work cleanly with empty RAG context."""
        prompt = COMPREHENSIVE_ANALYSIS_PROMPT.format(
            ticker="TSLA",
            price_data="Price: $250",
            fundamental_data="P/E: 60",
            history_period="3 months",
            history_data="Down 5%",
            analyst_data="Hold consensus",
            rag_context="",
        )

        assert "SEC filings" not in prompt
        assert "trust the context" not in prompt
        assert "TSLA" in prompt

    def test_portfolio_prompt_includes_rag(self, populated_vs):
        """Portfolio prompt should include per-holding RAG context."""
        retriever = RAGRetriever(populated_vs)
        retrieved = retriever.retrieve_for_portfolio_analysis(["AAPL"])
        raw_context = retriever.format_portfolio_context(retrieved)
        rag_block = RAG_PORTFOLIO_CONTEXT_BLOCK.format(rag_content=raw_context)

        prompt = PORTFOLIO_ANALYSIS_PROMPT.format(
            holdings_data="AAPL: 100 shares",
            market_data="AAPL: $195",
            num_holdings=1,
            total_value=19500.00,
            total_cost=18000.00,
            total_pnl=1500.00,
            total_pnl_pct=8.33,
            rag_context=rag_block,
            educational_section="",
        )

        assert "SEC filings" in prompt or "Filing context" in prompt

    def test_holding_prompt_accepts_rag(self):
        """Holding analysis prompt should accept rag_context placeholder."""
        prompt = HOLDING_ANALYSIS_PROMPT.format(
            ticker="AAPL",
            shares=100,
            avg_cost=180.0,
            total_cost=18000.0,
            current_price=195.0,
            market_value=19500.0,
            pnl=1500.0,
            pnl_pct=8.33,
            fundamental_data="P/E: 30",
            history_data="Up 10%",
            analyst_data="Buy",
            position_pct=25.0,
            portfolio_value=78000.0,
            rag_context="",
            educational_section="",
        )
        assert "AAPL" in prompt

    def test_potential_buy_prompt_accepts_rag(self):
        """Potential buy prompt should accept rag_context placeholder."""
        prompt = POTENTIAL_BUY_PROMPT.format(
            ticker="NVDA",
            price_data="Price: $800",
            fundamental_data="P/E: 65",
            history_data="Up 30%",
            analyst_data="Strong Buy",
            portfolio_summary="5 holdings, $100k total",
            rag_context="",
            educational_section="",
        )
        assert "NVDA" in prompt
