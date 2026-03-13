"""
RAG (Retrieval-Augmented Generation) retrieval layer.

Searches ChromaDB for relevant SEC filing chunks and news articles,
then formats them into context blocks for LLM prompt injection.
"""

from __future__ import annotations

from marketmind.db.vector_store import VectorStore


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text. Rough approximation: 1.3 tokens per word."""
    if not text:
        return 0
    return int(len(text.split()) * 1.3)


class RAGRetriever:
    """
    Retrieves relevant document chunks from the vector store
    and formats them as prompt context for LLM analysis.
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self.vs = vector_store

    # ──────────────────────────────────────────────
    # Retrieval methods
    # ──────────────────────────────────────────────

    def retrieve_for_stock_analysis(
        self, ticker: str, query: str | None = None
    ) -> dict:
        """
        Retrieve relevant documents for a comprehensive stock analysis.

        Searches SEC filings by section (MD&A, risk factors, business overview,
        material events, insider trades) and news, all filtered to this ticker.

        Args:
            ticker: Stock ticker symbol.
            query: Optional specific query for targeted retrieval.

        Returns:
            Dict with sec_filings, news, query_specific, and metadata keys.
        """
        ticker = ticker.upper()

        # Section-specific searches against sec_filings collection
        sec_filings = {}
        section_queries = {
            "mdna": ("management discussion revenue growth strategy outlook", 5),
            "risk_factors": ("risk factors challenges threats", 3),
            "business_overview": ("business overview operations products services", 2),
            "material_events": ("material event announcement", 3),
            "insider_trades": ("insider trading executive shares", 2),
        }

        # Map section query keys to the actual metadata section values
        section_metadata_map = {
            "mdna": "mdna",
            "risk_factors": "risk_factors",
            "business_overview": "business_overview",
            "material_events": "material_event",
            "insider_trades": "insider_trade",
        }

        all_dates: list[str] = []

        for section_key, (search_query, n_results) in section_queries.items():
            section_value = section_metadata_map[section_key]
            results = self.vs.search(
                query=search_query,
                collection="sec_filings",
                n_results=n_results,
                filters={"$and": [{"ticker": ticker}, {"section": section_value}]},
            )
            sec_filings[section_key] = [r["text"] for r in results]
            for r in results:
                date = r["metadata"].get("date", "")
                if date and date not in all_dates:
                    all_dates.append(date)

        # News search
        news_results = self.vs.search(
            query=f"{ticker} stock news recent developments",
            collection="news",
            n_results=5,
            filters={"ticker": ticker},
        )
        news_texts = [r["text"] for r in news_results]
        news_dates: list[str] = []
        for r in news_results:
            date = r["metadata"].get("date", "")
            if date and date not in news_dates:
                news_dates.append(date)

        # Query-specific search (across both collections)
        query_specific: list[str] = []
        if query:
            for coll in ["sec_filings", "news"]:
                results = self.vs.search(
                    query=query,
                    collection=coll,
                    n_results=3,
                    filters={"ticker": ticker},
                )
                query_specific.extend(r["text"] for r in results)

        total_chunks = (
            sum(len(v) for v in sec_filings.values())
            + len(news_texts)
            + len(query_specific)
        )

        return {
            "sec_filings": sec_filings,
            "news": news_texts,
            "query_specific": query_specific,
            "metadata": {
                "total_chunks_retrieved": total_chunks,
                "sec_filing_dates": sorted(all_dates),
                "news_dates": sorted(news_dates),
            },
        }

    def retrieve_for_portfolio_analysis(self, tickers: list[str]) -> dict:
        """
        Retrieve lighter context for portfolio-level analysis.

        For each ticker, retrieves 2 MD&A chunks and 2 news chunks.

        Args:
            tickers: List of portfolio ticker symbols.

        Returns:
            Dict keyed by ticker, each with "mdna" and "news" lists.
        """
        result: dict[str, dict] = {}

        for ticker in tickers:
            t = ticker.upper()
            mdna_results = self.vs.search(
                query="management discussion revenue growth outlook",
                collection="sec_filings",
                n_results=2,
                filters={"$and": [{"ticker": t}, {"section": "mdna"}]},
            )
            news_results = self.vs.search(
                query=f"{t} stock news recent",
                collection="news",
                n_results=2,
                filters={"ticker": t},
            )
            result[t] = {
                "mdna": [r["text"] for r in mdna_results],
                "news": [r["text"] for r in news_results],
            }

        return result

    def retrieve_for_potential_buy(
        self, ticker: str, portfolio_tickers: list[str]
    ) -> dict:
        """
        Retrieve context for evaluating a potential stock purchase.

        Full retrieval for the target ticker, plus sector comparison context
        from existing portfolio holdings in the same sector.

        Args:
            ticker: Candidate stock ticker.
            portfolio_tickers: Current portfolio ticker symbols.

        Returns:
            Dict with "target" (full retrieval) and "sector_comparison" keys.
        """
        # Full retrieval for the target
        target = self.retrieve_for_stock_analysis(ticker)

        # Sector comparison: find portfolio tickers in the same sector
        sector_context: dict[str, list[str]] = {}
        target_sector = _get_sector(ticker)

        if target_sector and portfolio_tickers:
            for pt in portfolio_tickers:
                pt = pt.upper()
                if pt == ticker.upper():
                    continue
                pt_sector = _get_sector(pt)
                if pt_sector and pt_sector == target_sector:
                    results = self.vs.search(
                        query="management discussion revenue growth outlook",
                        collection="sec_filings",
                        n_results=1,
                        filters={"$and": [{"ticker": pt}, {"section": "mdna"}]},
                    )
                    if results:
                        sector_context[pt] = [r["text"] for r in results]

        return {
            "target": target,
            "sector_comparison": sector_context,
        }

    # ──────────────────────────────────────────────
    # Formatting
    # ──────────────────────────────────────────────

    def format_context_for_prompt(
        self, retrieved: dict, max_words: int = 3000
    ) -> str:
        """
        Convert retrieved chunks into a formatted text block for prompt injection.

        Prioritizes MD&A and news (most valuable), then trims other sections
        if the total exceeds max_words.

        Args:
            retrieved: Dict from retrieve_for_stock_analysis.
            max_words: Maximum word count for the output block.

        Returns:
            Formatted string, or empty string if no content.
        """
        sec = retrieved.get("sec_filings", {})
        news = retrieved.get("news", [])
        query_specific = retrieved.get("query_specific", [])

        # Check if we have any content at all
        has_sec = any(sec.get(k) for k in sec)
        has_news = bool(news)
        has_query = bool(query_specific)

        if not has_sec and not has_news and not has_query:
            return ""

        # Build sections in priority order (highest value first)
        # We'll track word count and trim lower-priority sections if needed
        sections: list[tuple[str, str]] = []

        # Priority 1: MD&A
        if sec.get("mdna"):
            sections.append(("### Management Discussion & Analysis (most recent filings)",
                             "\n\n".join(sec["mdna"])))

        # Priority 2: News
        if news:
            sections.append(("### Recent News", "\n\n".join(news)))

        # Priority 3: Query-specific results
        if query_specific:
            sections.append(("### Query-Specific Context", "\n\n".join(query_specific)))

        # Priority 4: Risk factors
        if sec.get("risk_factors"):
            sections.append(("### Risk Factors", "\n\n".join(sec["risk_factors"])))

        # Priority 5: Material events
        if sec.get("material_events"):
            sections.append(("### Material Events (8-K filings)", "\n\n".join(sec["material_events"])))

        # Priority 6: Business overview
        if sec.get("business_overview"):
            sections.append(("### Business Overview", "\n\n".join(sec["business_overview"])))

        # Priority 7: Insider trades
        if sec.get("insider_trades"):
            sections.append(("### Insider Trading Activity", "\n\n".join(sec["insider_trades"])))

        # Assemble with word budget enforcement
        output_parts: list[str] = []
        words_used = 0

        for heading, content in sections:
            section_words = len(content.split())
            if words_used + section_words > max_words and output_parts:
                # Trim this section to fit remaining budget
                remaining = max_words - words_used
                if remaining > 50:
                    trimmed = " ".join(content.split()[:remaining])
                    output_parts.append(f"{heading}\n{trimmed}...")
                break
            output_parts.append(f"{heading}\n{content}")
            words_used += section_words

        if not output_parts:
            return ""

        return "\n\n".join(output_parts)

    def format_portfolio_context(self, retrieved: dict, max_words: int = 2000) -> str:
        """
        Format per-ticker portfolio context for the portfolio analysis prompt.

        Args:
            retrieved: Dict from retrieve_for_portfolio_analysis, keyed by ticker.
            max_words: Maximum total word count.

        Returns:
            Formatted string, or empty string if no content.
        """
        if not retrieved:
            return ""

        parts: list[str] = []
        words_used = 0
        # Budget per ticker (rough split)
        per_ticker_budget = max(100, max_words // max(len(retrieved), 1))

        for ticker, data in sorted(retrieved.items()):
            ticker_parts: list[str] = []

            if data.get("mdna"):
                ticker_parts.append("Filing context: " + " ".join(data["mdna"]))
            if data.get("news"):
                ticker_parts.append("Recent news: " + " ".join(data["news"]))

            if not ticker_parts:
                continue

            combined = "\n".join(ticker_parts)
            section_words = len(combined.split())

            # Trim if over per-ticker budget
            if section_words > per_ticker_budget:
                combined = " ".join(combined.split()[:per_ticker_budget]) + "..."
                section_words = per_ticker_budget

            if words_used + section_words > max_words:
                break

            parts.append(f"**{ticker}**\n{combined}")
            words_used += section_words

        return "\n\n".join(parts) if parts else ""

    def format_potential_buy_context(
        self, retrieved: dict, max_words: int = 3500
    ) -> str:
        """
        Format context for a potential buy evaluation.

        Includes full target context plus sector comparison.

        Args:
            retrieved: Dict from retrieve_for_potential_buy.
            max_words: Maximum total word count.

        Returns:
            Formatted string, or empty string if no content.
        """
        target_context = self.format_context_for_prompt(
            retrieved.get("target", {}), max_words=2500
        )

        sector = retrieved.get("sector_comparison", {})
        sector_parts: list[str] = []
        if sector:
            for ticker, chunks in sorted(sector.items()):
                if chunks:
                    sector_parts.append(f"**{ticker}** (same sector):\n" + "\n".join(chunks))

        parts: list[str] = []
        if target_context:
            parts.append(target_context)
        if sector_parts:
            sector_text = "\n\n".join(sector_parts)
            # Enforce remaining word budget
            target_words = len(target_context.split()) if target_context else 0
            remaining = max_words - target_words
            if remaining > 50:
                sector_words = len(sector_text.split())
                if sector_words > remaining:
                    sector_text = " ".join(sector_text.split()[:remaining]) + "..."
                parts.append(f"### Sector Comparison (existing holdings)\n{sector_text}")

        return "\n\n".join(parts) if parts else ""


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


# Cache for sector lookups
_sector_cache: dict[str, str] = {}


def _get_sector(ticker: str) -> str:
    """Look up a stock's sector via yfinance. Cached per session."""
    ticker = ticker.upper()
    if ticker in _sector_cache:
        return _sector_cache[ticker]

    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        sector = info.get("sector", "")
        _sector_cache[ticker] = sector
        return sector
    except Exception:
        _sector_cache[ticker] = ""
        return ""
