"""
Agent Orchestrator — The brain of MarketMind.

This is the central module that:
1. Receives a user query (e.g., "analyze AAPL")
2. Fetches relevant data using tools
3. Assembles context for the LLM
4. Routes to the appropriate model
5. Returns structured results

Phase 1: Single-stock analysis pipeline.
Phase 2: Portfolio-aware analysis with caching.
"""

from __future__ import annotations

from rich.console import Console

from marketmind.config import Settings
from marketmind.llm.router import LLMRouter
from marketmind.models.schemas import AnalysisResult, Holding, StockAnalysisCache, StockPrice
from marketmind.prompts.templates import (
    COMPREHENSIVE_ANALYSIS_PROMPT,
    EDUCATIONAL_ANALYSIS_PROMPT,
    EDUCATIONAL_PORTFOLIO_SECTION,
    HOLDING_ANALYSIS_PROMPT,
    PORTFOLIO_ANALYSIS_PROMPT,
    PORTFOLIO_ANALYSIS_SYSTEM,
    POTENTIAL_BUY_PROMPT,
    QUICK_PRICE_PROMPT,
    RAG_CONTEXT_BLOCK,
    RAG_EDUCATIONAL_NOTE,
    RAG_PORTFOLIO_CONTEXT_BLOCK,
    RAG_POTENTIAL_BUY_CONTEXT_BLOCK,
    STOCK_ANALYSIS_SYSTEM,
)
from marketmind.tools.fundamentals import get_analyst_summary, get_fundamentals
from marketmind.tools.price import get_price_history, get_stock_price

console = Console()


class MarketMindAgent:
    """
    Main agent that orchestrates stock and portfolio analysis.

    Phase 1: fetch data → format prompt → call LLM → return result.
    Phase 2: adds portfolio context, analysis caching, and multi-method analysis.
    """

    def __init__(self, settings: Settings, db=None, vector_store=None) -> None:
        self.settings = settings
        self.router = LLMRouter(settings)
        self.db = db
        self._repo = None
        self.rag_retriever = None
        if vector_store is not None:
            from marketmind.agent.rag import RAGRetriever

            self.rag_retriever = RAGRetriever(vector_store)

        from marketmind.agent.validator import AnalysisValidator

        self.validator = AnalysisValidator(self.router)

    @property
    def repo(self):
        """Lazily create repository only when needed."""
        if self._repo is None and self.db is not None:
            from marketmind.db.repository import PortfolioRepository

            self._repo = PortfolioRepository(self.db)
        return self._repo

    # ──────────────────────────────────────────────
    # Phase 1: Single-stock analysis
    # ──────────────────────────────────────────────

    def analyze(self, ticker: str, explain: bool = False, validate: bool = True) -> AnalysisResult:
        """
        Run a comprehensive analysis on a stock.

        Args:
            ticker: Stock ticker symbol
            explain: If True, uses educational prompt with beginner-friendly explanations
        """
        ticker = ticker.upper()
        analysis_type = "educational" if explain else "comprehensive"
        mode_label = " (educational mode)" if explain else ""
        console.print(f"\n[bold blue]Analyzing {ticker}{mode_label}...[/bold blue]\n")

        # Check cache first
        if self.repo:
            cached = self.repo.get_recent_analysis(ticker, analysis_type)
            if cached:
                console.print("  [green]Using cached analysis (< 24h old)[/green]\n")
                return AnalysisResult(
                    ticker=ticker,
                    analysis_type=analysis_type,
                    summary=cached.summary,
                    detailed_analysis=cached.detailed_analysis,
                    data_sources=["cache"],
                    model_used=cached.model_used,
                    cost_usd=0,
                )

        # Step 1: Fetch all data
        console.print("  📊 Fetching price data...")
        price = get_stock_price(ticker)

        console.print("  📈 Fetching fundamentals...")
        fundamentals = get_fundamentals(ticker)

        console.print("  📉 Fetching price history...")
        history = get_price_history(ticker, period="3mo")

        console.print("  🎯 Fetching analyst ratings...")
        analysts = get_analyst_summary(ticker)

        # Step 2: Format data for the prompt
        price_str = _format_price(price)
        fundamentals_str = _format_fundamentals(fundamentals)
        history_str = _format_history(history)
        analyst_str = _format_analysts(analysts)

        # Step 2b: Retrieve RAG context if available
        rag_context = ""
        if self.rag_retriever:
            console.print("  📄 Retrieving SEC filings and news context...")
            retrieved = self.rag_retriever.retrieve_for_stock_analysis(ticker)
            raw_context = self.rag_retriever.format_context_for_prompt(
                retrieved, max_words=3000
            )
            if raw_context:
                block = RAG_CONTEXT_BLOCK.format(rag_content=raw_context)
                if explain:
                    block += RAG_EDUCATIONAL_NOTE
                rag_context = block
                n = retrieved["metadata"]["total_chunks_retrieved"]
                console.print(f"  [green]Found {n} relevant document chunks[/green]")
            else:
                console.print(
                    "  [dim]No ingested documents found for this ticker. "
                    "Run 'marketmind ingest filings {0}' for richer analysis.[/dim]".format(ticker)
                )

        template = EDUCATIONAL_ANALYSIS_PROMPT if explain else COMPREHENSIVE_ANALYSIS_PROMPT
        prompt = template.format(
            ticker=ticker,
            price_data=price_str,
            fundamental_data=fundamentals_str,
            history_period="3 months",
            history_data=history_str,
            analyst_data=analyst_str,
            rag_context=rag_context,
        )

        # Step 3: Call LLM (analysis task -> Claude Sonnet)
        console.print("  🤖 Running AI analysis (Claude Sonnet)...\n")
        result = self.router.call(
            prompt=prompt,
            system_prompt=STOCK_ANALYSIS_SYSTEM,
            task_type="analysis",
            max_tokens=8192 if explain else None,
        )

        # Step 4: Build structured result
        cost = result["cost_record"].cost_usd if result["cost_record"] else 0

        analysis_result = AnalysisResult(
            ticker=ticker,
            analysis_type=analysis_type,
            summary=_extract_summary(result["content"]),
            detailed_analysis=result["content"],
            data_sources=_list_data_sources(price, fundamentals, history, analysts)
            + (["chromadb:sec_filings", "chromadb:news"] if rag_context else []),
            model_used=result["model"],
            cost_usd=cost,
        )

        # Step 5: Validate the analysis
        if validate:
            provided_data = {
                "Price Data": price_str,
                "Fundamentals": fundamentals_str,
                "Price History": history_str,
                "Analyst Consensus": analyst_str,
            }
            analysis_result = _run_validation(
                self.validator, analysis_result, provided_data, rag_context
            )

        # Cache the result
        if self.repo:
            self.repo.save_analysis(StockAnalysisCache(
                ticker=ticker,
                analysis_type=analysis_type,
                summary=analysis_result.summary,
                detailed_analysis=analysis_result.detailed_analysis,
                model_used=analysis_result.model_used,
                cost_usd=analysis_result.cost_usd,
            ))

        return analysis_result

    def quick_price(self, ticker: str) -> str:
        """Quick price check — uses cheap model (GPT-4o-mini)."""
        ticker = ticker.upper()
        price = get_stock_price(ticker)

        if not price:
            return f"Could not fetch price data for {ticker}. Check the ticker symbol."

        prompt = QUICK_PRICE_PROMPT.format(
            ticker=ticker,
            current_price=price.current_price,
            change_percent=price.change_percent,
            fifty_two_week_low=price.fifty_two_week_low,
            fifty_two_week_high=price.fifty_two_week_high,
            volume=price.volume,
        )

        result = self.router.call(
            prompt=prompt,
            task_type="routine",
        )

        return result["content"]

    # ──────────────────────────────────────────────
    # Phase 2: Portfolio analysis
    # ──────────────────────────────────────────────

    def analyze_portfolio(self, user_id: int, explain: bool = False, validate: bool = True) -> AnalysisResult:
        """
        Analyze the user's entire portfolio.

        Loads all holdings, fetches live prices and fundamentals,
        calculates portfolio-level metrics, and sends to Claude Sonnet.
        """
        if not self.repo:
            return _error_result("portfolio", "Database not initialized")

        holdings = self.repo.get_holdings(user_id)
        if not holdings:
            return _error_result("portfolio", "No holdings found. Add positions first.")

        console.print("\n[bold blue]Analyzing portfolio...[/bold blue]\n")

        # Fetch live data for each holding
        console.print("  📊 Fetching live prices and fundamentals...")
        prices: dict[str, StockPrice | None] = {}
        market_data_lines: list[str] = []

        for h in holdings:
            price = get_stock_price(h.ticker)
            prices[h.ticker] = price
            fundamentals = get_fundamentals(h.ticker)
            market_data_lines.append(f"\n### {h.ticker}")
            market_data_lines.append(_format_price(price))
            if fundamentals:
                market_data_lines.append(
                    f"Sector: {fundamentals.sector} | P/E: "
                    f"{fundamentals.pe_ratio:.2f}" if fundamentals.pe_ratio else "N/A"
                )

        # Calculate portfolio metrics
        holdings_str = _format_holdings_for_prompt(holdings, prices)
        total_value, total_cost, total_pnl, total_pnl_pct = _calc_portfolio_metrics(
            holdings, prices
        )

        # Retrieve RAG context for portfolio
        rag_context = ""
        if self.rag_retriever:
            console.print("  📄 Retrieving filing and news context for holdings...")
            tickers = [h.ticker for h in holdings]
            retrieved = self.rag_retriever.retrieve_for_portfolio_analysis(tickers)
            raw_context = self.rag_retriever.format_portfolio_context(
                retrieved, max_words=2000
            )
            if raw_context:
                block = RAG_PORTFOLIO_CONTEXT_BLOCK.format(rag_content=raw_context)
                if explain:
                    block += RAG_EDUCATIONAL_NOTE
                rag_context = block

        edu_section = EDUCATIONAL_PORTFOLIO_SECTION if explain else ""
        prompt = PORTFOLIO_ANALYSIS_PROMPT.format(
            holdings_data=holdings_str,
            market_data="\n".join(market_data_lines),
            num_holdings=len(holdings),
            total_value=total_value,
            total_cost=total_cost,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            rag_context=rag_context,
            educational_section=edu_section,
        )

        console.print("  🤖 Running AI portfolio analysis (Claude Sonnet)...\n")
        result = self.router.call(
            prompt=prompt,
            system_prompt=PORTFOLIO_ANALYSIS_SYSTEM,
            task_type="analysis",
            max_tokens=8192 if explain else None,
        )

        cost = result["cost_record"].cost_usd if result["cost_record"] else 0
        analysis_result = AnalysisResult(
            ticker="PORTFOLIO",
            analysis_type="portfolio_educational" if explain else "portfolio",
            summary=_extract_summary(result["content"]),
            detailed_analysis=result["content"],
            data_sources=["yfinance:price", "yfinance:fundamentals", "database:holdings"],
            model_used=result["model"],
            cost_usd=cost,
        )

        if validate:
            provided_data = {
                "Holdings": holdings_str,
                "Market Data": "\n".join(market_data_lines),
                "Portfolio Metrics": (
                    f"Total Value: ${total_value:,.2f}, Cost: ${total_cost:,.2f}, "
                    f"P&L: ${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)"
                ),
            }
            analysis_result = _run_validation(
                self.validator, analysis_result, provided_data, rag_context
            )

        return analysis_result

    def analyze_holding(
        self, ticker: str, user_id: int, explain: bool = False, validate: bool = True
    ) -> AnalysisResult:
        """Analyze a specific holding in portfolio context."""
        if not self.repo:
            return _error_result(ticker, "Database not initialized")

        ticker = ticker.upper()
        holding = self.repo.get_holding(user_id, ticker)
        if not holding:
            return _error_result(ticker, f"No holding found for {ticker}")

        console.print(f"\n[bold blue]Analyzing {ticker} holding...[/bold blue]\n")

        # Get portfolio context
        all_holdings = self.repo.get_holdings(user_id)
        all_prices: dict[str, StockPrice | None] = {}
        for h in all_holdings:
            all_prices[h.ticker] = get_stock_price(h.ticker)

        total_value, _, _, _ = _calc_portfolio_metrics(all_holdings, all_prices)

        # Fetch detailed data for this holding
        console.print("  📊 Fetching data...")
        price = all_prices.get(ticker) or get_stock_price(ticker)
        fundamentals = get_fundamentals(ticker)
        history = get_price_history(ticker, period="3mo")
        analysts = get_analyst_summary(ticker)

        current_price = price.current_price if price else 0
        market_value = holding.shares * current_price
        pnl = market_value - holding.total_cost_basis
        pnl_pct = (pnl / holding.total_cost_basis * 100) if holding.total_cost_basis > 0 else 0
        position_pct = (market_value / total_value * 100) if total_value > 0 else 0

        # Retrieve RAG context for this holding
        rag_context = ""
        if self.rag_retriever:
            console.print("  📄 Retrieving SEC filings and news context...")
            retrieved = self.rag_retriever.retrieve_for_stock_analysis(ticker)
            raw_context = self.rag_retriever.format_context_for_prompt(
                retrieved, max_words=3000
            )
            if raw_context:
                block = RAG_CONTEXT_BLOCK.format(rag_content=raw_context)
                if explain:
                    block += RAG_EDUCATIONAL_NOTE
                rag_context = block

        edu_section = EDUCATIONAL_PORTFOLIO_SECTION if explain else ""
        prompt = HOLDING_ANALYSIS_PROMPT.format(
            ticker=ticker,
            shares=holding.shares,
            avg_cost=holding.avg_cost_basis,
            total_cost=holding.total_cost_basis,
            current_price=current_price,
            market_value=market_value,
            pnl=pnl,
            pnl_pct=pnl_pct,
            fundamental_data=_format_fundamentals(fundamentals),
            history_data=_format_history(history),
            analyst_data=_format_analysts(analysts),
            position_pct=position_pct,
            portfolio_value=total_value,
            rag_context=rag_context,
            educational_section=edu_section,
        )

        console.print("  🤖 Running AI holding analysis (Claude Sonnet)...\n")
        result = self.router.call(
            prompt=prompt,
            system_prompt=PORTFOLIO_ANALYSIS_SYSTEM,
            task_type="analysis",
            max_tokens=8192 if explain else None,
        )

        price_str = _format_price(price)
        fundamentals_str = _format_fundamentals(fundamentals)
        history_str = _format_history(history)
        analyst_str = _format_analysts(analysts)

        cost = result["cost_record"].cost_usd if result["cost_record"] else 0
        analysis_result = AnalysisResult(
            ticker=ticker,
            analysis_type="holding_educational" if explain else "holding",
            summary=_extract_summary(result["content"]),
            detailed_analysis=result["content"],
            data_sources=_list_data_sources(price, fundamentals, history, analysts)
            + ["database:holdings"],
            model_used=result["model"],
            cost_usd=cost,
        )

        if validate:
            provided_data = {
                "Price Data": price_str,
                "Fundamentals": fundamentals_str,
                "Price History": history_str,
                "Analyst Consensus": analyst_str,
                "Holding Info": (
                    f"{holding.shares} shares @ ${holding.avg_cost_basis:.2f} avg cost, "
                    f"Market Value: ${market_value:,.2f}, P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%), "
                    f"Position: {position_pct:.1f}% of portfolio"
                ),
            }
            analysis_result = _run_validation(
                self.validator, analysis_result, provided_data, rag_context
            )

        # Cache the result
        self.repo.save_analysis(StockAnalysisCache(
            ticker=ticker,
            analysis_type=analysis_result.analysis_type,
            summary=analysis_result.summary,
            detailed_analysis=analysis_result.detailed_analysis,
            model_used=analysis_result.model_used,
            cost_usd=analysis_result.cost_usd,
        ))

        return analysis_result

    def evaluate_potential_buy(
        self, ticker: str, user_id: int, explain: bool = False, validate: bool = True
    ) -> AnalysisResult:
        """Evaluate a stock as a potential buy given portfolio context."""
        ticker = ticker.upper()
        console.print(f"\n[bold blue]Evaluating {ticker} as potential buy...[/bold blue]\n")

        # Fetch stock data
        console.print("  📊 Fetching stock data...")
        price = get_stock_price(ticker)
        fundamentals = get_fundamentals(ticker)
        history = get_price_history(ticker, period="3mo")
        analysts = get_analyst_summary(ticker)

        # Get portfolio context
        portfolio_summary = "No portfolio data available."
        portfolio_tickers: list[str] = []
        if self.repo:
            holdings = self.repo.get_holdings(user_id)
            if holdings:
                prices: dict[str, StockPrice | None] = {}
                for h in holdings:
                    prices[h.ticker] = get_stock_price(h.ticker)
                portfolio_summary = _format_portfolio_summary(holdings, prices)
                portfolio_tickers = [h.ticker for h in holdings]

        # Retrieve RAG context for potential buy evaluation
        rag_context = ""
        if self.rag_retriever:
            console.print("  📄 Retrieving SEC filings, news, and sector context...")
            retrieved = self.rag_retriever.retrieve_for_potential_buy(
                ticker, portfolio_tickers
            )
            raw_context = self.rag_retriever.format_potential_buy_context(
                retrieved, max_words=3500
            )
            if raw_context:
                block = RAG_POTENTIAL_BUY_CONTEXT_BLOCK.format(rag_content=raw_context)
                if explain:
                    block += RAG_EDUCATIONAL_NOTE
                rag_context = block

        edu_section = EDUCATIONAL_PORTFOLIO_SECTION if explain else ""
        prompt = POTENTIAL_BUY_PROMPT.format(
            ticker=ticker,
            price_data=_format_price(price),
            fundamental_data=_format_fundamentals(fundamentals),
            history_data=_format_history(history),
            analyst_data=_format_analysts(analysts),
            portfolio_summary=portfolio_summary,
            rag_context=rag_context,
            educational_section=edu_section,
        )

        console.print("  🤖 Running AI evaluation (Claude Sonnet)...\n")
        result = self.router.call(
            prompt=prompt,
            system_prompt=PORTFOLIO_ANALYSIS_SYSTEM,
            task_type="analysis",
            max_tokens=8192 if explain else None,
        )

        price_str = _format_price(price)
        fundamentals_str = _format_fundamentals(fundamentals)
        history_str = _format_history(history)
        analyst_str = _format_analysts(analysts)

        cost = result["cost_record"].cost_usd if result["cost_record"] else 0
        analysis_result = AnalysisResult(
            ticker=ticker,
            analysis_type="potential_buy_educational" if explain else "potential_buy",
            summary=_extract_summary(result["content"]),
            detailed_analysis=result["content"],
            data_sources=_list_data_sources(price, fundamentals, history, analysts)
            + ["database:portfolio"],
            model_used=result["model"],
            cost_usd=cost,
        )

        if validate:
            provided_data = {
                "Price Data": price_str,
                "Fundamentals": fundamentals_str,
                "Price History": history_str,
                "Analyst Consensus": analyst_str,
                "Portfolio Context": portfolio_summary,
            }
            analysis_result = _run_validation(
                self.validator, analysis_result, provided_data, rag_context
            )

        # Cache
        if self.repo:
            self.repo.save_analysis(StockAnalysisCache(
                ticker=ticker,
                analysis_type=analysis_result.analysis_type,
                summary=analysis_result.summary,
                detailed_analysis=analysis_result.detailed_analysis,
                model_used=analysis_result.model_used,
                cost_usd=analysis_result.cost_usd,
            ))

        return analysis_result


# ──────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────


def _format_price(price) -> str:
    """Format StockPrice for prompt inclusion."""
    if not price:
        return "Price data unavailable."
    return (
        f"Current Price: ${price.current_price:.2f}\n"
        f"Previous Close: ${price.previous_close:.2f}\n"
        f"Change: {price.change_percent:+.2f}%\n"
        f"Day Range: ${price.day_low:.2f} - ${price.day_high:.2f}\n"
        f"52-Week Range: ${price.fifty_two_week_low:.2f} - ${price.fifty_two_week_high:.2f}\n"
        f"Volume: {price.volume:,}"
    )


def _format_fundamentals(fundamentals) -> str:
    """Format StockFundamentals for prompt inclusion."""
    if not fundamentals:
        return "Fundamental data unavailable."

    lines = [
        f"Company: {fundamentals.company_name}",
        f"Sector: {fundamentals.sector} | Industry: {fundamentals.industry}",
        f"Market Cap: ${fundamentals.market_cap:,.0f}" if fundamentals.market_cap else "Market Cap: N/A",
        f"P/E Ratio (TTM): {fundamentals.pe_ratio:.2f}" if fundamentals.pe_ratio else "P/E Ratio: N/A",
        f"Forward P/E: {fundamentals.forward_pe:.2f}" if fundamentals.forward_pe else "Forward P/E: N/A",
        f"PEG Ratio: {fundamentals.peg_ratio:.2f}" if fundamentals.peg_ratio else "PEG Ratio: N/A",
        f"EPS (TTM): ${fundamentals.eps_ttm:.2f}" if fundamentals.eps_ttm else "EPS: N/A",
        f"Profit Margin: {fundamentals.profit_margin:.1%}" if fundamentals.profit_margin else "Profit Margin: N/A",
        f"ROE: {fundamentals.return_on_equity:.1%}" if fundamentals.return_on_equity else "ROE: N/A",
        f"Debt/Equity: {fundamentals.debt_to_equity:.1f}" if fundamentals.debt_to_equity else "Debt/Equity: N/A",
        f"Beta: {fundamentals.beta:.2f}" if fundamentals.beta else "Beta: N/A",
        f"Dividend Yield: {fundamentals.dividend_yield:.2%}" if fundamentals.dividend_yield else "Dividend Yield: N/A",
    ]
    return "\n".join(lines)


def _format_history(history) -> str:
    """Format price history dict for prompt inclusion."""
    if not history:
        return "Price history unavailable."
    return (
        f"Period: {history['start_date']} to {history['end_date']}\n"
        f"Start Price: ${history['start_price']}\n"
        f"End Price: ${history['end_price']}\n"
        f"Period Change: {history['price_change_percent']:+.2f}%\n"
        f"Period High: ${history['high']}\n"
        f"Period Low: ${history['low']}\n"
        f"Avg Daily Volume: {history['avg_volume']:,}"
    )


def _format_analysts(analysts) -> str:
    """Format analyst summary for prompt inclusion."""
    if not analysts:
        return "Analyst data unavailable."
    lines = [
        f"Consensus: {analysts['recommendation']}",
        f"Number of Analysts: {analysts['number_of_analysts']}",
        f"Mean Target: ${analysts['target_mean_price']}" if analysts['target_mean_price'] else "Mean Target: N/A",
        f"Target Range: ${analysts['target_low_price']} - ${analysts['target_high_price']}"
        if analysts['target_low_price'] and analysts['target_high_price']
        else "Target Range: N/A",
        f"Upside to Mean Target: {analysts['upside_percent']:+.1f}%"
        if analysts['upside_percent']
        else "Upside: N/A",
    ]
    return "\n".join(lines)


def _format_holdings_for_prompt(
    holdings: list[Holding], prices: dict[str, StockPrice | None]
) -> str:
    """Format all holdings with live prices into a prompt-ready string."""
    lines = []
    for h in holdings:
        price = prices.get(h.ticker)
        current = price.current_price if price else 0
        market_val = h.shares * current
        pnl = market_val - h.total_cost_basis
        pnl_pct = (pnl / h.total_cost_basis * 100) if h.total_cost_basis > 0 else 0
        lines.append(
            f"- {h.ticker}: {h.shares:.4f} shares @ ${h.avg_cost_basis:.2f} avg cost | "
            f"Current: ${current:.2f} | Value: ${market_val:,.2f} | "
            f"P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)"
        )
    return "\n".join(lines)


def _format_portfolio_summary(
    holdings: list[Holding], prices: dict[str, StockPrice | None]
) -> str:
    """Format a brief portfolio summary for the potential buy prompt."""
    total_value, total_cost, total_pnl, total_pnl_pct = _calc_portfolio_metrics(
        holdings, prices
    )
    lines = [
        f"Total Portfolio Value: ${total_value:,.2f}",
        f"Total Cost Basis: ${total_cost:,.2f}",
        f"Overall P&L: ${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)",
        f"Number of Holdings: {len(holdings)}",
        "",
        "Current Holdings:",
    ]
    for h in holdings:
        price = prices.get(h.ticker)
        current = price.current_price if price else 0
        market_val = h.shares * current
        pct_of_portfolio = (market_val / total_value * 100) if total_value > 0 else 0
        lines.append(f"  - {h.ticker}: ${market_val:,.2f} ({pct_of_portfolio:.1f}% of portfolio)")
    return "\n".join(lines)


def _calc_portfolio_metrics(
    holdings: list[Holding], prices: dict[str, StockPrice | None]
) -> tuple[float, float, float, float]:
    """Calculate total value, cost, P&L, and P&L % for a portfolio."""
    total_value = 0.0
    total_cost = 0.0
    for h in holdings:
        price = prices.get(h.ticker)
        current = price.current_price if price else 0
        total_value += h.shares * current
        total_cost += h.total_cost_basis
    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
    return total_value, total_cost, total_pnl, total_pnl_pct


def _extract_summary(analysis_text: str) -> str:
    """
    Extract a 1-2 sentence summary from the full analysis.

    Looks for a "Bottom Line" section. If not found, takes the first 200 chars.
    """
    for marker in ["**Bottom Line**", "Bottom Line:", "## Bottom Line"]:
        if marker in analysis_text:
            after_marker = analysis_text.split(marker, 1)[1].strip()
            sentences = after_marker.split(".")
            summary = ".".join(sentences[:3]).strip()
            if summary and not summary.endswith("."):
                summary += "."
            return summary

    return analysis_text[:200].strip() + "..."


def _list_data_sources(price, fundamentals, history, analysts) -> list[str]:
    """List which data sources were successfully used."""
    sources = []
    if price:
        sources.append("yfinance:price")
    if fundamentals:
        sources.append("yfinance:fundamentals")
    if history:
        sources.append("yfinance:history")
    if analysts:
        sources.append("yfinance:analysts")
    return sources


def _run_validation(
    validator,
    analysis_result: AnalysisResult,
    provided_data: dict,
    rag_context: str = "",
) -> AnalysisResult:
    """
    Run validation on an analysis result and attach the ValidationResult.

    Returns the analysis_result with validation and total_cost_usd updated.
    """
    console.print("  🔍 Validating analysis for accuracy...")
    validation = validator.validate(
        analysis_text=analysis_result.detailed_analysis,
        provided_data=provided_data,
        rag_context=rag_context,
    )
    analysis_result.validation = validation
    analysis_result.total_cost_usd = analysis_result.cost_usd + validation.validation_cost_usd

    if validation.was_validated:
        total_issues = (
            len(validation.unsupported_numbers)
            + len(validation.unsupported_claims)
            + len(validation.contradictions)
        )
        if validation.confidence_score >= 90:
            console.print(
                f"  [green]Validation passed — confidence: {validation.confidence_score}/100"
                f" ({total_issues} issues)[/green]"
            )
        elif validation.confidence_score >= 70:
            console.print(
                f"  [yellow]Validation warning — confidence: {validation.confidence_score}/100"
                f" ({total_issues} issues)[/yellow]"
            )
        else:
            console.print(
                f"  [red]Validation concern — confidence: {validation.confidence_score}/100"
                f" ({total_issues} issues)[/red]"
            )

    return analysis_result


def _error_result(ticker: str, message: str) -> AnalysisResult:
    """Return an error AnalysisResult."""
    return AnalysisResult(
        ticker=ticker,
        analysis_type="error",
        summary=message,
        detailed_analysis=message,
        data_sources=[],
        model_used="none",
        cost_usd=0,
    )
