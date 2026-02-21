"""
Agent Orchestrator — The brain of MarketMind.

This is the central module that:
1. Receives a user query (e.g., "analyze AAPL")
2. Fetches relevant data using tools
3. Assembles context for the LLM
4. Routes to the appropriate model
5. Returns structured results

In Phase 1, this is a simple procedural flow.
In Phase 3+, this becomes a LangGraph agent that autonomously
decides which tools to call.
"""

from rich.console import Console

from marketmind.config import Settings
from marketmind.llm.router import LLMRouter
from marketmind.models.schemas import AnalysisResult
from marketmind.prompts.templates import (
    COMPREHENSIVE_ANALYSIS_PROMPT,
    EDUCATIONAL_ANALYSIS_PROMPT,
    QUICK_PRICE_PROMPT,
    STOCK_ANALYSIS_SYSTEM,
)
from marketmind.tools.fundamentals import get_analyst_summary, get_fundamentals
from marketmind.tools.price import get_price_history, get_stock_price

console = Console()


class MarketMindAgent:
    """
    Main agent that orchestrates stock analysis.

    This is intentionally simple in Phase 1 — a linear pipeline:
    fetch data → format prompt → call LLM → return result.

    Later phases will replace this with LangGraph for dynamic tool selection.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.router = LLMRouter(settings)

    def analyze(self, ticker: str, explain: bool = False) -> AnalysisResult:
        """
        Run a comprehensive analysis on a stock.

        This is the main entry point. It:
        1. Fetches price, fundamentals, history, and analyst data
        2. Formats everything into a prompt
        3. Sends to Claude Sonnet for deep analysis
        4. Returns a structured result

        Args:
            ticker: Stock ticker symbol
            explain: If True, uses educational prompt with beginner-friendly explanations
        """
        ticker = ticker.upper()
        mode_label = " (educational mode)" if explain else ""
        console.print(f"\n[bold blue]Analyzing {ticker}{mode_label}...[/bold blue]\n")

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
        # We convert to readable strings. The LLM needs human-readable context,
        # not raw JSON (though it can handle JSON — readable is better for analysis).
        price_str = _format_price(price)
        fundamentals_str = _format_fundamentals(fundamentals)
        history_str = _format_history(history)
        analyst_str = _format_analysts(analysts)

        template = EDUCATIONAL_ANALYSIS_PROMPT if explain else COMPREHENSIVE_ANALYSIS_PROMPT
        prompt = template.format(
            ticker=ticker,
            price_data=price_str,
            fundamental_data=fundamentals_str,
            history_period="3 months",
            history_data=history_str,
            analyst_data=analyst_str,
        )

        # Step 3: Call LLM (analysis task → Claude Sonnet)
        # Educational mode needs more tokens for the longer explanations
        console.print("  🤖 Running AI analysis (Claude Sonnet)...\n")
        result = self.router.call(
            prompt=prompt,
            system_prompt=STOCK_ANALYSIS_SYSTEM,
            task_type="analysis",
            max_tokens=8192 if explain else None,
        )

        # Step 4: Build structured result
        cost = result["cost_record"].cost_usd if result["cost_record"] else 0

        return AnalysisResult(
            ticker=ticker,
            analysis_type="educational" if explain else "comprehensive",
            summary=_extract_summary(result["content"]),
            detailed_analysis=result["content"],
            data_sources=_list_data_sources(price, fundamentals, history, analysts),
            model_used=result["model"],
            cost_usd=cost,
        )

    def quick_price(self, ticker: str) -> str:
        """
        Quick price check — uses cheap model (GPT-4o-mini).

        This is for "What's AAPL at?" type questions where you don't
        need deep analysis, just a quick status.
        """
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
            task_type="routine",  # → GPT-4o-mini (cheap)
        )

        return result["content"]


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


def _extract_summary(analysis_text: str) -> str:
    """
    Extract a 1-2 sentence summary from the full analysis.

    Looks for a "Bottom Line" section. If not found, takes the first 2 sentences.
    """
    # Try to find a "Bottom Line" section
    for marker in ["**Bottom Line**", "Bottom Line:", "## Bottom Line"]:
        if marker in analysis_text:
            after_marker = analysis_text.split(marker, 1)[1].strip()
            # Take first 2-3 sentences
            sentences = after_marker.split(".")
            summary = ".".join(sentences[:3]).strip()
            if summary and not summary.endswith("."):
                summary += "."
            return summary

    # Fallback: first 200 chars
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
