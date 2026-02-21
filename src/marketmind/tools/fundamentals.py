"""
Stock Fundamentals Tool — Fetches fundamental analysis data.

This is the data Jeet actually uses for investment decisions:
- P/E ratio (current and forward)
- Earnings per share
- Analyst ratings and price targets
- Financial health metrics

All data from yfinance (free). We'll add Alpha Vantage and Finnhub
as supplementary sources in later phases.
"""

import yfinance as yf
from rich.console import Console

from marketmind.models.schemas import StockFundamentals

console = Console()


def get_fundamentals(ticker: str) -> StockFundamentals | None:
    """
    Fetch fundamental analysis data for a stock.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")

    Returns:
        StockFundamentals model, or None if ticker not found.
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info

        if not info or not info.get("shortName"):
            console.print(f"[yellow]Warning: No fundamental data found for {ticker}[/yellow]")
            return None

        return StockFundamentals(
            ticker=ticker.upper(),
            company_name=info.get("shortName", "Unknown"),
            sector=info.get("sector", "Unknown"),
            industry=info.get("industry", "Unknown"),
            market_cap=info.get("marketCap"),
            pe_ratio=info.get("trailingPE"),
            forward_pe=info.get("forwardPE"),
            peg_ratio=info.get("pegRatio"),
            price_to_book=info.get("priceToBook"),
            eps_ttm=info.get("trailingEps"),
            dividend_yield=info.get("dividendYield"),
            revenue_ttm=info.get("totalRevenue"),
            profit_margin=info.get("profitMargins"),
            return_on_equity=info.get("returnOnEquity"),
            debt_to_equity=info.get("debtToEquity"),
            beta=info.get("beta"),
            analyst_target_price=info.get("targetMeanPrice"),
            recommendation=info.get("recommendationKey"),
        )

    except Exception as e:
        console.print(f"[red]Error fetching fundamentals for {ticker}: {e}[/red]")
        return None


def get_analyst_summary(ticker: str) -> dict | None:
    """
    Fetch analyst recommendations summary.

    Returns counts of buy/hold/sell ratings and price target range.
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info

        if not info:
            return None

        return {
            "ticker": ticker.upper(),
            "recommendation": info.get("recommendationKey", "N/A"),
            "number_of_analysts": info.get("numberOfAnalystOpinions", 0),
            "target_mean_price": info.get("targetMeanPrice"),
            "target_high_price": info.get("targetHighPrice"),
            "target_low_price": info.get("targetLowPrice"),
            "target_median_price": info.get("targetMedianPrice"),
            "current_price": info.get("regularMarketPrice") or info.get("currentPrice"),
            "upside_percent": _calc_upside(
                info.get("regularMarketPrice") or info.get("currentPrice"),
                info.get("targetMeanPrice"),
            ),
        }

    except Exception as e:
        console.print(f"[red]Error fetching analyst data for {ticker}: {e}[/red]")
        return None


def _calc_upside(current: float | None, target: float | None) -> float | None:
    """Calculate upside/downside percentage to analyst target."""
    if not current or not target or current == 0:
        return None
    return round(((target - current) / current) * 100, 2)
