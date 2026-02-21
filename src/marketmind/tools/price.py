"""
Stock Price Tool — Fetches current and historical price data.

Uses yfinance (free, no API key needed) as the primary data source.
yfinance pulls data from Yahoo Finance.

Limitations to be aware of:
- yfinance is unofficial and could break if Yahoo changes their API
- Free tier has rate limits (not documented, but ~2000 requests/hour is safe)
- Real-time data has ~15 min delay (fine for our batch analysis phase)
"""

import yfinance as yf
from rich.console import Console

from marketmind.models.schemas import StockPrice

console = Console()


def get_stock_price(ticker: str) -> StockPrice | None:
    """
    Fetch current price data for a stock.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "MSFT")

    Returns:
        StockPrice model with current price data, or None if ticker not found.
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info

        # yfinance returns an empty dict or minimal data for invalid tickers
        if not info or "regularMarketPrice" not in info:
            console.print(f"[yellow]Warning: No price data found for {ticker}[/yellow]")
            return None

        current_price = info.get("regularMarketPrice") or info.get("currentPrice", 0)
        previous_close = info.get("regularMarketPreviousClose", 0)

        # Calculate change percent safely
        change_percent = 0.0
        if previous_close and previous_close > 0:
            change_percent = round(((current_price - previous_close) / previous_close) * 100, 2)

        return StockPrice(
            ticker=ticker.upper(),
            current_price=current_price,
            previous_close=previous_close,
            change_percent=change_percent,
            day_high=info.get("dayHigh", 0),
            day_low=info.get("dayLow", 0),
            volume=info.get("volume", 0),
            fifty_two_week_high=info.get("fiftyTwoWeekHigh", 0),
            fifty_two_week_low=info.get("fiftyTwoWeekLow", 0),
        )

    except Exception as e:
        console.print(f"[red]Error fetching price for {ticker}: {e}[/red]")
        return None


def get_price_history(ticker: str, period: str = "3mo") -> dict | None:
    """
    Fetch historical price data.

    Args:
        ticker: Stock ticker symbol
        period: Time period — "1mo", "3mo", "6mo", "1y", "2y", "5y"

    Returns:
        Dict with dates and closing prices, or None on error.
    """
    try:
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period=period)

        if hist.empty:
            return None

        return {
            "ticker": ticker.upper(),
            "period": period,
            "data_points": len(hist),
            "start_date": str(hist.index[0].date()),
            "end_date": str(hist.index[-1].date()),
            "start_price": round(hist["Close"].iloc[0], 2),
            "end_price": round(hist["Close"].iloc[-1], 2),
            "high": round(hist["High"].max(), 2),
            "low": round(hist["Low"].min(), 2),
            "avg_volume": int(hist["Volume"].mean()),
            "price_change_percent": round(
                ((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0]) * 100,
                2,
            ),
        }

    except Exception as e:
        console.print(f"[red]Error fetching history for {ticker}: {e}[/red]")
        return None
