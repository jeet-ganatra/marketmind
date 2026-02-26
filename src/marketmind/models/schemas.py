"""
Data models (schemas) for MarketMind.

Using Pydantic models gives us:
- Type safety: catch data issues early
- Serialization: easy to convert to/from JSON for LLM prompts
- Documentation: the schema IS the documentation
"""

from datetime import datetime

from pydantic import BaseModel, Field


class StockPrice(BaseModel):
    """Current or historical price data for a stock."""

    ticker: str
    current_price: float
    previous_close: float
    change_percent: float
    day_high: float
    day_low: float
    volume: int
    fifty_two_week_high: float
    fifty_two_week_low: float
    timestamp: datetime = Field(default_factory=datetime.now)


class StockFundamentals(BaseModel):
    """Fundamental analysis data for a stock."""

    ticker: str
    company_name: str
    sector: str
    industry: str
    market_cap: float | None = None
    pe_ratio: float | None = None
    forward_pe: float | None = None
    peg_ratio: float | None = None
    price_to_book: float | None = None
    eps_ttm: float | None = None
    dividend_yield: float | None = None
    revenue_ttm: float | None = None
    profit_margin: float | None = None
    return_on_equity: float | None = None
    debt_to_equity: float | None = None
    beta: float | None = None
    analyst_target_price: float | None = None
    recommendation: str | None = None  # e.g., "buy", "hold", "sell"
    timestamp: datetime = Field(default_factory=datetime.now)


class AnalysisResult(BaseModel):
    """Result of an AI-powered stock analysis."""

    ticker: str
    analysis_type: str  # e.g., "comprehensive", "quick", "comparison"
    summary: str  # Short 1-2 sentence summary
    detailed_analysis: str  # Full analysis text
    data_sources: list[str]  # Which tools/APIs were used
    model_used: str  # Which LLM generated this
    cost_usd: float  # How much this analysis cost in API fees
    timestamp: datetime = Field(default_factory=datetime.now)


class CostRecord(BaseModel):
    """Tracks the cost of a single LLM API call."""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    query_type: str  # "routine" or "analysis"
    timestamp: datetime = Field(default_factory=datetime.now)


# ──────────────────────────────────────────────────────────
# Phase 2: Portfolio Models
# ──────────────────────────────────────────────────────────


class User(BaseModel):
    """A MarketMind user (supports multi-user portfolios)."""

    id: int | None = None
    username: str
    created_at: datetime = Field(default_factory=datetime.now)


class Trade(BaseModel):
    """A single buy or sell transaction."""

    id: int | None = None
    user_id: int
    ticker: str
    trade_type: str  # "buy" or "sell"
    shares: float
    price_per_share: float
    total_amount: float
    trade_date: datetime
    source: str = "manual"  # "manual", "robinhood_csv", etc.
    created_at: datetime = Field(default_factory=datetime.now)


class Holding(BaseModel):
    """Current position in a stock (derived from trades via FIFO)."""

    id: int | None = None
    user_id: int
    ticker: str
    shares: float
    avg_cost_basis: float
    total_cost_basis: float
    first_purchase_date: datetime | None = None
    last_trade_date: datetime | None = None
    updated_at: datetime = Field(default_factory=datetime.now)


class WatchlistItem(BaseModel):
    """A ticker on the user's watchlist."""

    id: int | None = None
    user_id: int
    ticker: str
    notes: str = ""
    added_at: datetime = Field(default_factory=datetime.now)


class StockAnalysisCache(BaseModel):
    """Cached AI analysis result (shared across users)."""

    id: int | None = None
    ticker: str
    analysis_type: str  # "comprehensive", "educational", "portfolio", "holding", "potential_buy"
    user_id: int | None = None
    summary: str
    detailed_analysis: str
    model_used: str
    cost_usd: float
    created_at: datetime = Field(default_factory=datetime.now)
