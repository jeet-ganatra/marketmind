"""
Basic tests for Phase 1.

These tests verify that the tools work and return the right data structures.
They DON'T call the LLM APIs (to avoid costs in CI).
"""

from marketmind.models.schemas import StockFundamentals, StockPrice
from marketmind.tools.fundamentals import get_fundamentals
from marketmind.tools.price import get_stock_price


def test_get_stock_price_valid_ticker():
    """Test that we get price data for a known stock."""
    price = get_stock_price("AAPL")
    assert price is not None
    assert isinstance(price, StockPrice)
    assert price.ticker == "AAPL"
    assert price.current_price > 0


def test_get_stock_price_invalid_ticker():
    """Test graceful handling of invalid ticker."""
    price = get_stock_price("XYZXYZXYZ")
    assert price is None


def test_get_fundamentals_valid_ticker():
    """Test that we get fundamental data for a known stock."""
    fundamentals = get_fundamentals("MSFT")
    assert fundamentals is not None
    assert isinstance(fundamentals, StockFundamentals)
    assert fundamentals.ticker == "MSFT"
    assert fundamentals.company_name != "Unknown"


def test_get_fundamentals_invalid_ticker():
    """Test graceful handling of invalid ticker."""
    fundamentals = get_fundamentals("XYZXYZXYZ")
    assert fundamentals is None


def test_cost_tracker_calculation():
    """Test that cost calculation is correct."""
    from marketmind.llm.cost_tracker import CostTracker

    tracker = CostTracker(budget_limit=30.0, data_dir="/tmp/marketmind_test")
    cost = tracker.calculate_cost("gpt-4o-mini", input_tokens=1000, output_tokens=500)

    # gpt-4o-mini: $0.15/M input, $0.60/M output
    # 1000 input tokens = 0.00015
    # 500 output tokens = 0.0003
    expected = 0.00015 + 0.0003
    assert abs(cost - expected) < 0.0001


def test_cost_tracker_unknown_model():
    """Test that unknown models return 0 cost (not crash)."""
    from marketmind.llm.cost_tracker import CostTracker

    tracker = CostTracker(budget_limit=30.0, data_dir="/tmp/marketmind_test")
    cost = tracker.calculate_cost("some-future-model", input_tokens=1000, output_tokens=500)
    assert cost == 0.0
