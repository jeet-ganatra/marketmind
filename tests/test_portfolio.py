"""
Phase 2 tests — Portfolio tracking, FIFO cost basis, CSV import, caching.

All tests are offline (no API keys or network access required).
Uses temporary directories and in-memory-like SQLite for full isolation.
"""

import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from marketmind.db.database import Database
from marketmind.db.repository import PortfolioRepository
from marketmind.models.schemas import (
    Holding,
    StockAnalysisCache,
    Trade,
    User,
    WatchlistItem,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory for each test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def db(tmp_dir):
    """Provide an initialized Database instance."""
    database = Database(data_dir=tmp_dir)
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def repo(db):
    """Provide a PortfolioRepository wired to the test database."""
    return PortfolioRepository(db)


@pytest.fixture
def user(repo):
    """Provide a default test user."""
    return repo.create_user("testuser")


# ──────────────────────────────────────────────
# Database creation
# ──────────────────────────────────────────────


class TestDatabaseCreation:
    def test_db_file_created(self, tmp_dir):
        db = Database(data_dir=tmp_dir)
        db.initialize()
        assert (tmp_dir / "marketmind.db").exists()
        db.close()

    def test_tables_exist(self, db):
        cursor = db.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row["name"] for row in cursor.fetchall()]
        assert "users" in tables
        assert "trades" in tables
        assert "holdings" in tables
        assert "stock_analyses" in tables
        assert "watchlist" in tables

    def test_idempotent_init(self, tmp_dir):
        db = Database(data_dir=tmp_dir)
        db.initialize()
        db.initialize()  # Should not raise
        db.close()

    def test_context_manager(self, tmp_dir):
        with Database(data_dir=tmp_dir) as db:
            cursor = db.connection.execute("SELECT 1")
            assert cursor.fetchone()[0] == 1


# ──────────────────────────────────────────────
# User management
# ──────────────────────────────────────────────


class TestUserManagement:
    def test_create_user(self, repo):
        user = repo.create_user("alice")
        assert user.id is not None
        assert user.username == "alice"

    def test_get_user(self, repo):
        repo.create_user("bob")
        found = repo.get_user("bob")
        assert found is not None
        assert found.username == "bob"

    def test_get_user_not_found(self, repo):
        assert repo.get_user("nonexistent") is None

    def test_get_or_create_new(self, repo):
        user = repo.get_or_create_user("charlie")
        assert user.id is not None
        assert user.username == "charlie"

    def test_get_or_create_existing(self, repo):
        repo.create_user("diana")
        user = repo.get_or_create_user("diana")
        assert user.username == "diana"

    def test_duplicate_username_raises(self, repo):
        repo.create_user("eve")
        with pytest.raises(sqlite3.IntegrityError):
            repo.create_user("eve")


# ──────────────────────────────────────────────
# Trades and FIFO
# ──────────────────────────────────────────────


class TestTradesAndFIFO:
    def _make_trade(self, user_id, ticker, trade_type, shares, price, date_str="2025-01-15"):
        return Trade(
            user_id=user_id,
            ticker=ticker,
            trade_type=trade_type,
            shares=shares,
            price_per_share=price,
            total_amount=shares * price,
            trade_date=datetime.strptime(date_str, "%Y-%m-%d"),
        )

    def test_simple_buy(self, repo, user):
        trade = self._make_trade(user.id, "AAPL", "buy", 10, 198.50)
        repo.add_trade(trade)
        repo.recalculate_holdings(user.id)

        holdings = repo.get_holdings(user.id)
        assert len(holdings) == 1
        assert holdings[0].ticker == "AAPL"
        assert holdings[0].shares == pytest.approx(10, abs=0.001)
        assert holdings[0].avg_cost_basis == pytest.approx(198.50, abs=0.01)

    def test_two_buys_avg_cost(self, repo, user):
        repo.add_trade(self._make_trade(user.id, "AAPL", "buy", 10, 100.00, "2025-01-01"))
        repo.add_trade(self._make_trade(user.id, "AAPL", "buy", 10, 200.00, "2025-01-15"))
        repo.recalculate_holdings(user.id)

        holding = repo.get_holding(user.id, "AAPL")
        assert holding is not None
        assert holding.shares == pytest.approx(20, abs=0.001)
        # FIFO avg: (10*100 + 10*200) / 20 = 150
        assert holding.avg_cost_basis == pytest.approx(150.00, abs=0.01)

    def test_partial_sell_fifo(self, repo, user):
        """Sell 5 shares from first lot of 10@100. Remaining: 5@100 + 10@200."""
        repo.add_trade(self._make_trade(user.id, "AAPL", "buy", 10, 100.00, "2025-01-01"))
        repo.add_trade(self._make_trade(user.id, "AAPL", "buy", 10, 200.00, "2025-01-15"))
        repo.add_trade(self._make_trade(user.id, "AAPL", "sell", 5, 150.00, "2025-02-01"))
        repo.recalculate_holdings(user.id)

        holding = repo.get_holding(user.id, "AAPL")
        assert holding is not None
        assert holding.shares == pytest.approx(15, abs=0.001)
        # FIFO: sold 5 from first lot (5@100 remaining), second lot untouched (10@200)
        # Total cost: 5*100 + 10*200 = 2500. Avg: 2500/15 ≈ 166.67
        assert holding.total_cost_basis == pytest.approx(2500.00, abs=0.01)
        assert holding.avg_cost_basis == pytest.approx(166.6667, abs=0.01)

    def test_sell_across_lots(self, repo, user):
        """Sell 12 shares: consumes all of lot1 (10@100) + 2 from lot2 (10@200)."""
        repo.add_trade(self._make_trade(user.id, "AAPL", "buy", 10, 100.00, "2025-01-01"))
        repo.add_trade(self._make_trade(user.id, "AAPL", "buy", 10, 200.00, "2025-01-15"))
        repo.add_trade(self._make_trade(user.id, "AAPL", "sell", 12, 180.00, "2025-02-01"))
        repo.recalculate_holdings(user.id)

        holding = repo.get_holding(user.id, "AAPL")
        assert holding is not None
        assert holding.shares == pytest.approx(8, abs=0.001)
        # FIFO: 10@100 fully consumed, 2@200 consumed. Remaining: 8@200
        assert holding.total_cost_basis == pytest.approx(1600.00, abs=0.01)
        assert holding.avg_cost_basis == pytest.approx(200.00, abs=0.01)

    def test_full_sell_removes_holding(self, repo, user):
        """Selling all shares should remove the holding."""
        repo.add_trade(self._make_trade(user.id, "AAPL", "buy", 10, 100.00, "2025-01-01"))
        repo.add_trade(self._make_trade(user.id, "AAPL", "sell", 10, 150.00, "2025-02-01"))
        repo.recalculate_holdings(user.id)

        holdings = repo.get_holdings(user.id)
        assert len(holdings) == 0
        assert repo.get_holding(user.id, "AAPL") is None

    def test_multiple_tickers(self, repo, user):
        repo.add_trade(self._make_trade(user.id, "AAPL", "buy", 10, 100.00))
        repo.add_trade(self._make_trade(user.id, "MSFT", "buy", 5, 400.00))
        repo.recalculate_holdings(user.id)

        holdings = repo.get_holdings(user.id)
        assert len(holdings) == 2
        tickers = {h.ticker for h in holdings}
        assert tickers == {"AAPL", "MSFT"}

    def test_duplicate_detection(self, repo, user):
        trade = self._make_trade(user.id, "AAPL", "buy", 10, 198.50)
        repo.add_trade(trade)

        assert repo.trade_exists(
            user.id, "AAPL", "buy", 10, 198.50,
            datetime.strptime("2025-01-15", "%Y-%m-%d"),
        )
        assert not repo.trade_exists(
            user.id, "AAPL", "buy", 10, 198.50,
            datetime.strptime("2025-01-16", "%Y-%m-%d"),
        )

    def test_get_trades_filtered(self, repo, user):
        repo.add_trade(self._make_trade(user.id, "AAPL", "buy", 10, 100.00))
        repo.add_trade(self._make_trade(user.id, "MSFT", "buy", 5, 400.00))

        aapl_trades = repo.get_trades(user.id, ticker="AAPL")
        assert len(aapl_trades) == 1
        assert aapl_trades[0].ticker == "AAPL"

        all_trades = repo.get_trades(user.id)
        assert len(all_trades) == 2

    def test_fifo_step_by_step(self, repo, user):
        """
        Validates FIFO math at every step of a realistic trade sequence.

        The FIFO algorithm maintains a queue of "lots" (buy batches).
        When selling, it consumes lots from the FRONT (oldest first).

        Trade sequence:
          1. Buy  10 @ $100  →  lots: [10@100]
          2. Buy   5 @ $150  →  lots: [10@100, 5@150]
          3. Buy   5 @ $200  →  lots: [10@100, 5@150, 5@200]
          4. Sell  8 @ $180  →  consume 8 from front (10@100 → 2@100 remain)
                                lots: [2@100, 5@150, 5@200]
          5. Sell  4 @ $210  →  consume 2@100 (gone) + 2 from 5@150 (3@150 remain)
                                lots: [3@150, 5@200]
          6. Buy   3 @ $250  →  lots: [3@150, 5@200, 3@250]
          7. Sell 10 @ $230  →  consume 3@150 (gone) + 5@200 (gone) + 2 from 3@250
                                lots: [1@250]
        """

        # ── Step 1: Buy 10 @ $100 ──
        repo.add_trade(self._make_trade(user.id, "AAPL", "buy", 10, 100.00, "2025-01-01"))
        repo.recalculate_holdings(user.id)
        h = repo.get_holding(user.id, "AAPL")
        # lots: [10@100]  →  shares=10, cost=1000, avg=100
        assert h.shares == pytest.approx(10, abs=0.001)
        assert h.total_cost_basis == pytest.approx(1000.00, abs=0.01)
        assert h.avg_cost_basis == pytest.approx(100.00, abs=0.01)

        # ── Step 2: Buy 5 @ $150 ──
        repo.add_trade(self._make_trade(user.id, "AAPL", "buy", 5, 150.00, "2025-01-15"))
        repo.recalculate_holdings(user.id)
        h = repo.get_holding(user.id, "AAPL")
        # lots: [10@100, 5@150]  →  shares=15, cost=1750, avg=116.67
        assert h.shares == pytest.approx(15, abs=0.001)
        assert h.total_cost_basis == pytest.approx(1750.00, abs=0.01)
        assert h.avg_cost_basis == pytest.approx(116.6667, abs=0.01)

        # ── Step 3: Buy 5 @ $200 ──
        repo.add_trade(self._make_trade(user.id, "AAPL", "buy", 5, 200.00, "2025-02-01"))
        repo.recalculate_holdings(user.id)
        h = repo.get_holding(user.id, "AAPL")
        # lots: [10@100, 5@150, 5@200]  →  shares=20, cost=2750, avg=137.50
        assert h.shares == pytest.approx(20, abs=0.001)
        assert h.total_cost_basis == pytest.approx(2750.00, abs=0.01)
        assert h.avg_cost_basis == pytest.approx(137.50, abs=0.01)

        # ── Step 4: Sell 8 @ $180 ──
        # FIFO: consume 8 from lot[0] (10@100). 2@100 remain in that lot.
        # lots after: [2@100, 5@150, 5@200]
        repo.add_trade(self._make_trade(user.id, "AAPL", "sell", 8, 180.00, "2025-02-15"))
        repo.recalculate_holdings(user.id)
        h = repo.get_holding(user.id, "AAPL")
        # shares=12, cost = 2*100 + 5*150 + 5*200 = 200+750+1000 = 1950, avg=162.50
        assert h.shares == pytest.approx(12, abs=0.001)
        assert h.total_cost_basis == pytest.approx(1950.00, abs=0.01)
        assert h.avg_cost_basis == pytest.approx(162.50, abs=0.01)

        # ── Step 5: Sell 4 @ $210 ──
        # FIFO: consume 2@100 (lot gone) + 2 from 5@150 (3@150 remain)
        # lots after: [3@150, 5@200]
        repo.add_trade(self._make_trade(user.id, "AAPL", "sell", 4, 210.00, "2025-03-01"))
        repo.recalculate_holdings(user.id)
        h = repo.get_holding(user.id, "AAPL")
        # shares=8, cost = 3*150 + 5*200 = 450+1000 = 1450, avg=181.25
        assert h.shares == pytest.approx(8, abs=0.001)
        assert h.total_cost_basis == pytest.approx(1450.00, abs=0.01)
        assert h.avg_cost_basis == pytest.approx(181.25, abs=0.01)

        # ── Step 6: Buy 3 @ $250 ──
        # New lot appended at the BACK.
        # lots: [3@150, 5@200, 3@250]
        repo.add_trade(self._make_trade(user.id, "AAPL", "buy", 3, 250.00, "2025-03-15"))
        repo.recalculate_holdings(user.id)
        h = repo.get_holding(user.id, "AAPL")
        # shares=11, cost = 450+1000+750 = 2200, avg=200.00
        assert h.shares == pytest.approx(11, abs=0.001)
        assert h.total_cost_basis == pytest.approx(2200.00, abs=0.01)
        assert h.avg_cost_basis == pytest.approx(200.00, abs=0.01)

        # ── Step 7: Sell 10 @ $230 ──
        # FIFO: consume 3@150 (gone) + 5@200 (gone) + 2 from 3@250 (1@250 remain)
        # lots after: [1@250]
        repo.add_trade(self._make_trade(user.id, "AAPL", "sell", 10, 230.00, "2025-04-01"))
        repo.recalculate_holdings(user.id)
        h = repo.get_holding(user.id, "AAPL")
        # shares=1, cost = 1*250 = 250, avg=250.00
        assert h.shares == pytest.approx(1, abs=0.001)
        assert h.total_cost_basis == pytest.approx(250.00, abs=0.01)
        assert h.avg_cost_basis == pytest.approx(250.00, abs=0.01)

    def test_stock_split_adjusts_lots(self, repo, user):
        """
        Stock split should adjust existing lots, not add a $0 lot.

        Scenario (NVDA 10:1 split):
          1. Buy 5 @ $800           → lots: [5@800], cost=$4,000
          2. SPL adds 45 shares     → lots: [50@80], cost=$4,000 (spread, not new lot)
          3. Buy 10 @ $120          → lots: [50@80, 10@120], cost=$5,200
          4. Sell 3 @ $130          → FIFO: 3 from 50@80 → [47@80, 10@120]
                                       cost = 47*80 + 10*120 = 3,760+1,200 = $4,960
        """
        # Step 1: Pre-split buy
        repo.add_trade(self._make_trade(user.id, "NVDA", "buy", 5, 800.00, "2024-01-15"))
        repo.recalculate_holdings(user.id)
        h = repo.get_holding(user.id, "NVDA")
        assert h.shares == pytest.approx(5, abs=0.001)
        assert h.total_cost_basis == pytest.approx(4000.00, abs=0.01)

        # Step 2: Stock split — 10:1 (adds 45 shares)
        spl_trade = Trade(
            user_id=user.id,
            ticker="NVDA",
            trade_type="buy",
            shares=45,
            price_per_share=0.0,
            total_amount=0.0,
            trade_date=datetime.strptime("2024-06-10", "%Y-%m-%d"),
            source="robinhood_csv_spl",
        )
        repo.add_trade(spl_trade)
        repo.recalculate_holdings(user.id)
        h = repo.get_holding(user.id, "NVDA")
        # Total cost preserved, shares adjusted
        assert h.shares == pytest.approx(50, abs=0.001)
        assert h.total_cost_basis == pytest.approx(4000.00, abs=0.01)
        assert h.avg_cost_basis == pytest.approx(80.00, abs=0.01)

        # Step 3: Post-split buy
        repo.add_trade(self._make_trade(user.id, "NVDA", "buy", 10, 120.00, "2024-07-01"))
        repo.recalculate_holdings(user.id)
        h = repo.get_holding(user.id, "NVDA")
        assert h.shares == pytest.approx(60, abs=0.001)
        assert h.total_cost_basis == pytest.approx(5200.00, abs=0.01)
        # avg = 5200/60 ≈ 86.67
        assert h.avg_cost_basis == pytest.approx(86.6667, abs=0.01)

        # Step 4: Sell 3 shares — FIFO consumes from split-adjusted lot at $80
        repo.add_trade(self._make_trade(user.id, "NVDA", "sell", 3, 130.00, "2024-08-01"))
        repo.recalculate_holdings(user.id)
        h = repo.get_holding(user.id, "NVDA")
        assert h.shares == pytest.approx(57, abs=0.001)
        # cost = 47*80 + 10*120 = 3760 + 1200 = 4960
        assert h.total_cost_basis == pytest.approx(4960.00, abs=0.01)
        assert h.avg_cost_basis == pytest.approx(4960.00 / 57, abs=0.01)

    def test_mrgs_stock_split_adjusts_lots(self, repo, user):
        """
        MRGS-style stock split (used by Robinhood for some splits like SMCI).

        Unlike SPL where quantity = NEW shares added, MRGS records:
          - One row with quantity = TOTAL post-split shares (the "receive" side)
          - One row with quantity = old shares surrendered (the "surrender" side)

        Scenario (SMCI 10:1 split):
          1. Buy 3 @ $900           → lots: [3@900], cost=$2,700
          2. MRGS surrender 3 (skipped — qty <= existing)
          3. MRGS receive 30         → ratio=30/3=10, lots: [30@90], cost=$2,700
          4. Buy 70 @ $40            → lots: [30@90, 70@40], cost=$2,700+$2,800=$5,500
          5. Expected: 100 shares, avg=$55.00
        """
        # Step 1: Pre-split buy
        repo.add_trade(self._make_trade(user.id, "SMCI", "buy", 3, 900.00, "2024-01-15"))
        repo.recalculate_holdings(user.id)
        h = repo.get_holding(user.id, "SMCI")
        assert h.shares == pytest.approx(3, abs=0.001)
        assert h.total_cost_basis == pytest.approx(2700.00, abs=0.01)

        # Step 2: MRGS surrender row (3 shares — will be skipped since qty <= existing)
        mrgs_surrender = Trade(
            user_id=user.id,
            ticker="SMCI",
            trade_type="buy",
            shares=3,
            price_per_share=0.0,
            total_amount=0.0,
            trade_date=datetime.strptime("2024-10-01", "%Y-%m-%d"),
            source="robinhood_csv_mrgs",
        )
        repo.add_trade(mrgs_surrender)

        # Step 3: MRGS receive row (30 = total post-split shares)
        mrgs_receive = Trade(
            user_id=user.id,
            ticker="SMCI",
            trade_type="buy",
            shares=30,
            price_per_share=0.0,
            total_amount=0.0,
            trade_date=datetime.strptime("2024-10-01", "%Y-%m-%d"),
            source="robinhood_csv_mrgs",
        )
        repo.add_trade(mrgs_receive)
        repo.recalculate_holdings(user.id)
        h = repo.get_holding(user.id, "SMCI")
        # Total cost preserved, shares adjusted via ratio = 30/3 = 10
        assert h.shares == pytest.approx(30, abs=0.001)
        assert h.total_cost_basis == pytest.approx(2700.00, abs=0.01)
        assert h.avg_cost_basis == pytest.approx(90.00, abs=0.01)

        # Step 4: Post-split buy
        repo.add_trade(self._make_trade(user.id, "SMCI", "buy", 70, 40.00, "2024-11-01"))
        repo.recalculate_holdings(user.id)
        h = repo.get_holding(user.id, "SMCI")
        assert h.shares == pytest.approx(100, abs=0.001)
        # cost = 30*90 + 70*40 = 2700 + 2800 = 5500
        assert h.total_cost_basis == pytest.approx(5500.00, abs=0.01)
        assert h.avg_cost_basis == pytest.approx(55.00, abs=0.01)


# ──────────────────────────────────────────────
# CSV Import
# ──────────────────────────────────────────────


class TestCSVImport:
    def test_parse_fixture(self):
        from marketmind.tools.importers import parse_robinhood_csv

        csv_path = FIXTURES_DIR / "robinhood_sample.csv"
        trades, skipped = parse_robinhood_csv(csv_path, user_id=1)

        # Should find: Buy AAPL, Buy MSFT, Buy AAPL, Sell AAPL, SPL TSLA = 5 trades
        assert len(trades) == 5

        buy_tickers = [t.ticker for t in trades if t.trade_type == "buy"]
        sell_tickers = [t.ticker for t in trades if t.trade_type == "sell"]
        assert "AAPL" in buy_tickers
        assert "MSFT" in buy_tickers
        assert "AAPL" in sell_tickers

        # SPL row is now imported as a buy with source=robinhood_csv_spl
        spl_trades = [t for t in trades if t.source == "robinhood_csv_spl"]
        assert len(spl_trades) == 1
        assert spl_trades[0].ticker == "TSLA"
        assert spl_trades[0].price_per_share == 0.0

    def test_amount_parsing(self):
        from marketmind.tools.importers import _parse_amount

        assert _parse_amount("$1,985.00") == pytest.approx(1985.00)
        assert _parse_amount("($1,985.00)") == pytest.approx(-1985.00)
        assert _parse_amount("-$500.00") == pytest.approx(-500.00)
        assert _parse_amount("$3.50") == pytest.approx(3.50)

    def test_skipped_entries(self):
        from marketmind.tools.importers import parse_robinhood_csv

        csv_path = FIXTURES_DIR / "robinhood_sample.csv"
        _, skipped = parse_robinhood_csv(csv_path, user_id=1)

        # SPL is now imported as a trade, ACH/CDIV/INT are silently ignored.
        # No entries in our fixture produce skipped entries.
        assert len(skipped) == 0

    def test_trade_attributes(self):
        from marketmind.tools.importers import parse_robinhood_csv

        csv_path = FIXTURES_DIR / "robinhood_sample.csv"
        trades, _ = parse_robinhood_csv(csv_path, user_id=1)

        # First trade: Buy 10 AAPL @ $198.50
        first_buy = trades[0]
        assert first_buy.ticker == "AAPL"
        assert first_buy.trade_type == "buy"
        assert first_buy.shares == pytest.approx(10)
        assert first_buy.price_per_share == pytest.approx(198.50)
        assert first_buy.source == "robinhood_csv"

    def test_import_and_recalculate(self, repo, user):
        """Full integration: parse CSV → insert trades → recalculate holdings."""
        from marketmind.tools.importers import parse_robinhood_csv

        csv_path = FIXTURES_DIR / "robinhood_sample.csv"
        trades, _ = parse_robinhood_csv(csv_path, user_id=user.id)

        for trade in trades:
            repo.add_trade(trade)
        repo.recalculate_holdings(user.id)

        holdings = repo.get_holdings(user.id)
        tickers = {h.ticker for h in holdings}

        # After: Buy 10 AAPL, Buy 5 AAPL, Sell 8 AAPL → 7 AAPL remaining
        # MSFT: Buy 5 → 5 MSFT remaining
        assert "AAPL" in tickers
        assert "MSFT" in tickers

        aapl = repo.get_holding(user.id, "AAPL")
        assert aapl.shares == pytest.approx(7, abs=0.001)

        msft = repo.get_holding(user.id, "MSFT")
        assert msft.shares == pytest.approx(5, abs=0.001)


# ──────────────────────────────────────────────
# Watchlist
# ──────────────────────────────────────────────


class TestWatchlist:
    def test_add_to_watchlist(self, repo, user):
        item = repo.add_to_watchlist(user.id, "NVDA", "AI play")
        assert item.ticker == "NVDA"
        assert item.notes == "AI play"

    def test_list_watchlist(self, repo, user):
        repo.add_to_watchlist(user.id, "NVDA")
        repo.add_to_watchlist(user.id, "TSLA")
        items = repo.get_watchlist(user.id)
        assert len(items) == 2
        tickers = {i.ticker for i in items}
        assert tickers == {"NVDA", "TSLA"}

    def test_duplicate_ignored(self, repo, user):
        repo.add_to_watchlist(user.id, "NVDA")
        repo.add_to_watchlist(user.id, "NVDA")  # Should not raise
        items = repo.get_watchlist(user.id)
        assert len(items) == 1

    def test_remove_from_watchlist(self, repo, user):
        repo.add_to_watchlist(user.id, "NVDA")
        assert repo.remove_from_watchlist(user.id, "NVDA") is True
        items = repo.get_watchlist(user.id)
        assert len(items) == 0

    def test_remove_nonexistent(self, repo, user):
        assert repo.remove_from_watchlist(user.id, "ZZZZZ") is False


# ──────────────────────────────────────────────
# Analysis caching
# ──────────────────────────────────────────────


class TestAnalysisCaching:
    def test_save_and_retrieve(self, repo):
        analysis = StockAnalysisCache(
            ticker="AAPL",
            analysis_type="comprehensive",
            summary="Test summary",
            detailed_analysis="Detailed analysis text here.",
            model_used="claude-sonnet-4-20250514",
            cost_usd=0.05,
        )
        saved = repo.save_analysis(analysis)
        assert saved.id is not None

        cached = repo.get_recent_analysis("AAPL", "comprehensive")
        assert cached is not None
        assert cached.ticker == "AAPL"
        assert cached.summary == "Test summary"
        assert cached.detailed_analysis == "Detailed analysis text here."

    def test_staleness_check(self, repo, db):
        analysis = StockAnalysisCache(
            ticker="MSFT",
            analysis_type="comprehensive",
            summary="Old summary",
            detailed_analysis="Old analysis.",
            model_used="claude-sonnet-4-20250514",
            cost_usd=0.03,
        )
        repo.save_analysis(analysis)

        # Manually backdate the record to 25 hours ago
        old_time = (datetime.now() - timedelta(hours=25)).isoformat()
        db.connection.execute(
            "UPDATE stock_analyses SET created_at = ? WHERE ticker = 'MSFT'",
            (old_time,),
        )
        db.connection.commit()

        cached = repo.get_recent_analysis("MSFT", "comprehensive", max_age_hours=24)
        assert cached is None

    def test_different_analysis_types(self, repo):
        repo.save_analysis(StockAnalysisCache(
            ticker="AAPL",
            analysis_type="comprehensive",
            summary="Comprehensive",
            detailed_analysis="...",
            model_used="claude",
            cost_usd=0.05,
        ))
        repo.save_analysis(StockAnalysisCache(
            ticker="AAPL",
            analysis_type="educational",
            summary="Educational",
            detailed_analysis="...",
            model_used="claude",
            cost_usd=0.08,
        ))

        comp = repo.get_recent_analysis("AAPL", "comprehensive")
        edu = repo.get_recent_analysis("AAPL", "educational")
        assert comp.summary == "Comprehensive"
        assert edu.summary == "Educational"


# ──────────────────────────────────────────────
# User manager (file-based persistence)
# ──────────────────────────────────────────────


class TestUserManager:
    def test_default_username(self, tmp_dir):
        from marketmind.tools.user_manager import get_active_username

        assert get_active_username(tmp_dir) == "default"

    def test_set_and_get(self, tmp_dir):
        from marketmind.tools.user_manager import get_active_username, set_active_username

        set_active_username(tmp_dir, "alice")
        assert get_active_username(tmp_dir) == "alice"

    def test_overwrite(self, tmp_dir):
        from marketmind.tools.user_manager import get_active_username, set_active_username

        set_active_username(tmp_dir, "alice")
        set_active_username(tmp_dir, "bob")
        assert get_active_username(tmp_dir) == "bob"
