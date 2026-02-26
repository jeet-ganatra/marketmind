"""
Portfolio Repository — all database operations for portfolio tracking.

Handles:
- User management (create, get, get_or_create)
- Trade recording with duplicate prevention
- FIFO cost basis recalculation
- Holdings queries
- Analysis caching (save + freshness check)
- Watchlist management
"""

import sqlite3
from datetime import datetime, timedelta

from marketmind.db.database import Database
from marketmind.models.schemas import (
    Holding,
    StockAnalysisCache,
    Trade,
    User,
    WatchlistItem,
)


class PortfolioRepository:
    """Encapsulates all database queries, keeping SQL out of the CLI and orchestrator."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.conn = db.connection

    # ──────────────────────────────────────
    # User management
    # ──────────────────────────────────────

    def create_user(self, username: str) -> User:
        """Create a new user. Raises sqlite3.IntegrityError if username exists."""
        cursor = self.conn.execute(
            "INSERT INTO users (username) VALUES (?)", (username,)
        )
        self.conn.commit()
        return User(id=cursor.lastrowid, username=username)

    def get_user(self, username: str) -> User | None:
        """Get user by username, or None if not found."""
        row = self.conn.execute(
            "SELECT id, username, created_at FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if not row:
            return None
        return User(
            id=row["id"],
            username=row["username"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def get_or_create_user(self, username: str) -> User:
        """Get existing user or create a new one."""
        user = self.get_user(username)
        if user:
            return user
        return self.create_user(username)

    # ──────────────────────────────────────
    # Trade management
    # ──────────────────────────────────────

    def add_trade(self, trade: Trade) -> Trade:
        """
        Insert a trade record. Returns the trade with its new id.
        Caller should call recalculate_holdings() after batch inserts.
        """
        cursor = self.conn.execute(
            """INSERT INTO trades
               (user_id, ticker, trade_type, shares, price_per_share,
                total_amount, trade_date, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trade.user_id,
                trade.ticker.upper(),
                trade.trade_type,
                trade.shares,
                trade.price_per_share,
                trade.total_amount,
                trade.trade_date.isoformat(),
                trade.source,
            ),
        )
        self.conn.commit()
        trade.id = cursor.lastrowid
        return trade

    def trade_exists(
        self,
        user_id: int,
        ticker: str,
        trade_type: str,
        shares: float,
        price: float,
        trade_date: datetime,
    ) -> bool:
        """Check for duplicate trade (used during CSV import)."""
        row = self.conn.execute(
            """SELECT 1 FROM trades
               WHERE user_id = ? AND ticker = ? AND trade_type = ?
               AND abs(shares - ?) < 0.0001
               AND abs(price_per_share - ?) < 0.01
               AND date(trade_date) = date(?)
               LIMIT 1""",
            (user_id, ticker.upper(), trade_type, shares, price, trade_date.isoformat()),
        ).fetchone()
        return row is not None

    def get_trades(self, user_id: int, ticker: str | None = None) -> list[Trade]:
        """Get all trades for a user, optionally filtered by ticker."""
        if ticker:
            rows = self.conn.execute(
                """SELECT * FROM trades
                   WHERE user_id = ? AND ticker = ?
                   ORDER BY trade_date ASC""",
                (user_id, ticker.upper()),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT * FROM trades WHERE user_id = ? ORDER BY trade_date ASC""",
                (user_id,),
            ).fetchall()
        return [_row_to_trade(r) for r in rows]

    # ──────────────────────────────────────
    # Holdings (FIFO cost basis)
    # ──────────────────────────────────────

    def recalculate_holdings(self, user_id: int) -> None:
        """
        Recalculate all holdings for a user from their trade history
        using FIFO (First In, First Out) cost basis.

        FIFO: maintain a queue of buy lots. When selling, consume from
        the front (oldest lots first). Remaining lots determine cost basis.
        """
        tickers = self.conn.execute(
            "SELECT DISTINCT ticker FROM trades WHERE user_id = ?",
            (user_id,),
        ).fetchall()

        for row in tickers:
            self._recalculate_single_holding(user_id, row["ticker"])

    def _recalculate_single_holding(self, user_id: int, ticker: str) -> None:
        """FIFO recalculation for a single ticker."""
        trades = self.get_trades(user_id, ticker)
        if not trades:
            return

        # FIFO lot queue: [[remaining_shares, price_per_share], ...]
        lots: list[list[float]] = []
        first_purchase: datetime | None = None
        last_trade: datetime | None = None

        for trade in trades:
            last_trade = trade.trade_date
            if trade.trade_type == "buy":
                # Stock split (SPL): quantity = NEW shares added
                # ratio = (existing + new) / existing
                if trade.source.endswith("_spl") and lots and trade.price_per_share == 0:
                    existing_shares = sum(lot[0] for lot in lots)
                    if existing_shares > 0.0001:
                        ratio = (existing_shares + trade.shares) / existing_shares
                        for lot in lots:
                            lot[1] /= ratio  # spread cost across more shares
                            lot[0] *= ratio  # multiply share count
                # Merger/reorganization (MRGS): quantity = TOTAL post-split shares
                # ratio = total / existing
                # Skip surrendered shares (where quantity <= existing — that's the old shares)
                elif trade.source.endswith("_mrgs") and lots and trade.price_per_share == 0:
                    existing_shares = sum(lot[0] for lot in lots)
                    if existing_shares > 0.0001 and trade.shares > existing_shares:
                        ratio = trade.shares / existing_shares
                        for lot in lots:
                            lot[1] /= ratio  # spread cost across more shares
                            lot[0] *= ratio  # multiply share count
                    # else: surrendered shares row (qty <= existing), skip it
                else:
                    lots.append([trade.shares, trade.price_per_share])
                    if first_purchase is None:
                        first_purchase = trade.trade_date
            elif trade.trade_type == "sell":
                shares_to_sell = trade.shares
                while shares_to_sell > 0.0001 and lots:
                    lot = lots[0]
                    if lot[0] <= shares_to_sell:
                        shares_to_sell -= lot[0]
                        lots.pop(0)
                    else:
                        lot[0] -= shares_to_sell
                        shares_to_sell = 0

        total_shares = sum(lot[0] for lot in lots)
        total_cost = sum(lot[0] * lot[1] for lot in lots)
        avg_cost = total_cost / total_shares if total_shares > 0.0001 else 0

        if total_shares < 0.0001:
            # Position fully closed
            self.conn.execute(
                "DELETE FROM holdings WHERE user_id = ? AND ticker = ?",
                (user_id, ticker),
            )
        else:
            self.conn.execute(
                """INSERT INTO holdings
                   (user_id, ticker, shares, avg_cost_basis, total_cost_basis,
                    first_purchase_date, last_trade_date, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
                   ON CONFLICT(user_id, ticker) DO UPDATE SET
                       shares = excluded.shares,
                       avg_cost_basis = excluded.avg_cost_basis,
                       total_cost_basis = excluded.total_cost_basis,
                       first_purchase_date = excluded.first_purchase_date,
                       last_trade_date = excluded.last_trade_date,
                       updated_at = datetime('now')""",
                (
                    user_id,
                    ticker,
                    round(total_shares, 6),
                    round(avg_cost, 4),
                    round(total_cost, 4),
                    first_purchase.isoformat() if first_purchase else None,
                    last_trade.isoformat() if last_trade else None,
                ),
            )
        self.conn.commit()

    def get_holdings(self, user_id: int) -> list[Holding]:
        """Get all current holdings for a user."""
        rows = self.conn.execute(
            "SELECT * FROM holdings WHERE user_id = ? AND shares > 0 ORDER BY ticker",
            (user_id,),
        ).fetchall()
        return [_row_to_holding(r) for r in rows]

    def get_holding(self, user_id: int, ticker: str) -> Holding | None:
        """Get a specific holding."""
        row = self.conn.execute(
            "SELECT * FROM holdings WHERE user_id = ? AND ticker = ?",
            (user_id, ticker.upper()),
        ).fetchone()
        return _row_to_holding(row) if row else None

    # ──────────────────────────────────────
    # Analysis caching
    # ──────────────────────────────────────

    def save_analysis(self, analysis: StockAnalysisCache) -> StockAnalysisCache:
        """Save an analysis result to the shared cache."""
        cursor = self.conn.execute(
            """INSERT INTO stock_analyses
               (ticker, analysis_type, user_id, summary, detailed_analysis,
                model_used, cost_usd)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                analysis.ticker.upper(),
                analysis.analysis_type,
                analysis.user_id,
                analysis.summary,
                analysis.detailed_analysis,
                analysis.model_used,
                analysis.cost_usd,
            ),
        )
        self.conn.commit()
        analysis.id = cursor.lastrowid
        return analysis

    def get_recent_analysis(
        self,
        ticker: str,
        analysis_type: str,
        max_age_hours: int = 24,
    ) -> StockAnalysisCache | None:
        """Get a cached analysis if one exists within max_age_hours."""
        cutoff = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()
        row = self.conn.execute(
            """SELECT * FROM stock_analyses
               WHERE ticker = ? AND analysis_type = ?
               AND created_at > ?
               ORDER BY created_at DESC LIMIT 1""",
            (ticker.upper(), analysis_type, cutoff),
        ).fetchone()
        if not row:
            return None
        return _row_to_analysis_cache(row)

    # ──────────────────────────────────────
    # Watchlist
    # ──────────────────────────────────────

    def add_to_watchlist(self, user_id: int, ticker: str, notes: str = "") -> WatchlistItem:
        """Add a ticker to the watchlist. Ignores duplicates."""
        self.conn.execute(
            "INSERT OR IGNORE INTO watchlist (user_id, ticker, notes) VALUES (?, ?, ?)",
            (user_id, ticker.upper(), notes),
        )
        self.conn.commit()
        return WatchlistItem(user_id=user_id, ticker=ticker.upper(), notes=notes)

    def get_watchlist(self, user_id: int) -> list[WatchlistItem]:
        """Get all watchlist items for a user."""
        rows = self.conn.execute(
            "SELECT * FROM watchlist WHERE user_id = ? ORDER BY ticker",
            (user_id,),
        ).fetchall()
        return [_row_to_watchlist_item(r) for r in rows]

    def remove_from_watchlist(self, user_id: int, ticker: str) -> bool:
        """Remove a ticker from the watchlist. Returns True if removed."""
        cursor = self.conn.execute(
            "DELETE FROM watchlist WHERE user_id = ? AND ticker = ?",
            (user_id, ticker.upper()),
        )
        self.conn.commit()
        return cursor.rowcount > 0


# ──────────────────────────────────────────────
# Row-to-model converters
# ──────────────────────────────────────────────


def _row_to_trade(row: sqlite3.Row) -> Trade:
    return Trade(
        id=row["id"],
        user_id=row["user_id"],
        ticker=row["ticker"],
        trade_type=row["trade_type"],
        shares=row["shares"],
        price_per_share=row["price_per_share"],
        total_amount=row["total_amount"],
        trade_date=datetime.fromisoformat(row["trade_date"]),
        source=row["source"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _row_to_holding(row: sqlite3.Row) -> Holding:
    return Holding(
        id=row["id"],
        user_id=row["user_id"],
        ticker=row["ticker"],
        shares=row["shares"],
        avg_cost_basis=row["avg_cost_basis"],
        total_cost_basis=row["total_cost_basis"],
        first_purchase_date=(
            datetime.fromisoformat(row["first_purchase_date"])
            if row["first_purchase_date"]
            else None
        ),
        last_trade_date=(
            datetime.fromisoformat(row["last_trade_date"])
            if row["last_trade_date"]
            else None
        ),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _row_to_analysis_cache(row: sqlite3.Row) -> StockAnalysisCache:
    return StockAnalysisCache(
        id=row["id"],
        ticker=row["ticker"],
        analysis_type=row["analysis_type"],
        user_id=row["user_id"],
        summary=row["summary"],
        detailed_analysis=row["detailed_analysis"],
        model_used=row["model_used"],
        cost_usd=row["cost_usd"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _row_to_watchlist_item(row: sqlite3.Row) -> WatchlistItem:
    return WatchlistItem(
        id=row["id"],
        user_id=row["user_id"],
        ticker=row["ticker"],
        notes=row["notes"] or "",
        added_at=datetime.fromisoformat(row["added_at"]),
    )
