"""
Database connection manager for MarketMind.

Manages the SQLite database lifecycle: connection, schema creation,
and connection cleanup. All tables are created idempotently with
IF NOT EXISTS so initialize() is safe to call on every startup.
"""

import sqlite3
from pathlib import Path


class Database:
    """
    Manages the SQLite database connection and schema.

    Usage:
        db = Database(data_dir=Path("./data"))
        db.initialize()
        conn = db.connection
        # ... use conn ...
        db.close()
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.db_path = data_dir / "marketmind.db"
        self._connection: sqlite3.Connection | None = None

    @property
    def connection(self) -> sqlite3.Connection:
        """Get or create the database connection."""
        if self._connection is None:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(str(self.db_path))
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
        return self._connection

    def initialize(self) -> None:
        """Create all tables if they don't exist. Safe to call multiple times."""
        conn = self.connection
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                ticker TEXT NOT NULL,
                trade_type TEXT NOT NULL CHECK(trade_type IN ('buy', 'sell')),
                shares REAL NOT NULL CHECK(shares > 0),
                price_per_share REAL NOT NULL CHECK(price_per_share >= 0),
                total_amount REAL NOT NULL,
                trade_date TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'manual',
                raw_description TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS holdings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                ticker TEXT NOT NULL,
                shares REAL NOT NULL DEFAULT 0,
                avg_cost_basis REAL NOT NULL DEFAULT 0,
                total_cost_basis REAL NOT NULL DEFAULT 0,
                first_purchase_date TEXT,
                last_trade_date TEXT,
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(user_id, ticker)
            );

            CREATE TABLE IF NOT EXISTS stock_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                analysis_type TEXT NOT NULL,
                user_id INTEGER REFERENCES users(id),
                summary TEXT NOT NULL,
                detailed_analysis TEXT NOT NULL,
                model_used TEXT NOT NULL,
                cost_usd REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                ticker TEXT NOT NULL,
                notes TEXT DEFAULT '',
                added_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(user_id, ticker)
            );

            CREATE INDEX IF NOT EXISTS idx_trades_user_ticker
                ON trades(user_id, ticker, trade_date);
            CREATE INDEX IF NOT EXISTS idx_holdings_user
                ON holdings(user_id);
            CREATE INDEX IF NOT EXISTS idx_analyses_ticker_type
                ON stock_analyses(ticker, analysis_type, created_at);
            CREATE INDEX IF NOT EXISTS idx_watchlist_user
                ON watchlist(user_id);
        """)

    def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
