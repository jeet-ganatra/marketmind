"""
Brokerage CSV Importers — parse trade history from brokerage exports.

Currently supports:
- Robinhood (Activity export CSV)

Each importer returns a list of Trade objects and a list of skipped/logged
entries for transparency.
"""

import csv
from datetime import datetime
from pathlib import Path

from rich.console import Console

from marketmind.models.schemas import Trade

console = Console()

# Transaction codes that represent actual equity trades
EQUITY_TRADE_CODES = {"Buy", "Sell"}

# Codes that add shares without a cash purchase (splits, transfers, received shares).
# Treated as buys at $0 so FIFO has the correct share count.
SHARE_ADDITION_CODES = {"SPL", "ACATI", "REC"}

# Codes to log but skip (mergers, etc.)
LOG_CODES = {"MRGS"}

# Codes to silently ignore (cash, dividends, interest, fees, etc.)
IGNORE_CODES = {"ACH", "CDIV", "INT", "DTAX", "GOLD", "SLIP", "GDBP", "GPMC", "GMPC", "ABIP", "RTP"}


def parse_robinhood_csv(
    csv_path: Path,
    user_id: int,
) -> tuple[list[Trade], list[dict]]:
    """
    Parse a Robinhood activity CSV into Trade objects.

    Args:
        csv_path: Path to the CSV file
        user_id: The user_id to associate trades with

    Returns:
        Tuple of (trades, skipped_entries).
        trades: list of Trade objects ready for insertion
        skipped_entries: list of dicts with {row, reason} for transparency

    Robinhood CSV columns:
        Activity Date, Process Date, Settle Date, Instrument,
        Description, Trans Code, Quantity, Price, Amount
    """
    trades: list[Trade] = []
    skipped: list[dict] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row_num, row in enumerate(reader, start=2):  # header is row 1
            trans_code = (row.get("Trans Code") or "").strip()
            instrument = (row.get("Instrument") or "").strip()

            if not trans_code:
                continue

            if trans_code in EQUITY_TRADE_CODES:
                trade = _parse_equity_trade(row, user_id, row_num)
                if trade:
                    trades.append(trade)
                else:
                    skipped.append({"row": row_num, "reason": f"Could not parse trade: {row}"})
                continue

            if trans_code in SHARE_ADDITION_CODES:
                trade = _parse_share_addition(row, trans_code, user_id, row_num)
                if trade:
                    trades.append(trade)
                else:
                    skipped.append({
                        "row": row_num,
                        "reason": (
                            f"Logged (no quantity): {trans_code} - {instrument}"
                            f" - {(row.get('Description') or '')}"
                        ),
                    })
                continue

            if trans_code in LOG_CODES:
                skipped.append({
                    "row": row_num,
                    "reason": (
                        f"Logged (non-trade): {trans_code} - {instrument}"
                        f" - {row.get('Description', '')}"
                    ),
                })
                continue

            if trans_code in IGNORE_CODES:
                continue

            # Unknown code — log it
            skipped.append({
                "row": row_num,
                "reason": f"Unknown trans code: {trans_code} - {instrument}",
            })

    return trades, skipped


def _parse_equity_trade(row: dict, user_id: int, row_num: int) -> Trade | None:
    """Parse a single Buy or Sell row into a Trade model."""
    try:
        ticker = row["Instrument"].strip().upper()
        if not ticker:
            return None

        trans_code = row["Trans Code"].strip()
        trade_type = "buy" if trans_code == "Buy" else "sell"

        date_str = row["Activity Date"].strip()
        trade_date = datetime.strptime(date_str, "%m/%d/%Y")

        quantity_str = row["Quantity"].strip()
        if not quantity_str:
            return None
        shares = float(quantity_str)
        if shares <= 0:
            return None

        price_str = row["Price"].strip().lstrip("$").replace(",", "")
        if not price_str:
            return None
        price = float(price_str)

        amount_str = row["Amount"].strip()
        amount = _parse_amount(amount_str)

        return Trade(
            user_id=user_id,
            ticker=ticker,
            trade_type=trade_type,
            shares=shares,
            price_per_share=price,
            total_amount=abs(amount),
            trade_date=trade_date,
            source="robinhood_csv",
        )
    except (KeyError, ValueError) as e:
        console.print(f"  [yellow]Warning: Could not parse row {row_num}: {e}[/yellow]")
        return None


def _parse_share_addition(row: dict, trans_code: str, user_id: int, row_num: int) -> Trade | None:
    """
    Parse a stock split (SPL), account transfer (ACATI), or received shares (REC).

    These add shares without a cash purchase. We record them as buys at $0
    so the FIFO lot queue has the correct total share count. The original
    cost basis from prior buys is preserved — these $0 lots just ensure
    post-split/transfer sells don't exceed known shares.
    """
    try:
        ticker = (row.get("Instrument") or "").strip().upper()
        if not ticker:
            return None

        quantity_str = (row.get("Quantity") or "").strip()
        if not quantity_str:
            return None
        shares = float(quantity_str)
        if shares <= 0:
            return None

        date_str = (row.get("Activity Date") or "").strip()
        trade_date = datetime.strptime(date_str, "%m/%d/%Y") if date_str else datetime.now()

        return Trade(
            user_id=user_id,
            ticker=ticker,
            trade_type="buy",
            shares=shares,
            price_per_share=0.0,
            total_amount=0.0,
            trade_date=trade_date,
            source=f"robinhood_csv_{trans_code.lower()}",
        )
    except (KeyError, ValueError) as e:
        console.print(f"  [yellow]Warning: Could not parse {trans_code} row {row_num}: {e}[/yellow]")
        return None


def _parse_amount(amount_str: str) -> float:
    """
    Parse dollar amount, handling:
    - "$1,234.56"   ->  1234.56
    - "($1,234.56)" -> -1234.56  (parentheses = negative in accounting)
    - "-$1,234.56"  -> -1234.56
    """
    negative = False
    s = amount_str.strip()
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]
    if s.startswith("-"):
        negative = True
        s = s[1:]
    s = s.lstrip("$").replace(",", "")
    value = float(s) if s else 0.0
    return -value if negative else value
