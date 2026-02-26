# FIFO Cost Basis and Stock Splits: A Subtle Data Pipeline Bug

## The Problem

After importing Robinhood trade history via CSV, NVDA (the user's largest position) was completely missing from `portfolio show`. No error, no warning. The stock simply didn't appear.

The FIFO cost basis algorithm was computing 0 remaining shares for NVDA and deleting the holding entirely. Every other stock imported correctly. The bug was specific to stocks that had undergone a stock split.

---

## Root Cause, Phase 1: Missing Data

The Robinhood activity CSV contains records for every account event: buys, sells, dividends, transfers, and stock splits. Each record has a `Trans Code` column that identifies the event type.

Our CSV importer categorized transaction codes into three buckets:

```python
EQUITY_TRADE_CODES = {"Buy", "Sell"}      # Parse as trades
LOG_CODES = {"SPL", "MRGS", "ACATI"}      # Log and skip
IGNORE_CODES = {"ACH", "CDIV", "INT"}     # Silently ignore
```

The `SPL` (stock split) code was in `LOG_CODES`, meaning split records were logged for transparency but not imported. This seemed reasonable during initial development: splits don't involve cash changing hands, so they aren't "trades" in the traditional sense.

NVDA underwent a 10:1 stock split in June 2024. The split record was the only thing connecting the pre-split purchase history to the post-split share count. Without it, the FIFO algorithm saw this:

```
What the CSV contained:            What our importer processed:
─────────────────────────          ──────────────────────────────
Buy 5 NVDA @ $800                  Imported as buy
SPL: receive 45 new shares         Skipped (LOG_CODES)
Sell 10 NVDA @ $130                Imported as sell

FIFO calculation:
  Buy 5 shares → lots: [5 @ $800]
  Sell 10 shares → consume 5 from lot (lot empty), 5 more to sell but no lots left
  Remaining shares: 0 → holding deleted

Reality:
  Buy 5 → split 10:1 → 50 shares → sell 10 → 40 shares remaining
```

The sell of 10 post-split shares exceeded the 5 pre-split shares on record. The FIFO algorithm drained the lot queue, computed 0 remaining shares, and deleted the holding from the database.

---

## First Fix Attempt: The $0 Cost Lot

The initial fix treated SPL records as buy trades with the split shares at $0:

```python
# SPL row: Instrument=NVDA, Quantity=45, Price=$0.00
Trade(ticker="NVDA", trade_type="buy", shares=45, price_per_share=0.0)
```

The reasoning: adding 45 shares at $0 would give FIFO the correct total share count while preserving the original cost basis.

```
lots: [5 @ $800, 45 @ $0]
total shares = 50
total cost = 5 * 800 + 45 * 0 = $4,000
avg cost = $4,000 / 50 = $80 per share
```

$80 is the correct post-split average cost ($800 / 10 = $80). NVDA reappeared in the portfolio. The fix appeared to work.

---

## Root Cause, Phase 2: FIFO Ordering

The $0 lot approach preserved total cost but broke the lot-level cost distribution. The bug surfaced when the user noticed the average cost didn't match Robinhood's reported value.

FIFO is order-dependent. Sells always consume from the front of the lot queue (oldest lots first). With the $0 lot approach, the queue looked like this:

```
lots: [5 @ $800, 45 @ $0]
       ^^^^^^^
       front of queue (consumed first on sells)
```

When selling 3 shares, FIFO consumed from the $800 lot:

```
$0 Lot Approach (WRONG):
  Before sell:  [5 @ $800, 45 @ $0]
  Sell 3:       consume 3 from front lot
  After sell:   [2 @ $800, 45 @ $0]

  total cost = 2 * 800 + 45 * 0 = $1,600
  total shares = 47
  avg cost = $1,600 / 47 = $34.04

  Robinhood reported: $54.98
```

The problem: the $800 pre-split cost was concentrated in 5 shares at the front of the queue, while 45 shares carried $0 cost at the back. Selling even a few shares disproportionately consumed the high-cost lots, leaving behind mostly $0-cost shares and producing an artificially low average.

Total-level math was correct ($4,000 across 50 shares = $80 avg). But as soon as any sells occurred, the lot-level distribution produced wrong numbers.

---

## The Correct Fix: Lot Restructuring

A stock split is not a new transaction. It is a retroactive adjustment to every existing lot. The correct handling: when a split occurs, restructure all lots in the queue by applying the split ratio.

The algorithm:

1. Calculate the split ratio: `(existing_shares + new_shares) / existing_shares`
2. For each lot: multiply shares by ratio, divide price by ratio
3. Total cost per lot is preserved. Share count and per-share cost are adjusted.

```python
if trade.source.endswith("_spl") and lots and trade.price_per_share == 0:
    existing_shares = sum(lot[0] for lot in lots)
    ratio = (existing_shares + trade.shares) / existing_shares
    for lot in lots:
        lot[1] /= ratio  # spread cost across more shares
        lot[0] *= ratio  # multiply share count
```

Single-lot example:

```
Lot Restructuring (CORRECT):
  Before split:  [5 @ $800]                cost = $4,000
  SPL: +45 shares, ratio = (5+45)/5 = 10
  After split:   [50 @ $80]                cost = $4,000  (preserved)

  Sell 3:        [47 @ $80]
  avg cost = $80.00                          matches Robinhood
```

Multi-lot example (buys at different prices before the split):

```
Multiple pre-split lots:
  Before split:  [3 @ $750, 2 @ $800]      cost = $2,250 + $1,600 = $3,850
  10:1 split:    [30 @ $75, 20 @ $80]       cost = $2,250 + $1,600 = $3,850

  FIFO order preserved.
  Per-lot cost preserved.
  Total cost preserved.
  Sells now consume $75 lots first (correctly reflecting older, cheaper purchases).
```

The key insight: each lot's total cost (`shares * price`) is an invariant through a split. The restructuring redistributes shares and per-share price while holding this invariant.

---

## Lessons Learned

**"Skip and handle later" in data pipelines is risky.** The SPL records were categorized as "log and skip" during initial development because they weren't cash transactions. But they were load-bearing for every downstream calculation. Skipped records can silently break invariants that surface much later.

**Total-level correctness does not guarantee record-level correctness.** The $0 lot approach produced correct totals ($4,000 across 50 shares). But FIFO operates at the lot level, not the total level. The distribution of cost across lots matters as much as the sum.

**Always validate against a source of truth.** Without cross-checking against Robinhood's reported cost basis ($54.98 vs. our $40.34), the $0 lot bug would have shipped unnoticed. The total cost was correct, shares were correct, and the portfolio appeared complete. Only the per-share average revealed the problem.

**Stock splits are state mutations, not transactions.** A buy or sell appends to the trade history. A split changes the meaning of all prior data. This requires restructuring existing records (lot adjustment) rather than appending new ones (a $0 trade). Treating mutations as transactions is a category error that produces subtle bugs.

---

## Impact

This fix correctly handles all stock splits in the portfolio, not just NVDA. The FIFO engine detects split trades by their source tag (`robinhood_csv_spl`), calculates the ratio from the share counts, and restructures every lot in the queue. The resulting cost basis calculations match the brokerage's own numbers.

### Files Changed

| File | Change |
|---|---|
| `src/marketmind/tools/importers.py` | SPL moved from `LOG_CODES` to `SHARE_ADDITION_CODES`. New `_parse_share_addition()` function imports split records as tagged trades. |
| `src/marketmind/db/repository.py` | `_recalculate_single_holding()` detects SPL trades and applies lot restructuring instead of appending a new lot. |
| `tests/test_portfolio.py` | Added `test_stock_split_adjusts_lots` covering the full split scenario with pre-split buys, split adjustment, post-split buys, and post-split sells. |
