# When a Stock Split Isn't a Split: Handling MRGS Transaction Codes

## The Problem

After fixing the SPL stock split issue ([001](001-fifo-stock-split.md)), SMCI was still showing an incorrect cost basis in `portfolio show`. SMCI had undergone a 10:1 stock split in 2024, the same corporate action as NVDA. But Robinhood encoded it as `MRGS` (merger/reorganization) instead of `SPL`.

Since `MRGS` was in `LOG_CODES`, the split shares were being skipped entirely. The FIFO engine never saw them.

---

## Why MRGS Was Being Skipped

The CSV importer categorized `MRGS` as a non-trade event:

```python
SHARE_ADDITION_CODES = {"SPL", "ACATI", "REC"}   # Imported
LOG_CODES = {"MRGS"}                               # Logged and skipped
```

This seemed reasonable. "Merger" can mean many things: cash acquisitions, stock-for-stock conversions, ticker changes, reverse mergers. None of these are straightforward share additions, so the initial decision was to log them for transparency and handle them later.

This was the same pattern as the original SPL bug: skipping records that turned out to be load-bearing. A web search confirmed SMCI had no actual merger in 2024. The only corporate action was a 10:1 stock split. Robinhood just used a different transaction code to record it.

---

## The Twist: MRGS Encodes Splits Differently Than SPL

Moving `MRGS` into `SHARE_ADDITION_CODES` was necessary but not sufficient. The two codes encode the same event with different quantity semantics.

**SPL encoding** (NVDA, 10:1 split, 5 pre-split shares):

```
SPL row: Quantity = 45          ← new shares ADDED

Ratio = (existing + new) / existing
      = (5 + 45) / 5
      = 10
```

**MRGS encoding** (SMCI, 10:1 split, 3 pre-split shares):

```
MRGS row 1: Quantity = 30       ← TOTAL post-split shares (the "receive" side)
MRGS row 2: Quantity = 3S       ← old shares surrendered (the "surrender" side)

Ratio = new_total / existing
      = 30 / 3
      = 10
```

Both arrive at ratio = 10. Both represent the same real-world event. But the input format is completely different.

If you fed MRGS quantities into the SPL formula, you would get:

```
Wrong: (existing + mrgs_quantity) / existing = (3 + 30) / 3 = 11
Right: mrgs_quantity / existing              = 30 / 3       = 10
```

A ratio of 11 instead of 10 means every lot's per-share cost would be divided by 11 instead of 10. For a $900 pre-split cost basis: `$900 / 11 = $81.82` instead of the correct `$900 / 10 = $90.00`. Close enough to look plausible, wrong enough to corrupt every downstream calculation.

### The "3S" Quantity

The surrender row had a quantity of `3S`, not `3`. The trailing "S" likely stands for "surrendered." `float("3S")` raises a `ValueError`, so this row was silently skipped by the parser's exception handler.

This turned out to be harmless: the surrender row is redundant information (we already know the existing shares from the lot queue). But relying on exception-based skipping is fragile. We added explicit handling to strip trailing letters before parsing:

```python
quantity_str = quantity_str.rstrip("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
```

---

## The Fix

Two changes in the FIFO engine. SPL and MRGS both trigger lot restructuring, but with different ratio formulas:

```python
# SPL: quantity = NEW shares added
if trade.source.endswith("_spl") and lots and trade.price_per_share == 0:
    existing_shares = sum(lot[0] for lot in lots)
    ratio = (existing_shares + trade.shares) / existing_shares
    for lot in lots:
        lot[1] /= ratio
        lot[0] *= ratio

# MRGS: quantity = TOTAL post-split shares
# Skip surrender rows (qty <= existing)
elif trade.source.endswith("_mrgs") and lots and trade.price_per_share == 0:
    existing_shares = sum(lot[0] for lot in lots)
    if trade.shares > existing_shares:
        ratio = trade.shares / existing_shares
        for lot in lots:
            lot[1] /= ratio
            lot[0] *= ratio
```

The surrender row is identified by its quantity being less than or equal to the existing shares. A split always produces more shares than existed before, so the receive row will always have a quantity greater than existing shares.

Worked example:

```
Before split:   [3 @ $900]                   cost = $2,700
MRGS surrender: 3 shares (skipped: 3 <= 3)
MRGS receive:   30 shares, ratio = 30/3 = 10
After split:    [30 @ $90]                   cost = $2,700  (preserved)

Buy 70 @ $40:   [30 @ $90, 70 @ $40]        cost = $2,700 + $2,800 = $5,500
Total: 100 shares, avg = $55.00
```

---

## Lessons Learned

**The same real-world event can have multiple data representations.** A stock split can appear as `SPL` or `MRGS` depending on the brokerage's internal classification. Robust pipelines handle the semantics (what happened), not just the labels (what the code says).

**Never assume transaction codes are self-describing.** `MRGS` does not mean "this company merged with another company." It means "a corporate reorganization event occurred," which includes splits, reverse splits, name changes, and actual mergers. The code name is a hint, not a specification.

**This is the second time "skip and handle later" produced a bug in the same pipeline.** Pattern recognition: if you are skipping transaction types in a financial data pipeline, you are probably accumulating hidden bugs proportional to the number of codes you skip.

---

## Relationship to 001

This is the same class of bug as the SPL issue ([001](001-fifo-stock-split.md)): a skipped transaction code caused incorrect FIFO calculations. But it has an additional layer of complexity. The data encoding was different, requiring a separate code path with its own ratio formula rather than just expanding the existing SPL handler.

The two bugs together establish a pattern. Corporate actions that change share counts (splits, mergers, reorganizations) are not optional records in the CSV. They are structural. Skipping any of them will produce silent, plausible-looking errors in cost basis calculations.

---

## Impact

The FIFO engine now handles two variants of stock split encoding:

| Code | Quantity Means | Ratio Formula | Source Tag |
|---|---|---|---|
| `SPL` | New shares added | `(existing + new) / existing` | `robinhood_csv_spl` |
| `MRGS` | Total post-split shares | `total / existing` | `robinhood_csv_mrgs` |

Both preserve the per-lot cost invariant (`shares * price = constant`). The surrender row in MRGS is detected and skipped by comparing its quantity to the existing share count.

### Files Changed

| File | Change |
|---|---|
| `src/marketmind/tools/importers.py` | MRGS moved from `LOG_CODES` to `SHARE_ADDITION_CODES`. Quantity parsing strips trailing letters for surrender rows ("3S" to "3"). |
| `src/marketmind/db/repository.py` | `_recalculate_single_holding()` gains a MRGS branch with `ratio = trade.shares / existing_shares`. Surrender rows (qty <= existing) are skipped. |
| `tests/test_portfolio.py` | Added `test_mrgs_stock_split_adjusts_lots` covering the full SMCI scenario: pre-split buy, surrender row skipped, receive row triggers restructuring, post-split buy, correct totals. |
