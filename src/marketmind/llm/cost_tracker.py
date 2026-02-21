"""
LLM Cost Tracker.

Tracks every API call's token usage and cost. Enforces monthly budget limits.
Persists cost data to a local JSON file so it survives restarts.

Why this matters:
- LLM API costs can surprise you. At $15/million output tokens for Sonnet,
  a single verbose analysis could cost $0.05-0.10. That adds up.
- This module ensures you ALWAYS know your spend and never accidentally
  blow past your budget.
"""

import json
from datetime import datetime
from pathlib import Path

from marketmind.models.schemas import CostRecord

# Pricing per million tokens (as of early 2026 — update if prices change)
# Source: https://openai.com/pricing and https://docs.anthropic.com/en/docs/about-claude/models
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
}


class CostTracker:
    """
    Tracks LLM API costs and enforces budget limits.

    Usage:
        tracker = CostTracker(budget_limit=30.0, data_dir=Path("./data"))
        cost = tracker.record_usage("gpt-4o-mini", input_tokens=500, output_tokens=200, query_type="routine")
        tracker.get_monthly_spend()  # Returns total spend this month
    """

    def __init__(self, budget_limit: float, data_dir: Path) -> None:
        self.budget_limit = budget_limit
        self.data_dir = data_dir
        self.cost_file = data_dir / "costs.json"
        self._ensure_data_dir()

    def _ensure_data_dir(self) -> None:
        """Create data directory if it doesn't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _load_records(self) -> list[dict]:
        """Load cost records from disk."""
        if not self.cost_file.exists():
            return []
        with open(self.cost_file) as f:
            return json.load(f)

    def _save_records(self, records: list[dict]) -> None:
        """Save cost records to disk."""
        with open(self.cost_file, "w") as f:
            json.dump(records, f, indent=2, default=str)

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate the cost of an API call.

        Returns cost in USD. If model isn't in pricing table, returns 0 with a warning.
        """
        pricing = MODEL_PRICING.get(model)
        if not pricing:
            # Unknown model — don't block, but warn
            return 0.0

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return round(input_cost + output_cost, 6)

    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        query_type: str,
    ) -> CostRecord:
        """
        Record an API call's cost. Returns the CostRecord for display.
        """
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        record = CostRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            query_type=query_type,
        )

        records = self._load_records()
        records.append(record.model_dump())
        self._save_records(records)

        return record

    def get_monthly_spend(self) -> float:
        """Get total spend for the current month."""
        records = self._load_records()
        now = datetime.now()
        monthly_total = 0.0

        for record in records:
            record_time = datetime.fromisoformat(str(record["timestamp"]))
            if record_time.year == now.year and record_time.month == now.month:
                monthly_total += record["cost_usd"]

        return round(monthly_total, 4)

    def get_budget_status(self) -> dict:
        """
        Get current budget status.
        Returns dict with spend, limit, remaining, and whether we're over budget.
        """
        spend = self.get_monthly_spend()
        return {
            "monthly_spend": spend,
            "monthly_limit": self.budget_limit,
            "remaining": round(self.budget_limit - spend, 4),
            "over_budget": spend >= self.budget_limit,
            "usage_percent": round((spend / self.budget_limit) * 100, 1) if self.budget_limit > 0 else 0,
        }

    def check_budget(self) -> bool:
        """
        Returns True if we're within budget, False if we've exceeded it.
        Call this BEFORE making an API call.
        """
        return self.get_monthly_spend() < self.budget_limit
