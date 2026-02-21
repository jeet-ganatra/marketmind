"""
LLM Router — Routes queries to the appropriate model.

This is the core abstraction that lets MarketMind use multiple LLMs efficiently:
- Routine tasks (price lookups, formatting) → GPT-4o-mini (cheap)
- Deep analysis (stock analysis, buy/sell reasoning) → Claude Sonnet (powerful)

The router also:
- Enforces budget limits before making calls
- Tracks costs for every call
- Handles errors gracefully (if one provider is down, log and inform)
"""

from anthropic import Anthropic
from openai import OpenAI
from rich.console import Console

from marketmind.config import Settings
from marketmind.llm.cost_tracker import CostTracker

console = Console()


class LLMRouter:
    """
    Routes LLM calls to the appropriate provider based on task type.

    Usage:
        router = LLMRouter(settings)
        result = router.call(
            prompt="Analyze AAPL's current valuation",
            system_prompt="You are a stock market analyst...",
            task_type="analysis",  # or "routine"
        )
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cost_tracker = CostTracker(
            budget_limit=settings.monthly_budget_limit,
            data_dir=settings.data_dir,
        )

        # Initialize API clients
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        self.anthropic_client = Anthropic(api_key=settings.anthropic_api_key)

    def call(
        self,
        prompt: str,
        system_prompt: str = "",
        task_type: str = "routine",
        max_tokens: int | None = None,
    ) -> dict:
        """
        Make an LLM call, routed to the appropriate model.

        Args:
            prompt: The user/task prompt
            system_prompt: System instructions for the model
            task_type: "routine" (cheap model) or "analysis" (powerful model)
            max_tokens: Override the default max_tokens for this call

        Returns:
            dict with keys: "content", "model", "cost_record"
        """
        # Budget check — fail fast if we're over budget
        if not self.cost_tracker.check_budget():
            status = self.cost_tracker.get_budget_status()
            return {
                "content": (
                    f"⚠️ Monthly budget limit reached (${status['monthly_spend']:.2f} "
                    f"/ ${status['monthly_limit']:.2f}). "
                    "Update MONTHLY_BUDGET_LIMIT in .env to continue."
                ),
                "model": "none",
                "cost_record": None,
            }

        if task_type == "analysis":
            return self._call_anthropic(prompt, system_prompt, max_tokens=max_tokens)
        else:
            return self._call_openai(prompt, system_prompt)

    def _call_openai(self, prompt: str, system_prompt: str) -> dict:
        """Call OpenAI GPT-4o-mini for routine tasks."""
        model = self.settings.routine_model
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,  # Lower temp for factual/analytical tasks
            )

            content = response.choices[0].message.content or ""
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            cost_record = self.cost_tracker.record_usage(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                query_type="routine",
            )

            return {"content": content, "model": model, "cost_record": cost_record}

        except Exception as e:
            console.print(f"[red]OpenAI API error: {e}[/red]")
            return {"content": f"Error calling OpenAI: {e}", "model": model, "cost_record": None}

    def _call_anthropic(self, prompt: str, system_prompt: str, max_tokens: int | None = None) -> dict:
        """Call Anthropic Claude Sonnet for deep analysis."""
        model = self.settings.analysis_model
        try:
            kwargs = {
                "model": model,
                "max_tokens": max_tokens or 4096,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            response = self.anthropic_client.messages.create(**kwargs)

            content = response.content[0].text if response.content else ""
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            cost_record = self.cost_tracker.record_usage(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                query_type="analysis",
            )

            return {"content": content, "model": model, "cost_record": cost_record}

        except Exception as e:
            console.print(f"[red]Anthropic API error: {e}[/red]")
            return {"content": f"Error calling Anthropic: {e}", "model": model, "cost_record": None}
