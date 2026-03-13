"""
Analysis Validator: checks LLM output for hallucinated claims.

Uses GPT-4o-mini (cheap model) to cross-check the analysis against the
raw data and RAG context that was provided to the analysis model.
Flags unsupported numbers, unsupported claims, and contradictions.
"""

from __future__ import annotations

import json
import re

from rich.console import Console

from marketmind.llm.router import LLMRouter
from marketmind.models.schemas import ValidationIssue, ValidationResult

console = Console()

VALIDATION_SYSTEM_PROMPT = """You are a fact-checker for financial analysis. You verify \
that analysis claims are supported by the provided data. You are precise and objective. \
Only flag genuine issues, not general financial knowledge."""

VALIDATION_PROMPT = """You are a fact-checker for financial analysis. You will be given:
1. An analysis that was written about a stock
2. The raw data that was available when the analysis was written
3. Additional research context from SEC filings and news (if any)

Your job is to check the analysis for accuracy. Specifically:

1. UNSUPPORTED NUMBERS: List any specific numbers, percentages, dollar amounts,
   or statistics in the analysis that do NOT appear in the provided data or
   research context. Be precise about which number and where it appears.

2. UNSUPPORTED CLAIMS: List any factual claims about the company (business
   description, events, strategies, problems) that are NOT supported by the
   provided data or research context. General financial knowledge
   (e.g., "P/E above 20 is considered high") is acceptable and should NOT
   be flagged.

3. CONTRADICTIONS: List any statements in the analysis that directly
   contradict the provided data.

4. CONFIDENCE SCORE: Rate 0-100 how well-grounded the analysis is in the
   provided data. 90-100 means almost everything is supported. 70-89 means
   mostly supported with some unsupported claims. Below 70 means significant
   unsupported content.

Respond in this exact JSON format:
{{
    "unsupported_numbers": [
        {{"claim": "the specific claim", "detail": "why it's unsupported"}}
    ],
    "unsupported_claims": [
        {{"claim": "the specific claim", "detail": "why it's unsupported"}}
    ],
    "contradictions": [
        {{"claim": "the specific claim", "detail": "what the data actually says"}}
    ],
    "confidence_score": 85,
    "summary": "Brief overall assessment"
}}

If the analysis is well-grounded and you find no issues, return empty lists
and a high confidence score.

---

## Analysis to Validate:
{analysis_text}

## Raw Data Provided to the Analyst:
{provided_data}

## Research Context (SEC filings, news):
{rag_context}
"""


def _extract_json(text: str) -> dict | None:
    """
    Extract JSON from LLM response, handling markdown code fences.

    GPT-4o-mini sometimes wraps JSON in ```json ... ``` blocks.
    """
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try to find a JSON object anywhere in the text
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    return None


def _parse_issues(raw_list: list) -> list[ValidationIssue]:
    """Convert a list of raw dicts into ValidationIssue objects."""
    issues = []
    for item in raw_list:
        if isinstance(item, dict):
            issues.append(ValidationIssue(
                claim=str(item.get("claim", "")),
                detail=str(item.get("detail", "")),
            ))
    return issues


class AnalysisValidator:
    """
    Validates LLM analysis output against the data that was provided.

    Uses GPT-4o-mini (routed as "routine" task) to cross-check claims.
    """

    def __init__(self, router: LLMRouter) -> None:
        self.router = router

    def validate(
        self,
        analysis_text: str,
        provided_data: dict,
        rag_context: str = "",
    ) -> ValidationResult:
        """
        Validate an analysis against its source data.

        Args:
            analysis_text: The LLM-generated analysis to check.
            provided_data: Dict of raw data that was provided to the analyst
                           (e.g., price_data, fundamental_data, history_data, analyst_data).
            rag_context: The RAG context string included in the prompt, if any.

        Returns:
            ValidationResult with issues found and confidence score.
            If validation fails (API error, parse error), returns a result
            with was_validated=False so the analysis is still shown.
        """
        # Format provided data into readable text
        data_text = "\n\n".join(
            f"### {key}\n{value}" for key, value in provided_data.items() if value
        )

        rag_text = rag_context if rag_context else "No additional research context was provided."

        prompt = VALIDATION_PROMPT.format(
            analysis_text=analysis_text,
            provided_data=data_text,
            rag_context=rag_text,
        )

        try:
            result = self.router.call(
                prompt=prompt,
                system_prompt=VALIDATION_SYSTEM_PROMPT,
                task_type="routine",
            )

            validation_cost = (
                result["cost_record"].cost_usd if result["cost_record"] else 0
            )

            # Parse the JSON response
            parsed = _extract_json(result["content"])
            if parsed is None:
                console.print("  [dim]Validation response was not valid JSON, skipping.[/dim]")
                return ValidationResult(
                    was_validated=False,
                    summary="Validation skipped: could not parse response.",
                    validation_cost_usd=validation_cost,
                )

            return ValidationResult(
                unsupported_numbers=_parse_issues(parsed.get("unsupported_numbers", [])),
                unsupported_claims=_parse_issues(parsed.get("unsupported_claims", [])),
                contradictions=_parse_issues(parsed.get("contradictions", [])),
                confidence_score=int(parsed.get("confidence_score", 100)),
                summary=str(parsed.get("summary", "")),
                validation_cost_usd=validation_cost,
                was_validated=True,
            )

        except Exception as e:
            console.print(f"  [dim]Validation skipped due to error: {e}[/dim]")
            return ValidationResult(
                was_validated=False,
                summary=f"Validation skipped due to error: {e}",
                validation_cost_usd=0,
            )
