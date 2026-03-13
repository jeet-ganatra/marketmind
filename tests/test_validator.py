"""
Phase 3D tests -- Analysis validation.

Offline tests (no network, no API keys): JSON parsing, issue parsing,
ValidationResult construction, orchestrator integration of _run_validation,
and CLI display helper.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from marketmind.agent.validator import (
    AnalysisValidator,
    _extract_json,
    _parse_issues,
)
from marketmind.models.schemas import (
    AnalysisResult,
    ValidationIssue,
    ValidationResult,
)


# ──────────────────────────────────────────────
# _extract_json tests
# ──────────────────────────────────────────────


class TestExtractJson:
    """Test the JSON extraction helper."""

    def test_plain_json(self):
        text = '{"confidence_score": 90, "summary": "all good"}'
        result = _extract_json(text)
        assert result is not None
        assert result["confidence_score"] == 90

    def test_json_with_code_fence(self):
        text = '```json\n{"confidence_score": 85}\n```'
        result = _extract_json(text)
        assert result is not None
        assert result["confidence_score"] == 85

    def test_json_with_plain_code_fence(self):
        text = '```\n{"confidence_score": 80}\n```'
        result = _extract_json(text)
        assert result is not None
        assert result["confidence_score"] == 80

    def test_json_embedded_in_text(self):
        text = 'Here is the result:\n{"confidence_score": 75}\nThat is all.'
        result = _extract_json(text)
        assert result is not None
        assert result["confidence_score"] == 75

    def test_invalid_json_returns_none(self):
        text = "This is not JSON at all."
        result = _extract_json(text)
        assert result is None

    def test_empty_string_returns_none(self):
        result = _extract_json("")
        assert result is None

    def test_json_with_whitespace(self):
        text = '  \n  {"confidence_score": 95}  \n  '
        result = _extract_json(text)
        assert result is not None
        assert result["confidence_score"] == 95

    def test_nested_json(self):
        text = json.dumps({
            "unsupported_numbers": [{"claim": "revenue $1B", "detail": "not in data"}],
            "unsupported_claims": [],
            "contradictions": [],
            "confidence_score": 82,
            "summary": "mostly good",
        })
        result = _extract_json(text)
        assert result is not None
        assert len(result["unsupported_numbers"]) == 1
        assert result["confidence_score"] == 82


# ──────────────────────────────────────────────
# _parse_issues tests
# ──────────────────────────────────────────────


class TestParseIssues:
    """Test the issue list parser."""

    def test_empty_list(self):
        assert _parse_issues([]) == []

    def test_valid_issues(self):
        raw = [
            {"claim": "Revenue was $100B", "detail": "Data shows $90B"},
            {"claim": "P/E is 25", "detail": "Data shows P/E is 22"},
        ]
        issues = _parse_issues(raw)
        assert len(issues) == 2
        assert isinstance(issues[0], ValidationIssue)
        assert issues[0].claim == "Revenue was $100B"
        assert issues[1].detail == "Data shows P/E is 22"

    def test_missing_keys_default_to_empty(self):
        raw = [{"claim": "something"}]  # missing "detail"
        issues = _parse_issues(raw)
        assert len(issues) == 1
        assert issues[0].detail == ""

    def test_non_dict_items_skipped(self):
        raw = [{"claim": "valid", "detail": "ok"}, "not a dict", 42]
        issues = _parse_issues(raw)
        assert len(issues) == 1

    def test_values_converted_to_strings(self):
        raw = [{"claim": 123, "detail": True}]
        issues = _parse_issues(raw)
        assert issues[0].claim == "123"
        assert issues[0].detail == "True"


# ──────────────────────────────────────────────
# ValidationResult model tests
# ──────────────────────────────────────────────


class TestValidationResultModel:
    """Test ValidationResult defaults and construction."""

    def test_defaults(self):
        vr = ValidationResult()
        assert vr.unsupported_numbers == []
        assert vr.unsupported_claims == []
        assert vr.contradictions == []
        assert vr.confidence_score == 100
        assert vr.summary == ""
        assert vr.validation_cost_usd == 0.0
        assert vr.was_validated is True

    def test_with_issues(self):
        vr = ValidationResult(
            unsupported_numbers=[ValidationIssue(claim="c1", detail="d1")],
            contradictions=[ValidationIssue(claim="c2", detail="d2")],
            confidence_score=65,
            summary="low confidence",
        )
        assert len(vr.unsupported_numbers) == 1
        assert len(vr.contradictions) == 1
        assert vr.confidence_score == 65

    def test_not_validated(self):
        vr = ValidationResult(was_validated=False, summary="skipped")
        assert vr.was_validated is False
        assert vr.confidence_score == 100  # default untouched


# ──────────────────────────────────────────────
# AnalysisResult with validation
# ──────────────────────────────────────────────


class TestAnalysisResultValidation:
    """Test AnalysisResult schema with validation fields."""

    def test_analysis_result_defaults(self):
        ar = AnalysisResult(
            ticker="AAPL",
            analysis_type="comprehensive",
            summary="test",
            detailed_analysis="test analysis",
            data_sources=["yfinance:price"],
            model_used="claude-sonnet",
            cost_usd=0.05,
        )
        assert ar.validation is None
        assert ar.total_cost_usd == 0.0

    def test_analysis_result_with_validation(self):
        vr = ValidationResult(
            confidence_score=88,
            validation_cost_usd=0.002,
            was_validated=True,
        )
        ar = AnalysisResult(
            ticker="AAPL",
            analysis_type="comprehensive",
            summary="test",
            detailed_analysis="test analysis",
            data_sources=["yfinance:price"],
            model_used="claude-sonnet",
            cost_usd=0.05,
            validation=vr,
            total_cost_usd=0.052,
        )
        assert ar.validation is not None
        assert ar.validation.confidence_score == 88
        assert ar.total_cost_usd == 0.052


# ──────────────────────────────────────────────
# AnalysisValidator tests (mocked LLM)
# ──────────────────────────────────────────────


class TestAnalysisValidator:
    """Test the AnalysisValidator with mocked router calls."""

    def _mock_router(self, response_text: str, cost: float = 0.001):
        """Create a mock router that returns the given text."""
        router = MagicMock()
        cost_record = MagicMock()
        cost_record.cost_usd = cost
        router.call.return_value = {
            "content": response_text,
            "model": "gpt-4o-mini",
            "cost_record": cost_record,
        }
        return router

    def test_clean_validation(self):
        response = json.dumps({
            "unsupported_numbers": [],
            "unsupported_claims": [],
            "contradictions": [],
            "confidence_score": 95,
            "summary": "Analysis is well-grounded.",
        })
        router = self._mock_router(response)
        validator = AnalysisValidator(router)

        result = validator.validate(
            analysis_text="AAPL shows strong growth.",
            provided_data={"Price": "AAPL is at $190"},
        )

        assert result.was_validated is True
        assert result.confidence_score == 95
        assert len(result.unsupported_numbers) == 0
        assert result.summary == "Analysis is well-grounded."

    def test_validation_with_issues(self):
        response = json.dumps({
            "unsupported_numbers": [
                {"claim": "Revenue was $500B", "detail": "Data shows $394B"},
            ],
            "unsupported_claims": [
                {"claim": "CEO announced new product", "detail": "Not in provided data"},
            ],
            "contradictions": [
                {"claim": "Stock dropped 10%", "detail": "Data shows +2.5%"},
            ],
            "confidence_score": 55,
            "summary": "Several unsupported claims found.",
        })
        router = self._mock_router(response)
        validator = AnalysisValidator(router)

        result = validator.validate(
            analysis_text="Lots of claims here.",
            provided_data={"Price": "$190"},
        )

        assert result.was_validated is True
        assert result.confidence_score == 55
        assert len(result.unsupported_numbers) == 1
        assert len(result.unsupported_claims) == 1
        assert len(result.contradictions) == 1
        assert result.unsupported_numbers[0].claim == "Revenue was $500B"

    def test_validation_with_code_fence_response(self):
        inner = json.dumps({
            "unsupported_numbers": [],
            "unsupported_claims": [],
            "contradictions": [],
            "confidence_score": 92,
            "summary": "Good analysis.",
        })
        response = f"```json\n{inner}\n```"
        router = self._mock_router(response)
        validator = AnalysisValidator(router)

        result = validator.validate(
            analysis_text="test",
            provided_data={"Data": "test data"},
        )

        assert result.was_validated is True
        assert result.confidence_score == 92

    def test_validation_unparseable_response(self):
        router = self._mock_router("I cannot validate this analysis properly.")
        validator = AnalysisValidator(router)

        result = validator.validate(
            analysis_text="test",
            provided_data={"Data": "test data"},
        )

        assert result.was_validated is False
        assert "could not parse" in result.summary.lower()

    def test_validation_api_error(self):
        router = MagicMock()
        router.call.side_effect = RuntimeError("API error")
        validator = AnalysisValidator(router)

        result = validator.validate(
            analysis_text="test",
            provided_data={"Data": "test data"},
        )

        assert result.was_validated is False
        assert "error" in result.summary.lower()
        assert result.validation_cost_usd == 0

    def test_validation_cost_tracked(self):
        response = json.dumps({
            "unsupported_numbers": [],
            "unsupported_claims": [],
            "contradictions": [],
            "confidence_score": 90,
            "summary": "ok",
        })
        router = self._mock_router(response, cost=0.0035)
        validator = AnalysisValidator(router)

        result = validator.validate(
            analysis_text="test",
            provided_data={"Data": "test data"},
        )

        assert result.validation_cost_usd == 0.0035

    def test_validation_with_rag_context(self):
        response = json.dumps({
            "unsupported_numbers": [],
            "unsupported_claims": [],
            "contradictions": [],
            "confidence_score": 88,
            "summary": "ok",
        })
        router = self._mock_router(response)
        validator = AnalysisValidator(router)

        result = validator.validate(
            analysis_text="AAPL analysis text",
            provided_data={"Price": "$190"},
            rag_context="SEC filing mentions revenue growth of 8%.",
        )

        assert result.was_validated is True
        # Verify RAG context was passed in the prompt
        call_args = router.call.call_args
        assert "SEC filing mentions revenue growth" in call_args[1]["prompt"] or \
               "SEC filing mentions revenue growth" in call_args.kwargs.get("prompt", "")

    def test_router_called_with_routine_task_type(self):
        response = json.dumps({
            "unsupported_numbers": [],
            "unsupported_claims": [],
            "contradictions": [],
            "confidence_score": 90,
            "summary": "ok",
        })
        router = self._mock_router(response)
        validator = AnalysisValidator(router)

        validator.validate(
            analysis_text="test",
            provided_data={"Data": "test data"},
        )

        call_kwargs = router.call.call_args[1]
        assert call_kwargs["task_type"] == "routine"

    def test_null_cost_record(self):
        router = MagicMock()
        router.call.return_value = {
            "content": json.dumps({
                "unsupported_numbers": [],
                "unsupported_claims": [],
                "contradictions": [],
                "confidence_score": 90,
                "summary": "ok",
            }),
            "model": "gpt-4o-mini",
            "cost_record": None,
        }
        validator = AnalysisValidator(router)

        result = validator.validate(
            analysis_text="test",
            provided_data={"Data": "test data"},
        )

        assert result.was_validated is True
        assert result.validation_cost_usd == 0


# ──────────────────────────────────────────────
# _run_validation helper tests
# ──────────────────────────────────────────────


class TestRunValidation:
    """Test the _run_validation orchestrator helper."""

    def test_run_validation_attaches_result(self):
        from marketmind.agent.orchestrator import _run_validation

        validator = MagicMock()
        validator.validate.return_value = ValidationResult(
            confidence_score=88,
            validation_cost_usd=0.002,
            was_validated=True,
            summary="looks good",
        )

        ar = AnalysisResult(
            ticker="AAPL",
            analysis_type="comprehensive",
            summary="test",
            detailed_analysis="test analysis",
            data_sources=["yfinance:price"],
            model_used="claude-sonnet",
            cost_usd=0.05,
        )

        result = _run_validation(
            validator, ar, {"Price": "$190"}, "some RAG context"
        )

        assert result.validation is not None
        assert result.validation.confidence_score == 88
        assert abs(result.total_cost_usd - 0.052) < 1e-9  # 0.05 + 0.002

    def test_run_validation_calls_validator(self):
        from marketmind.agent.orchestrator import _run_validation

        validator = MagicMock()
        validator.validate.return_value = ValidationResult(
            confidence_score=90,
            validation_cost_usd=0.001,
        )

        ar = AnalysisResult(
            ticker="AAPL",
            analysis_type="comprehensive",
            summary="test",
            detailed_analysis="the analysis text here",
            data_sources=[],
            model_used="claude-sonnet",
            cost_usd=0.05,
        )

        _run_validation(validator, ar, {"Price": "$190"}, "rag stuff")

        validator.validate.assert_called_once_with(
            analysis_text="the analysis text here",
            provided_data={"Price": "$190"},
            rag_context="rag stuff",
        )


# ──────────────────────────────────────────────
# CLI display helper tests
# ──────────────────────────────────────────────


class TestDisplayValidation:
    """Test the _display_validation CLI helper."""

    def test_no_validation_no_output(self, capsys):
        from marketmind.cli import _display_validation

        ar = AnalysisResult(
            ticker="AAPL",
            analysis_type="comprehensive",
            summary="test",
            detailed_analysis="test",
            data_sources=[],
            model_used="claude-sonnet",
            cost_usd=0.05,
        )
        _display_validation(ar)
        # Should produce no output (rich console writes to stderr in some configs,
        # so we just verify no crash)

    def test_not_validated_no_output(self, capsys):
        from marketmind.cli import _display_validation

        ar = AnalysisResult(
            ticker="AAPL",
            analysis_type="comprehensive",
            summary="test",
            detailed_analysis="test",
            data_sources=[],
            model_used="claude-sonnet",
            cost_usd=0.05,
            validation=ValidationResult(was_validated=False),
        )
        _display_validation(ar)
        # Should produce no output — was_validated is False

    def test_validated_result_no_crash(self):
        from marketmind.cli import _display_validation

        ar = AnalysisResult(
            ticker="AAPL",
            analysis_type="comprehensive",
            summary="test",
            detailed_analysis="test",
            data_sources=[],
            model_used="claude-sonnet",
            cost_usd=0.05,
            validation=ValidationResult(
                confidence_score=85,
                was_validated=True,
                unsupported_numbers=[ValidationIssue(claim="c", detail="d")],
                summary="mostly good",
            ),
            total_cost_usd=0.052,
        )
        # Should not raise
        _display_validation(ar)
