"""
Application configuration.

Loads settings from .env file and environment variables.
Uses pydantic-settings for validation and type safety.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All application settings, loaded from .env file and environment variables.

    Why pydantic-settings?
    - Validates that required keys are present at startup (not at first API call)
    - Type coercion (e.g., MONTHLY_BUDGET_LIMIT string → float)
    - Centralized config — every setting lives in one place
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Don't fail on unexpected env vars
    )

    # LLM API Keys
    openai_api_key: str = Field(description="OpenAI API key for GPT-4o-mini")
    anthropic_api_key: str = Field(description="Anthropic API key for Claude Sonnet")

    # Market Data API Keys
    alpha_vantage_api_key: str = Field(default="", description="Alpha Vantage API key")
    finnhub_api_key: str = Field(default="", description="Finnhub API key")

    # Model Configuration
    routine_model: str = Field(
        default="gpt-4o-mini",
        description="Model for routine tasks (cheap, fast)",
    )
    analysis_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Model for deep analysis (powerful, more expensive)",
    )

    # Cost Tracking
    monthly_budget_limit: float = Field(
        default=30.0,
        description="Monthly spending limit in USD",
    )

    # Application Settings
    log_level: str = Field(default="INFO")
    data_dir: Path = Field(default=Path("./data"))

    def validate_keys(self) -> list[str]:
        """
        Check which API keys are configured.
        Returns a list of warnings for missing optional keys.
        """
        warnings = []
        if not self.openai_api_key or self.openai_api_key.startswith("sk-your"):
            warnings.append("OpenAI API key not configured — routine model will not work")
        if not self.anthropic_api_key or self.anthropic_api_key.startswith("sk-ant-your"):
            warnings.append("Anthropic API key not configured — analysis model will not work")
        if not self.alpha_vantage_api_key or "your" in self.alpha_vantage_api_key:
            warnings.append("Alpha Vantage API key not configured — some data sources unavailable")
        if not self.finnhub_api_key or "your" in self.finnhub_api_key:
            warnings.append("Finnhub API key not configured — news/analyst data unavailable")
        return warnings


def get_settings() -> Settings:
    """Load and return application settings."""
    return Settings()
