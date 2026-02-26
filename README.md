# MarketMind 🧠📈

An AI-powered stock market decision-support agent that helps investors make data-driven decisions by combining real-time market data, fundamental analysis, and LLM reasoning.

## What It Does

- **Stock Analysis**: Deep-dive analysis of any US equity using fundamentals, analyst ratings, and financial data
- **Portfolio Tracking**: Import your holdings, track performance, and get portfolio-level insights
- **Intelligent Research**: RAG-powered Q&A over earnings transcripts and SEC filings
- **Shared Knowledge Base**: Multi-user system where analyses compound — run an analysis once, benefit everyone
- **Smart Cost Management**: Tiered LLM routing (cheap model for lookups, powerful model for analysis)

## Architecture

```
CLI Interface → FastAPI → Agent Orchestrator → LLM (GPT-4o-mini / Claude Sonnet)
                              ↓
                        Tool Registry
                    (market data, fundamentals,
                     news, SEC filings)
                              ↓
                     PostgreSQL + pgvector
                   (portfolios, research, cache)
```

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Agent Framework | LangChain + LangGraph |
| LLMs | OpenAI GPT-4o-mini (routine) + Anthropic Claude Sonnet (analysis) |
| Data | yfinance, Alpha Vantage, SEC EDGAR, Finnhub |
| Database | PostgreSQL + pgvector |
| Cache | Redis |
| API | FastAPI |
| CLI | Typer |
| Cloud | AWS (ECS Fargate, RDS, ElastiCache) |
| CI/CD | GitHub Actions |

## Quick Start

### Prerequisites
- Python 3.12+
- [Poetry](https://python-poetry.org/docs/#installation) package manager
- API keys for OpenAI, Anthropic, Alpha Vantage, Finnhub

### Setup

```bash
# Clone the repo
git clone git@github.com:<YOUR_USERNAME>/marketmind.git
cd marketmind

# Install dependencies
poetry install

# Copy environment template and add your API keys
cp .env.example .env
# Edit .env with your API keys

# Run the CLI
poetry run marketmind analyze AAPL
```

## Project Structure

```
marketmind/
├── src/marketmind/
│   ├── cli.py              # CLI entry point (Typer)
│   ├── config.py           # Settings & API key management
│   ├── agent/
│   │   ├── orchestrator.py # Main agent logic
│   │   └── router.py       # Routes queries to appropriate LLM
│   ├── llm/
│   │   ├── openai_client.py
│   │   ├── anthropic_client.py
│   │   └── cost_tracker.py
│   ├── tools/
│   │   ├── price.py        # Stock price data
│   │   ├── fundamentals.py # P/E, EPS, financials
│   │   ├── news.py         # News headlines
│   │   └── analyst.py      # Analyst ratings
│   └── models/
│       └── schemas.py      # Pydantic data models
├── tests/
├── pyproject.toml
└── .env.example
```

## Engineering Decisions

We document significant technical decisions and interesting bugs in [`docs/engineering-decisions/`](docs/engineering-decisions/):

- [FIFO Cost Basis and Stock Splits](docs/engineering-decisions/001-fifo-stock-split.md) — How a skipped CSV record caused the largest portfolio position to silently vanish, and why the obvious fix was subtly wrong
- [When a Stock Split Isn't a Split: MRGS Codes](docs/engineering-decisions/002-mrgs-merger-splits.md) — Same bug, different transaction code, and why the fix required its own ratio formula

## Development Roadmap

- [x] Phase 1: CLI + basic stock analysis with LLM
- [x] Phase 2: Portfolio tracking & multi-user support
- [ ] Phase 3: RAG over earnings transcripts & SEC filings
- [ ] Phase 4: Advanced agent (multi-step reasoning, stock screener)
- [ ] Phase 5: AWS deployment
- [ ] Phase 6: Web interface & real-time data

## Privacy

- Portfolio data is stored locally and never shared
- LLM API calls are anonymized — no personal identifiers are sent
- Shared analyses contain only stock-level data, never user portfolio details

## License

MIT
