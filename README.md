# MarketMind

An AI-powered stock market decision-support agent that combines real-time market data, SEC filings, financial news, and LLM reasoning to help investors make data-driven decisions.

## What It Does

- **Stock Analysis**: Deep-dive analysis of any US equity using price data, fundamentals, analyst ratings, and price history — powered by Claude Sonnet
- **Portfolio Tracking**: Import trades from Robinhood CSV or add manually, with FIFO cost basis tracking and live P&L
- **Document Ingestion**: Ingest SEC 10-K/10-Q filings and financial news into a local vector database for RAG-enhanced analysis
- **RAG-Powered Analysis**: Analyses automatically reference ingested SEC filings and news for grounded, citation-backed insights
- **Analysis Validation**: Every analysis is fact-checked by a second LLM pass that flags unsupported numbers, claims, and contradictions
- **Smart Cost Management**: Tiered LLM routing — GPT-4o-mini for routine tasks, Claude Sonnet for deep analysis, with monthly budget tracking
- **Educational Mode**: Any analysis command accepts `--explain` for beginner-friendly explanations of financial metrics

## Architecture

```
CLI (Typer)
  └── Agent Orchestrator
        ├── LLM Router ──→ GPT-4o-mini (routine) / Claude Sonnet (analysis)
        ├── Tools
        │     ├── yfinance (price, fundamentals, history, analysts)
        │     ├── SEC EDGAR (10-K, 10-Q filings)
        │     └── RSS feeds (Yahoo Finance, Google News, Seeking Alpha)
        ├── RAG Retriever ──→ ChromaDB (filings + news vectors)
        ├── Analysis Validator ──→ GPT-4o-mini (fact-checking)
        └── SQLite (portfolios, trades, holdings, watchlist, cache)
```

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| LLMs | OpenAI GPT-4o-mini (routine) + Anthropic Claude Sonnet (analysis) |
| Market Data | yfinance, SEC EDGAR |
| Vector Store | ChromaDB + sentence-transformers |
| Database | SQLite (WAL mode) |
| CLI | Typer + Rich |
| Package Manager | Poetry |

## Quick Start

### Prerequisites
- Python 3.12+
- [Poetry](https://python-poetry.org/docs/#installation) package manager
- API keys for OpenAI and Anthropic

### Setup

```bash
# Clone the repo
git clone git@github.com:<YOUR_USERNAME>/marketmind.git
cd marketmind

# Install dependencies
poetry install

# Copy environment template and add your API keys
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY and ANTHROPIC_API_KEY

# Verify setup
poetry run marketmind check-setup

# Run your first analysis
poetry run marketmind analyze AAPL
```

## Commands

### Stock Analysis

```bash
# Deep analysis (Claude Sonnet)
marketmind analyze AAPL

# Educational mode with metric explanations
marketmind analyze AAPL --explain

# Skip the validation step
marketmind analyze AAPL --no-validate

# Quick price check (GPT-4o-mini, cheap)
marketmind price TSLA

# Learn a financial concept
marketmind learn "P/E ratio"
```

### Portfolio Management

```bash
# Set active user
marketmind user set jay
marketmind user show

# Add trades manually
marketmind portfolio add AAPL 10 198.50 --date 2025-01-15
marketmind portfolio add MSFT 5 420.00

# Import from Robinhood CSV
marketmind portfolio import ~/Downloads/robinhood_activity.csv

# View portfolio with live prices and P&L
marketmind portfolio show

# View trade history
marketmind portfolio history
marketmind portfolio history --ticker AAPL

# Sell (full or partial)
marketmind portfolio remove AAPL              # sell all at market price
marketmind portfolio remove AAPL --shares 5   # partial sell
```

### Portfolio Analysis

```bash
# AI analysis of full portfolio
marketmind portfolio analyze

# Analyze a specific holding in portfolio context
marketmind portfolio analyze AAPL --explain

# Evaluate a stock as a potential buy
marketmind evaluate NVDA
marketmind evaluate NVDA --explain
```

### Document Ingestion (RAG)

```bash
# Ingest SEC filings for a ticker
marketmind ingest filings AAPL --type both --quarters 8

# Ingest filings for all portfolio/watchlist tickers
marketmind ingest filings all

# Ingest financial news
marketmind ingest news all                    # all portfolio/watchlist tickers
marketmind ingest news --tickers AAPL,NVDA    # specific tickers

# Ingest everything (filings + news) for all tickers
marketmind ingest all

# Check ingestion status
marketmind ingest status
```

### Watchlist

```bash
marketmind watchlist add TSLA --reason "Watching for pullback"
marketmind watchlist show
marketmind watchlist remove TSLA
```

### Utilities

```bash
# Check API spending
marketmind cost

# Verify API keys and dependencies
marketmind check-setup
```

## Project Structure

```
marketmind/
├── src/marketmind/
│   ├── cli.py                 # CLI entry point (Typer)
│   ├── config.py              # Settings & API key management
│   ├── agent/
│   │   ├── orchestrator.py    # Main agent logic (4 analysis methods)
│   │   ├── rag.py             # RAG retrieval & context formatting
│   │   └── validator.py       # Analysis fact-checking via LLM
│   ├── llm/
│   │   ├── router.py          # Tiered LLM routing (routine vs analysis)
│   │   └── cost_tracker.py    # Monthly budget tracking
│   ├── tools/
│   │   ├── price.py           # Stock price data (yfinance)
│   │   ├── fundamentals.py    # P/E, EPS, financials (yfinance)
│   │   ├── sec_filings.py     # SEC EDGAR filing ingestion
│   │   ├── news_feeds.py      # RSS feed ingestion (per-ticker + general)
│   │   ├── importers.py       # Robinhood CSV parser
│   │   └── user_manager.py    # Active user file persistence
│   ├── db/
│   │   ├── database.py        # SQLite connection & schema
│   │   ├── repository.py      # Portfolio queries & FIFO cost basis
│   │   └── vector_store.py    # ChromaDB wrapper for RAG
│   ├── models/
│   │   └── schemas.py         # Pydantic data models
│   └── prompts/
│       └── templates.py       # LLM prompt templates
├── tests/
│   ├── test_tools.py          # Market data & cost tracker tests
│   ├── test_portfolio.py      # Database, FIFO, CSV import tests
│   ├── test_ingestion.py      # Vector store & filing ingestion tests
│   ├── test_rag.py            # RAG retrieval & formatting tests
│   └── test_validator.py      # Analysis validation tests
├── docs/engineering-decisions/ # Technical decision records
├── pyproject.toml
└── .env.example
```

## How Analysis Validation Works

Every analysis goes through a two-step process:

1. **Analysis** (Claude Sonnet): Generates the deep analysis from market data + RAG context
2. **Validation** (GPT-4o-mini): Cross-checks the analysis against the raw data that was provided, flagging:
   - **Unsupported numbers**: Statistics not found in the source data
   - **Unsupported claims**: Factual claims not backed by provided context
   - **Contradictions**: Statements that directly conflict with the data

Each analysis receives a confidence score (0-100). Use `--no-validate` to skip this step.

## Engineering Decisions

Significant technical decisions and interesting bugs are documented in [`docs/engineering-decisions/`](docs/engineering-decisions/):

- [FIFO Cost Basis and Stock Splits](docs/engineering-decisions/001-fifo-stock-split.md) — How a skipped CSV record caused the largest portfolio position to silently vanish, and why the obvious fix was subtly wrong
- [When a Stock Split Isn't a Split: MRGS Codes](docs/engineering-decisions/002-mrgs-merger-splits.md) — Same bug, different transaction code, and why the fix required its own ratio formula

## Development

```bash
# Run all tests (some require network for yfinance/EDGAR)
poetry run pytest -v

# Run offline-only tests
poetry run pytest tests/test_portfolio.py tests/test_validator.py -v
```

## Development Roadmap

- [x] Phase 1: CLI + single-stock analysis with tiered LLM routing
- [x] Phase 2: Portfolio tracking, Robinhood import, FIFO cost basis
- [x] Phase 3A: Document ingestion pipeline (SEC filings + news → ChromaDB)
- [x] Phase 3B: RAG retrieval layer (vector search → prompt context)
- [x] Phase 3D: Analysis validation (LLM fact-checking)
- [ ] Phase 4: Advanced agent (multi-step reasoning, stock screener)
- [ ] Phase 5: Web interface & deployment

## Privacy

- All data is stored locally (SQLite + ChromaDB) and never shared
- LLM API calls contain only stock-level data — no personal identifiers are sent
- Portfolio data never leaves your machine except as anonymized context in LLM prompts

## License

MIT
