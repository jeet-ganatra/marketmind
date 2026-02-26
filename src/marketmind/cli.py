"""
MarketMind CLI — The primary user interface.

Commands:
  marketmind analyze <TICKER>             — Deep analysis using Claude Sonnet
  marketmind analyze <TICKER> --explain   — Educational analysis with metric explanations
  marketmind price <TICKER>               — Quick price check using GPT-4o-mini
  marketmind learn <TOPIC>                — Learn a financial concept using GPT-4o-mini
  marketmind cost                         — Show API spending for the current month
  marketmind check-setup                  — Verify all API keys and dependencies

  marketmind user set <USERNAME>          — Set active user profile
  marketmind user show                    — Show active user profile

  marketmind portfolio show               — Show all holdings with live prices
  marketmind portfolio add <TICKER> <SHARES> <PRICE>  — Record a buy trade
  marketmind portfolio remove <TICKER>    — Record a sell trade (full or partial)
  marketmind portfolio import <CSV_FILE>  — Import trades from Robinhood CSV
  marketmind portfolio history            — Show trade history
  marketmind portfolio analyze [TICKER]   — AI analysis of portfolio or single holding

  marketmind watchlist add <TICKER>       — Add ticker to watchlist
  marketmind watchlist show               — Show watchlist
  marketmind watchlist remove <TICKER>    — Remove ticker from watchlist

  marketmind evaluate <TICKER>            — Evaluate a stock as a potential buy

Usage:
  poetry run marketmind analyze AAPL
  poetry run marketmind portfolio show
  poetry run marketmind evaluate NVDA --explain
"""

from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from marketmind.config import get_settings

app = typer.Typer(
    name="marketmind",
    help="AI-powered stock market decision-support agent",
    no_args_is_help=True,
)
console = Console()

# ──────────────────────────────────────────────
# Sub-apps
# ──────────────────────────────────────────────

user_app = typer.Typer(help="Manage user profiles")
portfolio_app = typer.Typer(help="Portfolio tracking and analysis")
watchlist_app = typer.Typer(help="Manage your watchlist")

app.add_typer(user_app, name="user")
app.add_typer(portfolio_app, name="portfolio")
app.add_typer(watchlist_app, name="watchlist")


# ──────────────────────────────────────────────
# Helper: initialize DB + resolve active user
# ──────────────────────────────────────────────


def _get_db_and_user():
    """
    Returns (Database, PortfolioRepository, User) tuple.

    Initializes the database, resolves the active username,
    and creates the user if needed.
    """
    from marketmind.db.database import Database
    from marketmind.db.repository import PortfolioRepository
    from marketmind.tools.user_manager import get_active_username

    settings = get_settings()
    db = Database(data_dir=settings.data_dir)
    db.initialize()
    repo = PortfolioRepository(db)
    username = get_active_username(settings.data_dir)
    user = repo.get_or_create_user(username)
    return db, repo, user


# ──────────────────────────────────────────────
# User commands
# ──────────────────────────────────────────────


@user_app.command("set")
def user_set(
    username: str = typer.Argument(help="Username to set as active"),
) -> None:
    """Set the active user profile."""
    from marketmind.tools.user_manager import set_active_username

    settings = get_settings()
    set_active_username(settings.data_dir, username)

    # Ensure user exists in DB
    from marketmind.db.database import Database
    from marketmind.db.repository import PortfolioRepository

    db = Database(data_dir=settings.data_dir)
    db.initialize()
    repo = PortfolioRepository(db)
    repo.get_or_create_user(username)
    db.close()

    console.print(f"\n[green]Active user set to:[/green] [bold]{username}[/bold]\n")


@user_app.command("show")
def user_show() -> None:
    """Show the current active user profile."""
    from marketmind.tools.user_manager import get_active_username

    settings = get_settings()
    username = get_active_username(settings.data_dir)
    console.print(f"\n[bold]Active user:[/bold] {username}\n")


# ──────────────────────────────────────────────
# Portfolio commands
# ──────────────────────────────────────────────


@portfolio_app.command("show")
def portfolio_show() -> None:
    """Show all holdings with live prices, P&L, and day change."""
    db, repo, user = _get_db_and_user()
    holdings = repo.get_holdings(user.id)

    if not holdings:
        console.print("\n[yellow]No holdings found.[/yellow] Use 'portfolio add' or 'portfolio import' to get started.\n")
        db.close()
        return

    from marketmind.tools.price import get_stock_price

    table = Table(title=f"Portfolio — {user.username}", show_lines=True)
    table.add_column("Ticker", style="bold cyan")
    table.add_column("Shares", justify="right")
    table.add_column("Avg Cost", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Value", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("P&L %", justify="right")
    table.add_column("Day %", justify="right")

    total_value = 0.0
    total_cost = 0.0

    for h in holdings:
        price = get_stock_price(h.ticker)
        current_price = price.current_price if price else 0.0
        day_change = price.change_percent if price else 0.0

        market_value = h.shares * current_price
        pnl = market_value - h.total_cost_basis
        pnl_pct = (pnl / h.total_cost_basis * 100) if h.total_cost_basis > 0 else 0

        total_value += market_value
        total_cost += h.total_cost_basis

        pnl_color = "green" if pnl >= 0 else "red"
        day_color = "green" if day_change >= 0 else "red"

        table.add_row(
            h.ticker,
            f"{h.shares:.4f}",
            f"${h.avg_cost_basis:.2f}",
            f"${current_price:.2f}",
            f"${market_value:,.2f}",
            f"[{pnl_color}]${pnl:,.2f}[/{pnl_color}]",
            f"[{pnl_color}]{pnl_pct:+.2f}%[/{pnl_color}]",
            f"[{day_color}]{day_change:+.2f}%[/{day_color}]",
        )

    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
    total_color = "green" if total_pnl >= 0 else "red"

    table.add_row(
        "[bold]TOTAL[/bold]",
        "",
        f"${total_cost:,.2f}",
        "",
        f"[bold]${total_value:,.2f}[/bold]",
        f"[bold {total_color}]${total_pnl:,.2f}[/bold {total_color}]",
        f"[bold {total_color}]{total_pnl_pct:+.2f}%[/bold {total_color}]",
        "",
    )

    console.print()
    console.print(table)
    console.print()
    db.close()


@portfolio_app.command("add")
def portfolio_add(
    ticker: str = typer.Argument(help="Stock ticker symbol"),
    shares: float = typer.Argument(help="Number of shares"),
    price: float = typer.Argument(help="Price per share"),
    date: str = typer.Option(None, "--date", help="Trade date (YYYY-MM-DD). Defaults to today."),
) -> None:
    """Record a buy trade for a stock."""
    from marketmind.models.schemas import Trade

    db, repo, user = _get_db_and_user()
    trade_date = datetime.strptime(date, "%Y-%m-%d") if date else datetime.now()

    trade = Trade(
        user_id=user.id,
        ticker=ticker.upper(),
        trade_type="buy",
        shares=shares,
        price_per_share=price,
        total_amount=shares * price,
        trade_date=trade_date,
        source="manual",
    )

    repo.add_trade(trade)
    repo.recalculate_holdings(user.id)

    console.print(
        f"\n[green]Bought {shares} shares of {ticker.upper()} "
        f"@ ${price:.2f} (${shares * price:,.2f})[/green]\n"
    )
    db.close()


@portfolio_app.command("remove")
def portfolio_remove(
    ticker: str = typer.Argument(help="Stock ticker symbol to sell"),
    shares: float = typer.Option(None, "--shares", help="Number of shares to sell. Defaults to full position."),
    price: float = typer.Option(None, "--price", help="Sell price per share. Defaults to current market price."),
) -> None:
    """Record a sell trade (full or partial position)."""
    from marketmind.models.schemas import Trade

    db, repo, user = _get_db_and_user()
    ticker = ticker.upper()

    holding = repo.get_holding(user.id, ticker)
    if not holding:
        console.print(f"\n[red]No holding found for {ticker}.[/red]\n")
        db.close()
        raise typer.Exit(1)

    sell_shares = shares if shares is not None else holding.shares

    if sell_shares > holding.shares + 0.0001:
        console.print(
            f"\n[red]Cannot sell {sell_shares} shares — you only hold {holding.shares:.4f} shares of {ticker}.[/red]\n"
        )
        db.close()
        raise typer.Exit(1)

    if price is None:
        from marketmind.tools.price import get_stock_price

        live = get_stock_price(ticker)
        if live:
            sell_price = live.current_price
            console.print(f"  Using current market price: ${sell_price:.2f}")
        else:
            console.print(f"\n[red]Could not fetch price for {ticker}. Use --price to specify.[/red]\n")
            db.close()
            raise typer.Exit(1)
    else:
        sell_price = price

    trade = Trade(
        user_id=user.id,
        ticker=ticker,
        trade_type="sell",
        shares=sell_shares,
        price_per_share=sell_price,
        total_amount=sell_shares * sell_price,
        trade_date=datetime.now(),
        source="manual",
    )

    repo.add_trade(trade)
    repo.recalculate_holdings(user.id)

    console.print(
        f"\n[green]Sold {sell_shares:.4f} shares of {ticker} "
        f"@ ${sell_price:.2f} (${sell_shares * sell_price:,.2f})[/green]\n"
    )
    db.close()


@portfolio_app.command("import")
def portfolio_import(
    csv_file: Path = typer.Argument(help="Path to Robinhood CSV file", exists=True),
) -> None:
    """Import trades from a Robinhood activity CSV export."""
    from marketmind.tools.importers import parse_robinhood_csv

    db, repo, user = _get_db_and_user()

    console.print(f"\n[bold blue]Importing trades from {csv_file.name}...[/bold blue]\n")

    trades, skipped = parse_robinhood_csv(csv_file, user.id)

    imported = 0
    duplicates = 0
    for trade in trades:
        if repo.trade_exists(
            user.id, trade.ticker, trade.trade_type,
            trade.shares, trade.price_per_share, trade.trade_date,
        ):
            duplicates += 1
            continue
        repo.add_trade(trade)
        imported += 1

    repo.recalculate_holdings(user.id)

    console.print(f"  [green]Imported:[/green] {imported} trades")
    if duplicates:
        console.print(f"  [yellow]Skipped (duplicates):[/yellow] {duplicates}")
    if skipped:
        console.print(f"  [dim]Skipped (non-trade entries):[/dim] {len(skipped)}")
        for entry in skipped[:5]:
            console.print(f"    [dim]Row {entry['row']}: {entry['reason']}[/dim]")
        if len(skipped) > 5:
            console.print(f"    [dim]... and {len(skipped) - 5} more[/dim]")

    console.print()
    db.close()


@portfolio_app.command("history")
def portfolio_history(
    ticker: str = typer.Option(None, "--ticker", help="Filter by ticker symbol"),
) -> None:
    """Show trade history."""
    db, repo, user = _get_db_and_user()
    trades = repo.get_trades(user.id, ticker=ticker)

    if not trades:
        msg = f"for {ticker.upper()}" if ticker else ""
        console.print(f"\n[yellow]No trade history found {msg}.[/yellow]\n")
        db.close()
        return

    table = Table(title="Trade History", show_lines=True)
    table.add_column("Date", style="dim")
    table.add_column("Type")
    table.add_column("Ticker", style="bold cyan")
    table.add_column("Shares", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Source", style="dim")

    for t in trades:
        type_color = "green" if t.trade_type == "buy" else "red"
        table.add_row(
            t.trade_date.strftime("%Y-%m-%d"),
            f"[{type_color}]{t.trade_type.upper()}[/{type_color}]",
            t.ticker,
            f"{t.shares:.4f}",
            f"${t.price_per_share:.2f}",
            f"${t.total_amount:,.2f}",
            t.source,
        )

    console.print()
    console.print(table)
    console.print()
    db.close()


@portfolio_app.command("analyze")
def portfolio_analyze(
    ticker: str = typer.Argument(None, help="Ticker to analyze (omit for full portfolio)"),
    explain: bool = typer.Option(False, "--explain", help="Include beginner-friendly explanations"),
) -> None:
    """
    AI-powered portfolio analysis.

    Without a ticker: analyzes the full portfolio.
    With a ticker: analyzes that specific holding in portfolio context.
    """
    settings = get_settings()
    warnings = settings.validate_keys()
    for w in warnings:
        console.print(f"[yellow]⚠ {w}[/yellow]")

    db, repo, user = _get_db_and_user()

    from marketmind.agent.orchestrator import MarketMindAgent

    agent = MarketMindAgent(settings, db=db)

    if ticker:
        result = agent.analyze_holding(ticker, user.id, explain=explain)
    else:
        result = agent.analyze_portfolio(user.id, explain=explain)

    if result.analysis_type == "error":
        console.print(f"\n[red]{result.detailed_analysis}[/red]\n")
        db.close()
        raise typer.Exit(1)

    title = f"📊 Portfolio Analysis: {ticker.upper()}" if ticker else "📊 Portfolio Analysis"
    console.print(Panel(result.detailed_analysis, title=title, border_style="blue"))

    meta_table = Table(show_header=False, box=None)
    meta_table.add_row("Model", result.model_used)
    meta_table.add_row("Cost", f"${result.cost_usd:.4f}")
    meta_table.add_row("Timestamp", str(result.timestamp.strftime("%Y-%m-%d %H:%M")))
    console.print(meta_table)
    console.print()
    db.close()


# ──────────────────────────────────────────────
# Watchlist commands
# ──────────────────────────────────────────────


@watchlist_app.command("add")
def watchlist_add(
    ticker: str = typer.Argument(help="Stock ticker symbol to watch"),
    reason: str = typer.Option("", "--reason", help="Why you're watching this stock"),
) -> None:
    """Add a ticker to your watchlist."""
    db, repo, user = _get_db_and_user()
    repo.add_to_watchlist(user.id, ticker.upper(), reason)
    console.print(f"\n[green]Added {ticker.upper()} to watchlist.[/green]")
    if reason:
        console.print(f"  [dim]Reason: {reason}[/dim]")
    console.print()
    db.close()


@watchlist_app.command("show")
def watchlist_show() -> None:
    """Show your watchlist."""
    db, repo, user = _get_db_and_user()
    items = repo.get_watchlist(user.id)

    if not items:
        console.print("\n[yellow]Watchlist is empty.[/yellow] Use 'watchlist add' to add tickers.\n")
        db.close()
        return

    table = Table(title=f"Watchlist — {user.username}", show_lines=True)
    table.add_column("Ticker", style="bold cyan")
    table.add_column("Notes")
    table.add_column("Added", style="dim")

    for item in items:
        table.add_row(
            item.ticker,
            item.notes or "—",
            item.added_at.strftime("%Y-%m-%d"),
        )

    console.print()
    console.print(table)
    console.print()
    db.close()


@watchlist_app.command("remove")
def watchlist_remove(
    ticker: str = typer.Argument(help="Stock ticker symbol to remove"),
) -> None:
    """Remove a ticker from your watchlist."""
    db, repo, user = _get_db_and_user()
    removed = repo.remove_from_watchlist(user.id, ticker.upper())

    if removed:
        console.print(f"\n[green]Removed {ticker.upper()} from watchlist.[/green]\n")
    else:
        console.print(f"\n[yellow]{ticker.upper()} was not on your watchlist.[/yellow]\n")
    db.close()


# ──────────────────────────────────────────────
# Top-level commands
# ──────────────────────────────────────────────


@app.command()
def analyze(
    ticker: str = typer.Argument(help="Stock ticker symbol (e.g., AAPL, MSFT, NVDA)"),
    explain: bool = typer.Option(False, "--explain", help="Include beginner-friendly explanations of every metric"),
) -> None:
    """
    Run a comprehensive AI analysis on a stock.

    Fetches price, fundamentals, history, and analyst data, then sends
    everything to Claude Sonnet for deep analysis.

    Use --explain for educational mode with metric explanations.

    Example: marketmind analyze AAPL
             marketmind analyze AAPL --explain
    """
    settings = get_settings()
    warnings = settings.validate_keys()
    for w in warnings:
        console.print(f"[yellow]⚠ {w}[/yellow]")

    from marketmind.agent.orchestrator import MarketMindAgent
    from marketmind.db.database import Database

    db = Database(data_dir=settings.data_dir)
    db.initialize()
    agent = MarketMindAgent(settings, db=db)
    result = agent.analyze(ticker, explain=explain)

    # Display results
    console.print(Panel(result.detailed_analysis, title=f"📊 Analysis: {result.ticker}", border_style="blue"))

    # Show metadata
    meta_table = Table(show_header=False, box=None)
    meta_table.add_row("Model", result.model_used)
    meta_table.add_row("Cost", f"${result.cost_usd:.4f}")
    meta_table.add_row("Data Sources", ", ".join(result.data_sources))
    meta_table.add_row("Timestamp", str(result.timestamp.strftime("%Y-%m-%d %H:%M")))
    console.print(meta_table)
    db.close()


@app.command()
def evaluate(
    ticker: str = typer.Argument(help="Stock ticker symbol to evaluate as a potential buy"),
    explain: bool = typer.Option(False, "--explain", help="Include beginner-friendly explanations"),
) -> None:
    """
    Evaluate a stock as a potential buy against your current portfolio.

    Fetches stock data and compares it to your portfolio for diversification,
    sector overlap, and position sizing recommendations.

    Example: marketmind evaluate NVDA
             marketmind evaluate NVDA --explain
    """
    settings = get_settings()
    warnings = settings.validate_keys()
    for w in warnings:
        console.print(f"[yellow]⚠ {w}[/yellow]")

    db, repo, user = _get_db_and_user()

    from marketmind.agent.orchestrator import MarketMindAgent

    agent = MarketMindAgent(settings, db=db)
    result = agent.evaluate_potential_buy(ticker, user.id, explain=explain)

    console.print(Panel(result.detailed_analysis, title=f"🔍 Evaluation: {result.ticker}", border_style="green"))

    meta_table = Table(show_header=False, box=None)
    meta_table.add_row("Model", result.model_used)
    meta_table.add_row("Cost", f"${result.cost_usd:.4f}")
    meta_table.add_row("Data Sources", ", ".join(result.data_sources))
    meta_table.add_row("Timestamp", str(result.timestamp.strftime("%Y-%m-%d %H:%M")))
    console.print(meta_table)
    console.print()
    db.close()


@app.command()
def price(
    ticker: str = typer.Argument(help="Stock ticker symbol"),
) -> None:
    """
    Quick price check (uses cheap model).

    Example: marketmind price TSLA
    """
    settings = get_settings()

    from marketmind.agent.orchestrator import MarketMindAgent

    agent = MarketMindAgent(settings)
    result = agent.quick_price(ticker)

    console.print(f"\n[bold]{ticker.upper()}[/bold]: {result}\n")


@app.command()
def learn(
    topic: str = typer.Argument(help="Financial concept to learn about (e.g., 'P/E ratio')"),
) -> None:
    """
    Learn a financial concept explained in plain English.

    Uses the cheap model (GPT-4o-mini) to explain financial terms and concepts
    with real examples, for someone with zero finance background.

    Example: marketmind learn "P/E ratio"
             marketmind learn "how to read an earnings report"
    """
    settings = get_settings()

    from marketmind.llm.router import LLMRouter
    from marketmind.prompts.templates import LEARN_TOPIC_PROMPT

    router = LLMRouter(settings)
    prompt = LEARN_TOPIC_PROMPT.format(topic=topic)

    console.print(f"\n[bold blue]Learning about: {topic}[/bold blue]\n")
    result = router.call(prompt=prompt, task_type="routine")

    console.print(Panel(result["content"], title=f"📚 {topic}", border_style="green"))

    # Show cost
    if result["cost_record"]:
        console.print(f"  [dim]Model: {result['model']} | Cost: ${result['cost_record'].cost_usd:.4f}[/dim]\n")


@app.command()
def cost() -> None:
    """
    Show API spending for the current month.

    Displays total spend, budget remaining, and per-model breakdown.
    """
    settings = get_settings()
    from marketmind.llm.cost_tracker import CostTracker

    tracker = CostTracker(
        budget_limit=settings.monthly_budget_limit,
        data_dir=settings.data_dir,
    )

    status = tracker.get_budget_status()

    # Color coding based on usage
    if status["usage_percent"] >= 90:
        color = "red"
    elif status["usage_percent"] >= 70:
        color = "yellow"
    else:
        color = "green"

    console.print(f"\n[bold]MarketMind — Cost Report[/bold]\n")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Monthly Spend", f"[{color}]${status['monthly_spend']:.4f}[/{color}]")
    table.add_row("Monthly Limit", f"${status['monthly_limit']:.2f}")
    table.add_row("Remaining", f"${status['remaining']:.4f}")
    table.add_row("Usage", f"[{color}]{status['usage_percent']:.1f}%[/{color}]")
    console.print(table)
    console.print()


@app.command(name="check-setup")
def check_setup() -> None:
    """
    Verify that all API keys and dependencies are configured correctly.

    Run this after initial setup to make sure everything works.
    """
    console.print("\n[bold]MarketMind — Setup Check[/bold]\n")

    # Check .env file
    try:
        settings = get_settings()
        console.print("[green]✓[/green] .env file loaded")
    except Exception as e:
        console.print(f"[red]✗ Failed to load .env: {e}[/red]")
        console.print("  → Make sure you've copied .env.example to .env and filled in your keys")
        raise typer.Exit(1)

    # Check API keys
    warnings = settings.validate_keys()
    if not warnings:
        console.print("[green]✓[/green] All API keys configured")
    else:
        for w in warnings:
            console.print(f"[yellow]⚠ {w}[/yellow]")

    # Check yfinance
    try:
        from marketmind.tools.price import get_stock_price

        test_price = get_stock_price("AAPL")
        if test_price:
            console.print(f"[green]✓[/green] yfinance working (AAPL: ${test_price.current_price})")
        else:
            console.print("[yellow]⚠ yfinance returned no data — might be a network issue[/yellow]")
    except Exception as e:
        console.print(f"[red]✗ yfinance error: {e}[/red]")

    # Check OpenAI
    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'ok' and nothing else."}],
            max_tokens=5,
        )
        console.print("[green]✓[/green] OpenAI API working")
    except Exception as e:
        console.print(f"[red]✗ OpenAI API error: {e}[/red]")

    # Check Anthropic
    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=settings.anthropic_api_key)
        response = client.messages.create(
            model=settings.analysis_model,
            max_tokens=5,
            messages=[{"role": "user", "content": "Say 'ok' and nothing else."}],
        )
        console.print("[green]✓[/green] Anthropic API working")
    except Exception as e:
        console.print(f"[red]✗ Anthropic API error: {e}[/red]")

    console.print("\n[bold]Setup check complete.[/bold]\n")


if __name__ == "__main__":
    app()
