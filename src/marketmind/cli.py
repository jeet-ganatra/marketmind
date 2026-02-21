"""
MarketMind CLI — The primary user interface.

Commands:
  marketmind analyze <TICKER>             — Deep analysis using Claude Sonnet
  marketmind analyze <TICKER> --explain   — Educational analysis with metric explanations
  marketmind price <TICKER>               — Quick price check using GPT-4o-mini
  marketmind learn <TOPIC>                — Learn a financial concept using GPT-4o-mini
  marketmind cost                         — Show API spending for the current month
  marketmind check-setup                  — Verify all API keys and dependencies

Usage:
  poetry run marketmind analyze AAPL
  poetry run marketmind analyze AAPL --explain
  poetry run marketmind price MSFT
  poetry run marketmind learn "P/E ratio"
  poetry run marketmind cost
"""

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

    agent = MarketMindAgent(settings)
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
