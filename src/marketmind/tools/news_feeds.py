"""
News feed ingestion via RSS.

Parses free RSS feeds from major financial news sources,
filters by ticker/company name, and stores in ChromaDB.
"""

import hashlib
from datetime import datetime

import feedparser
from rich.console import Console

from marketmind.db.vector_store import VectorStore, chunk_text

console = Console()

# General market feeds (used when no tickers specified)
GENERAL_FEEDS = {
    "cnbc_top": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "cnbc_finance": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
}

# Per-ticker feed templates ({ticker} is replaced at runtime)
TICKER_FEEDS = {
    "yahoo_headline": "https://finance.yahoo.com/rss/headline?s={ticker}",
    "google_news": "https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en",
    "seeking_alpha": "https://seekingalpha.com/api/sa/combined/{ticker}.xml",
}

NEWS_COLLECTION = "news"

# Cache for ticker -> company name lookups
_company_name_cache: dict[str, str] = {}


def _get_company_name(ticker: str) -> str:
    """Look up company name from ticker using yfinance. Cached per session."""
    ticker = ticker.upper()
    if ticker in _company_name_cache:
        return _company_name_cache[ticker]

    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        name = info.get("shortName", "") or info.get("longName", "")
        _company_name_cache[ticker] = name
        return name
    except Exception:
        _company_name_cache[ticker] = ""
        return ""


def _article_matches_tickers(
    article: dict, tickers: list[str], company_names: dict[str, str]
) -> bool:
    """Check if an article mentions any of the given tickers or company names."""
    text = (
        article.get("title", "") + " " + article.get("summary", "")
    ).upper()

    for ticker in tickers:
        if ticker.upper() in text:
            return True
        name = company_names.get(ticker.upper(), "")
        if name and name.upper() in text:
            return True

    return False


def _matched_tickers(
    article: dict, tickers: list[str], company_names: dict[str, str]
) -> list[str]:
    """Return the list of tickers that an article mentions."""
    text = (
        article.get("title", "") + " " + article.get("summary", "")
    ).upper()

    matched = []
    for ticker in tickers:
        if ticker.upper() in text:
            matched.append(ticker.upper())
        elif (name := company_names.get(ticker.upper(), "")) and name.upper() in text:
            matched.append(ticker.upper())

    return matched


def _parse_feed(feed_url: str, source_name: str) -> list[dict]:
    """Parse a single RSS feed and return a list of article dicts."""
    articles: list[dict] = []
    try:
        feed = feedparser.parse(feed_url)

        if feed.bozo and not feed.entries:
            console.print(f"  [yellow]Warning: Feed {source_name} returned no entries[/yellow]")
            return articles

        for entry in feed.entries:
            title = entry.get("title", "").strip()
            summary = entry.get("summary", "").strip()
            link = entry.get("link", "").strip()

            # Parse published date
            published = ""
            if "published_parsed" in entry and entry.published_parsed:
                try:
                    published = datetime(*entry.published_parsed[:6]).isoformat()
                except Exception:
                    published = entry.get("published", "")
            elif "published" in entry:
                published = entry.get("published", "")

            if not title:
                continue

            articles.append({
                "title": title,
                "summary": summary,
                "published": published,
                "source": source_name,
                "link": link,
            })

    except Exception as e:
        console.print(f"  [yellow]Warning: Failed to parse feed {source_name}: {e}[/yellow]")

    return articles


def fetch_news(tickers: list[str] | None = None) -> list[dict]:
    """
    Fetch articles from RSS feeds.

    When tickers are provided, fetches per-ticker feeds (Yahoo Finance headline,
    Google News, Seeking Alpha) for each ticker. Falls back to general market
    feeds when no tickers are specified.

    Args:
        tickers: If provided, fetch ticker-specific feeds for these symbols.

    Returns:
        List of article dicts with keys: title, summary, published, source, link, ticker.
    """
    articles: list[dict] = []
    seen_links: set[str] = set()

    def _dedup_append(article: dict) -> None:
        """Append article if its link hasn't been seen yet."""
        link = article.get("link", "")
        key = link if link else article["title"]
        if key not in seen_links:
            seen_links.add(key)
            articles.append(article)

    if tickers:
        # Fetch per-ticker feeds for each ticker
        for ticker in tickers:
            t = ticker.upper()
            for source_name, url_template in TICKER_FEEDS.items():
                feed_url = url_template.format(ticker=t)
                for article in _parse_feed(feed_url, f"{source_name}_{t}"):
                    article["ticker"] = t
                    _dedup_append(article)

        # Also check general feeds for mentions
        company_names: dict[str, str] = {}
        for t in tickers:
            company_names[t.upper()] = _get_company_name(t)

        for source_name, feed_url in GENERAL_FEEDS.items():
            for article in _parse_feed(feed_url, source_name):
                if _article_matches_tickers(article, tickers, company_names):
                    _dedup_append(article)
    else:
        # No tickers: return all articles from general feeds
        for source_name, feed_url in GENERAL_FEEDS.items():
            for article in _parse_feed(feed_url, source_name):
                _dedup_append(article)

    return articles


def ingest_news(
    tickers: list[str] | None = None,
    vector_store: VectorStore | None = None,
) -> dict:
    """
    Fetch and ingest news articles into ChromaDB.

    Args:
        tickers: Optional ticker filter.
        vector_store: VectorStore instance.

    Returns:
        Summary dict with num_articles_found, num_new_stored, num_duplicates_skipped.
    """
    if vector_store is None:
        return {"error": "VectorStore not provided"}

    articles = fetch_news(tickers=tickers)
    num_found = len(articles)
    num_stored = 0
    num_skipped = 0

    # Pre-resolve company names for articles from general feeds (no .ticker set)
    company_names: dict[str, str] = {}
    if tickers:
        for t in tickers:
            company_names[t.upper()] = _get_company_name(t)

    for article in articles:
        # Build document text from title + summary
        text = article["title"]
        if article.get("summary"):
            text += "\n\n" + article["summary"]

        chunks = chunk_text(text, chunk_size=300)
        if not chunks:
            # Short articles: store as a single doc if at least 10 words
            if len(text.split()) >= 10:
                chunks = [text]
            else:
                continue

        # Determine ticker labels for storage
        if "ticker" in article:
            # Per-ticker feed: already tagged
            ticker_labels = [article["ticker"]]
        elif tickers:
            # General feed article: match against tickers
            matched = _matched_tickers(article, tickers, company_names)
            ticker_labels = matched if matched else ["general"]
        else:
            ticker_labels = ["general"]

        # Store one copy per matched ticker so per-ticker queries work
        stored_any = False
        for ticker_label in ticker_labels:
            documents = []
            for i, chunk in enumerate(chunks):
                documents.append({
                    "text": chunk,
                    "metadata": {
                        "doc_type": "news",
                        "ticker": ticker_label,
                        "date": article.get("published", ""),
                        "source": f"rss_{article['source']}",
                        "link": article.get("link", ""),
                        "section": "news_article",
                        "chunk_index": i,
                    },
                })

            n = vector_store.add_documents(documents, collection=NEWS_COLLECTION)
            if n > 0:
                stored_any = True

        if stored_any:
            num_stored += 1
        else:
            num_skipped += 1

    return {
        "num_articles_found": num_found,
        "num_new_stored": num_stored,
        "num_duplicates_skipped": num_skipped,
    }
