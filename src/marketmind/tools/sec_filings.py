"""
SEC EDGAR filing ingestion via EdgarTools.

Fetches 10-K, 10-Q, 8-K filings and Form 4 insider trades,
extracts key sections, chunks them, and stores in ChromaDB.
"""

from rich.console import Console

from marketmind.db.vector_store import VectorStore, chunk_text

console = Console()

# Sections to extract from 10-K filings (property name -> metadata section label)
TENK_SECTIONS = {
    "management_discussion": "mdna",
    "risk_factors": "risk_factors",
    "business": "business_overview",
}

# Sections to extract from 10-Q filings (bracket key -> metadata section label)
TENQ_SECTIONS = {
    "Part I, Item 2": "mdna",
    "Part II, Item 1A": "risk_factors",
}

SEC_COLLECTION = "sec_filings"


def _ensure_identity(settings) -> bool:
    """Set EdgarTools identity from config. Returns False if not configured."""
    if not settings.sec_edgar_identity:
        console.print(
            "[red]SEC_EDGAR_IDENTITY not configured in .env.[/red]\n"
            "Add: SEC_EDGAR_IDENTITY=MarketMind your.email@example.com"
        )
        return False

    from edgar import set_identity
    set_identity(settings.sec_edgar_identity)
    return True


def _extract_section_text(obj, section_key: str, use_bracket: bool = False) -> str | None:
    """
    Safely extract text from a filing object section.

    Args:
        obj: TenK, TenQ, or EightK object.
        section_key: Property name or bracket key.
        use_bracket: If True, use obj[section_key] instead of getattr.

    Returns:
        Section text as string, or None if not available.
    """
    try:
        if use_bracket:
            section = obj[section_key]
        else:
            section = getattr(obj, section_key, None)

        if section is None:
            return None

        text = str(section).strip()
        return text if text else None
    except Exception:
        return None


def _store_chunks(
    vector_store: VectorStore,
    text: str,
    ticker: str,
    doc_type: str,
    section: str,
    date: str,
    source: str = "sec_edgar",
    chunk_size: int = 500,
) -> int:
    """Chunk text and store in vector store. Returns number of chunks stored."""
    chunks = chunk_text(text, chunk_size=chunk_size)
    if not chunks:
        return 0

    documents = []
    for i, chunk in enumerate(chunks):
        documents.append({
            "text": chunk,
            "metadata": {
                "ticker": ticker,
                "doc_type": doc_type,
                "section": section,
                "date": date,
                "source": source,
                "chunk_index": i,
            },
        })

    return vector_store.add_documents(documents, collection=SEC_COLLECTION)


def ingest_company_filings(
    ticker: str,
    filing_type: str = "both",
    num_quarters: int = 8,
    vector_store: VectorStore | None = None,
    settings=None,
) -> dict:
    """
    Ingest SEC filings (10-K and/or 10-Q) for a company.

    Args:
        ticker: Stock ticker symbol.
        filing_type: "10-K", "10-Q", or "both".
        num_quarters: Number of recent filings to fetch per type.
        vector_store: VectorStore instance for storage.
        settings: Application settings (for SEC identity).

    Returns:
        Summary dict with ticker, filing_type, num_filings_processed,
        num_chunks_stored, and sections_extracted.
    """
    if settings and not _ensure_identity(settings):
        return {"error": "SEC_EDGAR_IDENTITY not configured"}

    if vector_store is None:
        return {"error": "VectorStore not provided"}

    from edgar import Company

    ticker = ticker.upper()
    total_filings = 0
    total_chunks = 0
    sections_found: list[str] = []

    try:
        company = Company(ticker)
    except Exception as e:
        console.print(f"[red]Could not find company for {ticker}: {e}[/red]")
        return {"error": f"Company not found: {e}"}

    # Ingest 10-K filings
    if filing_type in ("10-K", "both"):
        try:
            filings = company.get_filings(form="10-K").head(num_quarters)
            for filing in filings:
                filing_date = str(filing.filing_date)

                # Check for duplicates
                if vector_store.has_document(ticker, "10-K", filing_date, SEC_COLLECTION):
                    console.print(f"  [dim]Skipping 10-K {filing_date} (already ingested)[/dim]")
                    continue

                console.print(f"  Processing 10-K filed {filing_date}...")
                try:
                    tenk = filing.obj()
                    if tenk is None:
                        console.print(f"  [yellow]Warning: Could not parse 10-K {filing_date}[/yellow]")
                        continue
                except Exception as e:
                    console.print(f"  [yellow]Warning: Error parsing 10-K {filing_date}: {e}[/yellow]")
                    continue

                filing_chunks = 0
                for prop_name, section_label in TENK_SECTIONS.items():
                    text = _extract_section_text(tenk, prop_name)
                    if text:
                        n = _store_chunks(
                            vector_store, text, ticker, "10-K",
                            section_label, filing_date,
                        )
                        filing_chunks += n
                        if n > 0 and section_label not in sections_found:
                            sections_found.append(section_label)
                    else:
                        console.print(
                            f"    [dim]Section {section_label} not available[/dim]"
                        )

                total_chunks += filing_chunks
                total_filings += 1
        except Exception as e:
            console.print(f"  [yellow]Warning: Error fetching 10-K filings: {e}[/yellow]")

    # Ingest 10-Q filings
    if filing_type in ("10-Q", "both"):
        try:
            filings = company.get_filings(form="10-Q").head(num_quarters)
            for filing in filings:
                filing_date = str(filing.filing_date)

                if vector_store.has_document(ticker, "10-Q", filing_date, SEC_COLLECTION):
                    console.print(f"  [dim]Skipping 10-Q {filing_date} (already ingested)[/dim]")
                    continue

                console.print(f"  Processing 10-Q filed {filing_date}...")
                try:
                    tenq = filing.obj()
                    if tenq is None:
                        console.print(f"  [yellow]Warning: Could not parse 10-Q {filing_date}[/yellow]")
                        continue
                except Exception as e:
                    console.print(f"  [yellow]Warning: Error parsing 10-Q {filing_date}: {e}[/yellow]")
                    continue

                filing_chunks = 0
                for bracket_key, section_label in TENQ_SECTIONS.items():
                    text = _extract_section_text(tenq, bracket_key, use_bracket=True)
                    if text:
                        n = _store_chunks(
                            vector_store, text, ticker, "10-Q",
                            section_label, filing_date,
                        )
                        filing_chunks += n
                        if n > 0 and section_label not in sections_found:
                            sections_found.append(section_label)
                    else:
                        console.print(
                            f"    [dim]Section {section_label} not available[/dim]"
                        )

                total_chunks += filing_chunks
                total_filings += 1
        except Exception as e:
            console.print(f"  [yellow]Warning: Error fetching 10-Q filings: {e}[/yellow]")

    return {
        "ticker": ticker,
        "filing_type": filing_type,
        "num_filings_processed": total_filings,
        "num_chunks_stored": total_chunks,
        "sections_extracted": sections_found,
    }


def ingest_insider_trades(
    ticker: str,
    num_filings: int = 20,
    vector_store: VectorStore | None = None,
    settings=None,
) -> dict:
    """
    Ingest Form 4 insider trading filings for a company.

    Args:
        ticker: Stock ticker symbol.
        num_filings: Number of recent Form 4 filings to fetch.
        vector_store: VectorStore instance.
        settings: Application settings.

    Returns:
        Summary dict with ticker, num_filings_processed, num_chunks_stored.
    """
    if settings and not _ensure_identity(settings):
        return {"error": "SEC_EDGAR_IDENTITY not configured"}

    if vector_store is None:
        return {"error": "VectorStore not provided"}

    from edgar import Company

    ticker = ticker.upper()
    total_filings = 0
    total_chunks = 0

    try:
        company = Company(ticker)
        filings = company.get_filings(form="4").head(num_filings)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not fetch Form 4 filings for {ticker}: {e}[/yellow]")
        return {"ticker": ticker, "num_filings_processed": 0, "num_chunks_stored": 0}

    for filing in filings:
        filing_date = str(filing.filing_date)

        if vector_store.has_document(ticker, "Form4", filing_date, SEC_COLLECTION):
            continue

        try:
            form4 = filing.obj()
            if form4 is None:
                continue

            # Build a text summary of the insider trade
            lines = [
                f"Form 4 Insider Trading: {ticker}",
                f"Filing Date: {filing_date}",
            ]

            owner_name = getattr(form4, "reporting_owner_name", "Unknown")
            relationship = getattr(form4, "reporting_owner_relationship", "Unknown")
            lines.append(f"Insider: {owner_name} ({relationship})")

            transactions = getattr(form4, "transactions", [])
            if transactions:
                for txn in transactions:
                    txn_date = getattr(txn, "transaction_date", "N/A")
                    txn_code = getattr(txn, "transaction_code", "N/A")
                    shares = getattr(txn, "shares", 0)
                    price = getattr(txn, "price_per_share", 0)
                    lines.append(
                        f"  {txn_date}: {txn_code} {shares:,.0f} shares @ ${price:,.2f}"
                    )

            try:
                net = form4.get_net_shares_traded()
                direction = "NET BUY" if net > 0 else "NET SELL" if net < 0 else "NEUTRAL"
                lines.append(f"Net shares traded: {net:,.0f} ({direction})")
            except Exception:
                pass

            text = "\n".join(lines)
            n = _store_chunks(
                vector_store, text, ticker, "Form4",
                "insider_trade", filing_date,
                chunk_size=200,
            )
            total_chunks += n
            total_filings += 1

        except Exception as e:
            console.print(f"  [yellow]Warning: Error parsing Form 4 {filing_date}: {e}[/yellow]")
            continue

    return {
        "ticker": ticker,
        "num_filings_processed": total_filings,
        "num_chunks_stored": total_chunks,
    }


def ingest_material_events(
    ticker: str,
    num_filings: int = 10,
    vector_store: VectorStore | None = None,
    settings=None,
) -> dict:
    """
    Ingest 8-K filings (material events) for a company.

    8-K filings cover earnings releases, acquisitions, management changes,
    and other material events that companies must disclose.

    Args:
        ticker: Stock ticker symbol.
        num_filings: Number of recent 8-K filings to fetch.
        vector_store: VectorStore instance.
        settings: Application settings.

    Returns:
        Summary dict with ticker, num_filings_processed, num_chunks_stored.
    """
    if settings and not _ensure_identity(settings):
        return {"error": "SEC_EDGAR_IDENTITY not configured"}

    if vector_store is None:
        return {"error": "VectorStore not provided"}

    from edgar import Company

    ticker = ticker.upper()
    total_filings = 0
    total_chunks = 0

    try:
        company = Company(ticker)
        filings = company.get_filings(form="8-K").head(num_filings)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not fetch 8-K filings for {ticker}: {e}[/yellow]")
        return {"ticker": ticker, "num_filings_processed": 0, "num_chunks_stored": 0}

    for filing in filings:
        filing_date = str(filing.filing_date)

        if vector_store.has_document(ticker, "8-K", filing_date, SEC_COLLECTION):
            console.print(f"  [dim]Skipping 8-K {filing_date} (already ingested)[/dim]")
            continue

        console.print(f"  Processing 8-K filed {filing_date}...")

        try:
            eightk = filing.obj()
            if eightk is None:
                # Fall back to plain text extraction
                text = filing.text()
                if text:
                    n = _store_chunks(
                        vector_store, text, ticker, "8-K",
                        "material_event", filing_date,
                    )
                    total_chunks += n
                    total_filings += 1
                continue

            # Extract items from the 8-K
            items = getattr(eightk, "items", [])
            filing_text_parts = [f"8-K Filing: {ticker} ({filing_date})"]

            for item_name in items:
                item_text = _extract_section_text(eightk, item_name, use_bracket=True)
                if item_text:
                    filing_text_parts.append(f"\n{item_name}:\n{item_text}")

            if len(filing_text_parts) > 1:
                full_text = "\n".join(filing_text_parts)
                n = _store_chunks(
                    vector_store, full_text, ticker, "8-K",
                    "material_event", filing_date,
                )
                total_chunks += n
                total_filings += 1
            else:
                # No items extracted; try plain text
                text = filing.text()
                if text:
                    n = _store_chunks(
                        vector_store, text, ticker, "8-K",
                        "material_event", filing_date,
                    )
                    total_chunks += n
                    total_filings += 1

        except Exception as e:
            console.print(f"  [yellow]Warning: Error parsing 8-K {filing_date}: {e}[/yellow]")
            continue

    return {
        "ticker": ticker,
        "num_filings_processed": total_filings,
        "num_chunks_stored": total_chunks,
    }
