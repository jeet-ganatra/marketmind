"""
Vector store layer using ChromaDB for document storage and semantic search.

Stores chunked, embedded documents (SEC filings, news articles) for RAG retrieval.
Uses sentence-transformers for local embeddings (no API key needed).
"""

import hashlib
from pathlib import Path

from rich.console import Console

console = Console()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks by word count.

    Args:
        text: The text to chunk.
        chunk_size: Target number of words per chunk.
        overlap: Number of words to overlap between consecutive chunks.

    Returns:
        List of text chunks. Trailing chunks under 50 words are skipped.
    """
    if not text or not text.strip():
        return []

    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()] if len(words) >= 50 else []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        # Skip tiny trailing chunks
        if len(chunk_words) < 50 and chunks:
            break

        chunks.append(" ".join(chunk_words))
        start += chunk_size - overlap

    return chunks


def _document_id(collection: str, metadata: dict, chunk_index: int) -> str:
    """Generate a deterministic document ID from metadata and chunk index."""
    key_parts = [
        collection,
        metadata.get("ticker", ""),
        metadata.get("doc_type", ""),
        metadata.get("date", ""),
        metadata.get("section", ""),
        metadata.get("source", ""),
        str(chunk_index),
    ]
    raw = "|".join(key_parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


class VectorStore:
    """
    ChromaDB-backed vector store for document embeddings.

    Uses a local PersistentClient so data survives restarts.
    The embedding model is loaded lazily on first use.
    """

    def __init__(
        self,
        data_dir: Path,
        embedding_model: str = "all-mpnet-base-v2",
        client=None,
    ) -> None:
        """
        Args:
            data_dir: Root data directory. ChromaDB stored at {data_dir}/chromadb/.
            embedding_model: Sentence-transformers model name.
            client: Optional ChromaDB client for testing (e.g., EphemeralClient).
        """
        self._data_dir = data_dir
        self._model_name = embedding_model
        self._client = client
        self._embedding_fn = None

    def _get_client(self):
        """Lazy-init ChromaDB PersistentClient."""
        if self._client is None:
            import chromadb

            chroma_dir = self._data_dir / "chromadb"
            chroma_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(chroma_dir))
        return self._client

    def _get_embedding_function(self):
        """Lazy-init the sentence-transformers embedding function."""
        if self._embedding_fn is None:
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )

            console.print(
                f"[dim]Loading embedding model ({self._model_name})...[/dim]"
            )
            self._embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name=self._model_name
            )
        return self._embedding_fn

    def _get_collection(self, name: str):
        """Get or create a ChromaDB collection with cosine similarity."""
        client = self._get_client()
        ef = self._get_embedding_function()
        return client.get_or_create_collection(
            name=name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: list[dict], collection: str) -> int:
        """
        Add chunked documents to a collection.

        Each document dict must have:
            - "text": str, the chunk text
            - "metadata": dict with keys like ticker, doc_type, section, date, source

        Uses upsert for idempotency. Returns the number of documents upserted.
        """
        if not documents:
            return 0

        coll = self._get_collection(collection)

        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict] = []

        for i, doc in enumerate(documents):
            text = doc.get("text", "").strip()
            meta = doc.get("metadata", {})
            if not text:
                continue

            doc_id = _document_id(collection, meta, i)
            ids.append(doc_id)
            texts.append(text)
            # ChromaDB metadata values must be str, int, float, or bool
            clean_meta = {k: str(v) if not isinstance(v, (int, float, bool)) else v
                         for k, v in meta.items()}
            metadatas.append(clean_meta)

        if not ids:
            return 0

        # Upsert in batches of 500 to avoid ChromaDB limits
        batch_size = 500
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            coll.upsert(
                ids=ids[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )

        return len(ids)

    def search(
        self,
        query: str,
        collection: str,
        n_results: int = 5,
        filters: dict | None = None,
    ) -> list[dict]:
        """
        Semantic search over a collection.

        Args:
            query: Search query text.
            collection: Collection name ("sec_filings" or "news").
            n_results: Number of results to return.
            filters: Optional metadata filters (e.g., {"ticker": "AAPL"}).

        Returns:
            List of dicts with "text", "metadata", and "distance" keys.
        """
        coll = self._get_collection(collection)

        where = None
        if filters:
            if len(filters) == 1:
                where = filters
            else:
                where = {"$and": [{k: v} for k, v in filters.items()]}

        results = coll.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        output: list[dict] = []
        if results and results["ids"] and results["ids"][0]:
            for j in range(len(results["ids"][0])):
                output.append({
                    "text": results["documents"][0][j],
                    "metadata": results["metadatas"][0][j],
                    "distance": results["distances"][0][j],
                })
        return output

    def has_document(
        self,
        ticker: str,
        doc_type: str,
        date: str,
        collection: str,
    ) -> bool:
        """Check if a document with the given metadata already exists."""
        coll = self._get_collection(collection)
        result = coll.get(
            where={"$and": [
                {"ticker": ticker},
                {"doc_type": doc_type},
                {"date": date},
            ]},
            include=[],
        )
        return len(result["ids"]) > 0

    def get_document_count(self, collection: str, ticker: str | None = None) -> int:
        """Get count of documents in a collection, optionally filtered by ticker."""
        coll = self._get_collection(collection)
        if ticker is None:
            return coll.count()
        result = coll.get(where={"ticker": ticker}, include=[])
        return len(result["ids"])

    def delete_documents(self, ticker: str, collection: str) -> None:
        """Delete all documents for a ticker from a collection."""
        coll = self._get_collection(collection)
        coll.delete(where={"ticker": ticker})
