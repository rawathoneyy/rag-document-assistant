"""Embedding helpers for the RAG document assistant."""

import os

import cohere
from dotenv import load_dotenv

load_dotenv()

_client = None


def _get_api_key(key_name: str) -> str | None:
    """Check Streamlit Cloud's st.secrets first (when deployed), falling
    back to a local .env file otherwise."""
    try:
        import streamlit as st
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    return os.environ.get(key_name)


def _get_client():
    global _client
    if _client is None:
        api_key = _get_api_key("COHERE_API_KEY")
        if not api_key:
            raise ValueError(
                "COHERE_API_KEY not found. Add it to .env locally, or to "
                "Streamlit Cloud's Secrets settings when deployed."
            )
        _client = cohere.Client(api_key)
    return _client


def embed_chunks(chunk_texts: list[str]) -> list[list[float]]:
    """Embed a list of plain text strings in one batch call (more
    efficient than one API call per chunk). input_type='search_document'
    tells Cohere these are documents that will be searched against later.

    Takes plain strings, not chunk dicts - the page_number pairing
    happens one level up in build_document_store, so this function
    doesn't need to know about page numbers at all.
    """
    client = _get_client()
    response = client.embed(
        texts=chunk_texts,
        model="embed-english-v3.0",
        input_type="search_document",
    )
    return response.embeddings


def embed_query(query: str) -> list[float]:
    """Embed a single search query. input_type='search_query' is
    deliberately different from 'search_document' above - Cohere tunes
    the embedding differently depending on which side of the search
    it's used for, which improves retrieval quality."""
    client = _get_client()
    response = client.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query",
    )
    return response.embeddings[0]


def build_document_store(chunks: list[dict]) -> list[dict]:
    """Embed all chunks and pair each with its embedding and page number,
    returning a list of {"text": ..., "page_number": ..., "embedding": ...}
    dicts. This is our simple, in-memory "vector store" - no FAISS needed
    at this scale.

    chunks is now a list of {"text": ..., "page_number": ...} dicts
    (from document_loader.chunk_pages), not plain strings - that's what
    makes real page-number citations possible downstream.
    """
    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_chunks(chunk_texts)

    store = []
    for chunk, embedding in zip(chunks, embeddings):
        store.append({
            "text": chunk["text"],
            "page_number": chunk["page_number"],
            "embedding": embedding,
        })

    return store


if __name__ == "__main__":
    from pathlib import Path

    from document_loader import chunk_pages, extract_pages_from_pdf

    project_root = Path(__file__).resolve().parent.parent
    test_pdf = project_root / "data" / "artificial_intelligence_and_machine_learning.pdf"

    pages = extract_pages_from_pdf(test_pdf)
    chunks = chunk_pages(pages)

    print(f"Building document store from {len(chunks)} chunks...")
    store = build_document_store(chunks)

    print(f"Store has {len(store)} entries.")
    print(f"First entry's text (preview): {store[0]['text'][:100]}...")
    print(f"First entry's page number: {store[0]['page_number']}")
    print(f"First entry's embedding length: {len(store[0]['embedding'])}")