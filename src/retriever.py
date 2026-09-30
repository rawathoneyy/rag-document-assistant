"""Retrieval logic: find the most relevant chunks for a given query."""

from embedder import embed_query


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Manual cosine similarity - no numpy or sklearn needed."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def retrieve_top_chunks(query: str, store: list[dict], top_k: int = 3) -> list[dict]:
    """Embed the query, compare it against every chunk in the store, and
    return the top_k most similar chunks, each with its similarity score
    and originating page number attached."""
    query_embedding = embed_query(query)

    scored_chunks = []
    for entry in store:
        score = cosine_similarity(query_embedding, entry["embedding"])
        scored_chunks.append({
            "text": entry["text"],
            "page_number": entry["page_number"],
            "score": score,
        })

    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return scored_chunks[:top_k]


if __name__ == "__main__":
    from pathlib import Path

    from document_loader import chunk_pages, extract_pages_from_pdf
    from embedder import build_document_store

    project_root = Path(__file__).resolve().parent.parent
    test_pdf = project_root / "data" / "artificial_intelligence_and_machine_learning.pdf"

    pages = extract_pages_from_pdf(test_pdf)
    chunks = chunk_pages(pages)
    store = build_document_store(chunks)

    test_query = "What is overfitting?"
    print(f"Query: {test_query}\n")

    results = retrieve_top_chunks(test_query, store, top_k=3)
    for i, result in enumerate(results, 1):
        print(f"--- Result {i} (page {result['page_number']}, score: {result['score']:.4f}) ---")
        print(result["text"][:200])
        print()