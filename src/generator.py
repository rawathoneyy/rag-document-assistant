"""Generate a grounded answer from retrieved chunks, using Groq's LLM."""

import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_client = None


def _get_api_key(key_name: str) -> str | None:
    """Check Streamlit Cloud's st.secrets first, falling back to .env."""
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
        api_key = _get_api_key("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Add it to .env locally, or to "
                "Streamlit Cloud's Secrets settings when deployed."
            )
        _client = Groq(api_key=api_key)
    return _client


def rewrite_query_with_history(question: str, conversation_history: list[dict] | None) -> str:
    """Rewrite a possibly-vague follow-up question into a standalone
    question, replacing pronouns like 'that' or 'it' with the actual
    topic - critical for retrieval to work on follow-ups."""
    if not conversation_history:
        return question

    client = _get_client()

    history_text = "\n".join(
        f"Q: {turn['question']}\nA: {turn['answer']}"
        for turn in conversation_history[-3:]
    )

    system_prompt = (
        "Given a conversation history and a new question, rewrite the new "
        "question as a standalone question that makes sense without the "
        "history - replace pronouns like 'that', 'it', or 'this' with the "
        "actual topic being discussed. If the question is already "
        "standalone, return it unchanged. Return ONLY the rewritten "
        "question, nothing else."
    )

    user_prompt = f"Conversation history:\n{history_text}\n\nNew question: {question}"

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=100,
        reasoning_effort="low",
    )

    return response.choices[0].message.content.strip()


def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    conversation_history: list[dict] | None = None,
) -> dict:
    """Generate a grounded answer using only the retrieved chunks as context.

    Returns a dict with "answer" and "sources" - citations are a
    structural guarantee from our own code (real page numbers), not
    dependent on the LLM choosing to mention them in its text.
    """
    client = _get_client()

    context_text = "\n\n---\n\n".join(
        f"[Page {chunk['page_number']}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    )

    system_prompt = (
        "You are a helpful study assistant. You will be given content "
        "pulled from a document and a question. Answer the question using "
        "ONLY information from the provided document content - do not use "
        "any outside knowledge, even if you know more about the topic. If "
        "the provided content doesn't contain enough information to answer "
        "the question, say so clearly rather than guessing or filling gaps "
        "with outside knowledge.\n\n"
        "Explain your answer clearly and simply, as if teaching someone "
        "encountering this topic for the first time. Break down complex "
        "ideas into plain language. You may rephrase or reorganize the "
        "content to make it clearer, but every fact you state must be "
        "traceable back to what was provided.\n\n"
        "Important wording rule: never use the word 'excerpt' or "
        "'excerpts' in your answer. Refer to it plainly as 'the document' "
        "or 'the document doesn't mention...' - the app cites sources "
        "separately by page number, so your answer text should read "
        "naturally without referencing excerpts at all.\n\n"
        "If earlier conversation turns are provided, use them to understand "
        "what the user is referring to (e.g. 'that' or 'it'), but your "
        "answer must still only state facts drawn from the document content "
        "given for THIS question."
    )

    user_prompt = (
        f"Document content:\n\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the document content above. "
        "Do not use the word 'excerpt' anywhere in your answer."
    )

    messages = [{"role": "system", "content": system_prompt}]

    if conversation_history:
        for turn in conversation_history[-3:]:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})

    messages.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        temperature=0.3,
        max_tokens=500,
        reasoning_effort="low",
    )

    answer_text = response.choices[0].message.content.strip()

    sources = [
        {
            "page_number": chunk["page_number"],
            "text": chunk["text"],
            "score": chunk["score"],
        }
        for chunk in retrieved_chunks
    ]

    return {"answer": answer_text, "sources": sources}


def generate_followup_suggestions(
    question: str,
    answer: str,
    retrieved_chunks: list[dict] | None = None,
) -> list[str]:
    """Ask the LLM for 2-3 short, natural follow-up questions - grounded
    in the actual retrieved document content, not generic curiosity.

    retrieved_chunks is passed in so the model can only suggest questions
    that are plausibly answerable from what's actually in the document,
    rather than the kind of question that just failed (e.g. asking for a
    specific number the document never states).
    """
    client = _get_client()

    context_text = "\n\n".join(chunk["text"] for chunk in (retrieved_chunks or []))

    system_prompt = (
        "Given a question, an answer, and the document content that "
        "answer was based on, suggest exactly 3 short, natural follow-up "
        "questions a curious learner might ask next. "
        "CRITICAL: only suggest questions that are plausibly answerable "
        "from the document content provided below - do not suggest "
        "questions that ask for specific facts, numbers, or details that "
        "are not present in that content. If the document only discusses "
        "a topic in general terms, suggest follow-ups that explore those "
        "general terms further, not ones that assume specific data exists. "
        "Each should be under 8 words where possible. Return ONLY the 3 "
        "questions, one per line - no numbering, no bullets, no quotes."
    )
    user_prompt = (
        f"Document content this answer was based on:\n\n{context_text}\n\n"
        f"Question: {question}\nAnswer: {answer}"
    )

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.5,
        max_tokens=120,
        reasoning_effort="low",
    )

    lines = response.choices[0].message.content.strip().split("\n")
    suggestions = [line.strip("-•* ").strip() for line in lines if line.strip()]
    return suggestions[:3]


def generate_document_topics(chunks: list[dict], sample_size: int = 6) -> list[str]:
    """Ask the LLM for a few example topics covered in this specific
    document, based on a sample of its chunks."""
    client = _get_client()

    sample_text = "\n\n".join(c["text"] for c in chunks[:sample_size])

    system_prompt = (
        "Given excerpts from a document, list exactly 3 short topic "
        "phrases (2-4 words each) that this document covers, suitable as "
        "example prompts for a reader deciding what to ask about. Return "
        "ONLY the 3 phrases, one per line, no numbering or bullets."
    )
    user_prompt = f"Document excerpts:\n\n{sample_text}"

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=60,
        reasoning_effort="low",
    )

    lines = response.choices[0].message.content.strip().split("\n")
    topics = [line.strip("-•* ").strip() for line in lines if line.strip()]
    return topics[:3]


if __name__ == "__main__":
    from pathlib import Path

    from document_loader import chunk_pages, extract_pages_from_pdf
    from embedder import build_document_store
    from retriever import retrieve_top_chunks

    project_root = Path(__file__).resolve().parent.parent
    test_pdf = project_root / "data" / "artificial_intelligence_and_machine_learning.pdf"

    pages = extract_pages_from_pdf(test_pdf)
    chunks = chunk_pages(pages)
    store = build_document_store(chunks)

    question = "What is overfitting?"
    retrieved = retrieve_top_chunks(question, store, top_k=3)
    result = generate_answer(question, retrieved)

    print(f"Answer:\n{result['answer']}\n")

    suggestions = generate_followup_suggestions(question, result["answer"], retrieved)
    print("Suggested follow-ups:")
    for s in suggestions:
        print(f"  - {s}")