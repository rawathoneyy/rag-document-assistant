"""Streamlit chat interface for the RAG Document Assistant."""

import sys
from pathlib import Path

import streamlit as st

APP_NAME = "DocQuery"

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from document_loader import chunk_pages, extract_pages_from_pdf

st.set_page_config(
    page_title=f"{APP_NAME} — Document Assistant",
    page_icon="📖",
    layout="wide",
)

# ---------------- Custom CSS ----------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 3.5rem;
        max-width: 1200px;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: #0d0d12;
        border-right: 1px solid rgba(255,255,255,0.06);
        min-width: 260px !important;
        max-width: 420px !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
        padding-left: 1.25rem;
        padding-right: 1.25rem;
    }
    .folio-logo {
        font-size: 19px !important;
        margin-bottom: 1rem;
    }
    .folio-logo b { color: #fff; }
    .folio-sub { color: rgba(255,255,255,0.45); font-size: 14.5px; margin-left: 4px; }

    [data-testid="stFileUploaderDropzone"] {
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
    }
    [data-testid="stFileUploader"] > section > div:first-child:not(:has([data-testid="stFileUploaderFile"])) [data-testid="stFileUploaderDropzone"] {
        border: 1.5px dashed rgba(255,255,255,0.18) !important;
        background: rgba(255,255,255,0.02) !important;
        padding: 1rem !important;
        border-radius: 12px !important;
    }

    [data-testid="stFileUploaderFile"] {
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        padding: 10px 12px !important;
        margin-top: 10px !important;
    }
    [data-testid="stFileUploaderFile"] div {
        font-size: 15px !important;
    }
    [data-testid="stFileUploaderFile"] small {
        color: rgba(255,255,255,0.45) !important;
        font-size: 13px !important;
    }
    [data-testid="stFileUploaderDeleteBtn"] button {
        color: rgba(255,255,255,0.6) !important;
    }

    .chip-label {
        font-size: 14.5px !important;
        color: rgba(255,255,255,0.45);
        margin: 1.25rem 0 0.6rem;
    }

    section[data-testid="stSidebar"] [data-testid="stButton"] {
        width: 100%;
    }
    section[data-testid="stSidebar"] [data-testid="stButton"] button {
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.02);
        color: rgba(255,255,255,0.85);
        border-radius: 8px;
        font-size: 15px !important;
        padding: 9px 12px;
        width: 100%;
        white-space: normal;
        text-align: left;
        line-height: 1.4;
    }
    section[data-testid="stSidebar"] [data-testid="stButton"] button:hover {
        border-color: #7c6cf0;
        color: #fff;
    }

    .sidebar-footer {
        font-size: 13px;
        color: rgba(255,255,255,0.35);
        line-height: 1.5;
        margin-top: 2rem;
    }

    .st-key-new_followup_chips,
    [class*="st-key-hist_"] {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .st-key-new_followup_chips [data-testid="stButton"],
    [class*="st-key-hist_"] [data-testid="stButton"] {
        width: auto;
    }
    .st-key-new_followup_chips [data-testid="stButton"] button,
    [class*="st-key-hist_"] [data-testid="stButton"] button {
        white-space: nowrap;
        font-size: 15.5px;
    }

    .folio-header-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 30px; height: 30px;
        border-radius: 8px;
        background: rgba(124,108,240,0.15);
        margin-right: 8px;
        font-size: 15px;
    }

    .st-key-new_chat_btn button {
        font-size: 14px !important;
        padding: 5px 16px !important;
        min-height: 0 !important;
    }

    /* ---- Chat messages ---- */
    /* Primary approach: force the row to flex-end */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        box-shadow: none !important;
        width: 100% !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        display: flex !important;
        flex-direction: row-reverse !important;
        justify-content: flex-start !important;
        width: 100% !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageAvatarUser"] {
        display: none;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
        margin-left: 0 !important;
        margin-right: 0 !important;
        background: linear-gradient(135deg, #6d5ef8, #8b5cf6) !important;
        border-radius: 16px !important;
        padding: 10px 16px !important;
        flex-grow: 0 !important;
        flex-shrink: 0 !important;
        width: fit-content !important;
        max-width: 70% !important;
        color: #fff !important;
        font-size: 19px !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        display: flex !important;
        width: 100% !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
        font-size: 19px;
        line-height: 1.8;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] h1 {
        font-size: 27px !important;
        font-weight: 700 !important;
        margin: 0.6em 0 0.3em !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] h2 {
        font-size: 23px !important;
        font-weight: 700 !important;
        margin: 0.6em 0 0.3em !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] h3 {
        font-size: 20px !important;
        font-weight: 600 !important;
        margin: 0.5em 0 0.25em !important;
    }

    .source-tags { margin-top: 10px; }
    .source-tag {
        display: inline-block;
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 13.5px;
        color: rgba(255,255,255,0.6);
        margin-right: 6px;
        font-family: monospace;
    }
    .source-label {
        font-size: 14px;
        color: rgba(255,255,255,0.4);
        margin-right: 6px;
    }

    /* ---- Mobile ---- */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1.5rem;
            padding-left: 0.75rem;
            padding-right: 0.75rem;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
            max-width: 88% !important;
            font-size: 16.5px !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
            font-size: 16.5px;
        }
        .st-key-new_followup_chips [data-testid="stButton"] button,
        [class*="st-key-hist_"] [data-testid="stButton"] button {
            font-size: 13.5px;
            padding: 6px 10px;
        }
        .folio-header-icon {
            width: 26px; height: 26px;
            font-size: 13px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Session state ----------------
for key, default in [
    ("document_store", None),
    ("processed_filename", None),
    ("conversation_history", []),
    ("pending_question", None),
    ("document_topics", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def format_source_tags(sources: list[dict]) -> str:
    """Real page-number citations, deduped and sorted."""
    seen_pages = sorted({s["page_number"] for s in sources})
    return "".join(f'<span class="source-tag">Page {p}</span>' for p in seen_pages)


# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown(
        f'<div class="folio-logo">📖 <b>{APP_NAME}</b><span class="folio-sub">Document assistant</span></div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload document", type=["pdf"], label_visibility="collapsed")
    st.caption("PDF up to 25 MB")

    if uploaded_file is not None:
        if st.session_state.processed_filename != uploaded_file.name:
            with st.spinner("Reading and indexing document..."):
                from embedder import build_document_store
                from generator import generate_document_topics

                pages = extract_pages_from_pdf(uploaded_file)
                if not pages:
                    st.error("Couldn't extract any text from this PDF.")
                else:
                    chunks = chunk_pages(pages)
                    store = build_document_store(chunks)
                    st.session_state.document_store = store
                    st.session_state.processed_filename = uploaded_file.name
                    st.session_state.conversation_history = []
                    st.session_state.document_topics = generate_document_topics(chunks)

    if st.session_state.document_topics:
        st.markdown('<div class="chip-label">You can ask about</div>', unsafe_allow_html=True)
        for topic in st.session_state.document_topics:
            if st.button(topic, key=f"topic_{topic}"):
                st.session_state.pending_question = f"Tell me about {topic.lower()}"
                st.rerun()

    st.markdown(
        '<div class="sidebar-footer">Answers are generated from your document and include source references.</div>',
        unsafe_allow_html=True,
    )

# ---------------- Header ----------------
header_col1, header_col2 = st.columns([5, 1])
with header_col1:
    st.markdown(
        f'<span class="folio-header-icon">📖</span>'
        f'<b style="font-size:17px;">{APP_NAME}</b> '
        '<span style="color:rgba(255,255,255,0.45); font-size:14.5px;">Document assistant</span>',
        unsafe_allow_html=True,
    )
with header_col2:
    if st.button("New chat", use_container_width=True, key="new_chat_btn"):
        st.session_state.conversation_history = []
        st.rerun()

# ---------------- Empty state ----------------
if st.session_state.document_store is None:
    st.markdown(
        """
        <div style="display:flex; flex-direction:column; align-items:center;
                    justify-content:center; padding:4rem 1.5rem; text-align:center;
                    gap:8px; min-height:280px;">
            <p style="font-size:16px; font-weight:600; color:#fff;">
                Upload a document to get started
            </p>
            <p style="font-size:14px; color:rgba(255,255,255,0.5); max-width:340px; line-height:1.5;">
                Ask questions and get answers grounded in what your document
                actually says, with sources for every answer.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


def render_message(role: str, content: str, sources: list | None = None, suggestions: list | None = None, key_prefix: str = ""):
    avatar = "📄" if role == "assistant" else None
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)
        if sources:
            st.markdown(
                f'<div class="source-tags"><span class="source-label">Sources</span>{format_source_tags(sources)}</div>',
                unsafe_allow_html=True,
            )
        if suggestions:
            with st.container(key=f"{key_prefix}_followup_chips"):
                for suggestion in suggestions:
                    if st.button(suggestion, key=f"{key_prefix}_{suggestion}"):
                        st.session_state.pending_question = suggestion
                        st.rerun()


# ---------------- Render history ----------------
history = st.session_state.conversation_history
for i, turn in enumerate(history):
    is_last = i == len(history) - 1
    render_message("user", turn["question"])
    render_message(
        "assistant",
        turn["answer"],
        sources=turn["sources"],
        suggestions=turn["suggestions"] if is_last else None,
        key_prefix=f"hist_{i}",
    )

# ---------------- Input ----------------
typed_question = st.chat_input("Ask anything about this document...")
question = st.session_state.pending_question or typed_question
st.session_state.pending_question = None

if question:
    render_message("user", question)

    with st.chat_message("assistant", avatar="📄"):
        with st.spinner("Thinking..."):
            from generator import (
                generate_answer,
                generate_followup_suggestions,
                rewrite_query_with_history,
            )
            from retriever import retrieve_top_chunks

            search_question = rewrite_query_with_history(question, history)
            retrieved = retrieve_top_chunks(search_question, st.session_state.document_store, top_k=3)
            result = generate_answer(question, retrieved, conversation_history=history)
            suggestions = generate_followup_suggestions(question, result["answer"], retrieved)

        st.markdown(result["answer"])
        st.markdown(
            f'<div class="source-tags"><span class="source-label">Sources</span>{format_source_tags(result["sources"])}</div>',
            unsafe_allow_html=True,
        )
        with st.container(key="new_followup_chips"):
            for suggestion in suggestions:
                if st.button(suggestion, key=f"new_{suggestion}"):
                    st.session_state.pending_question = suggestion
                    st.rerun()

    st.session_state.conversation_history.append(
        {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "suggestions": suggestions,
        }
    )