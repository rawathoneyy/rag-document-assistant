# DocQuery — RAG-Powered Document Assistant

Ask questions about any PDF and get answers grounded entirely in that document — with real page-number citations, not guesses or hallucinations.

## What it does

Upload a PDF, then ask questions about it in a chat interface. DocQuery retrieves the most relevant sections of the document for each question, generates an answer using only that content, and cites the exact page numbers the answer came from. If the document doesn't contain enough information to answer, it says so explicitly rather than filling gaps with outside knowledge.

## Why it's different from a basic "chat with your PDF" demo

- **Real citations, not placeholders.** Answers are grounded to actual page numbers, tracked from PDF extraction through chunking, embedding, and retrieval — not an arbitrary "chunk 1 of 3" label.
- **Conversation memory that actually works for retrieval.** Follow-up questions like "how can I detect that early?" are rewritten into standalone queries before retrieval runs — tested and confirmed necessary: a vague follow-up scored ~0.28 similarity against the right content before this fix, and ~0.51 after.
- **Grounded follow-up suggestions.** Suggested next questions are generated from the retrieved content itself, not generic curiosity, so the app never suggests a question it can't actually answer.
- **Honest failure mode.** Verified through testing: when the source document doesn't cover something (e.g. a document that defines overfitting but gives no worked example), the app explicitly says so rather than quietly blending in outside knowledge to seem more helpful.

## How it works

1. **Extraction** (`document_loader.py`) — `pdfplumber` extracts text page by page, preserving which page number each block of text came from.
2. **Chunking** — each page's text is split into overlapping ~800-character chunks (150-character overlap), independently per page, so every chunk stays traceable to exactly one page.
3. **Embedding** (`embedder.py`) — chunks are embedded in one batched call via Cohere's `embed-english-v3.0` model, using `input_type="search_document"`. Queries are embedded separately with `input_type="search_query"`, since Cohere tunes embeddings differently depending on which side of the search they're used for.
4. **Retrieval** (`retriever.py`) — a plain in-memory store compares the query embedding against every chunk via manual cosine similarity and returns the top matches. No vector database — deliberately, see below.
5. **Generation** (`generator.py`) — Groq's `openai/gpt-oss-20b` generates an answer constrained to only the retrieved content. Source page numbers are returned as structured data alongside the answer, independent of whatever the model's own text says — citation is a guarantee from the code, not something the model has to remember to mention.
6. **Interface** (`app.py`) — a Streamlit chat UI with conversation history, page-number source tags, and dynamic topic/follow-up suggestion chips.

## Deliberate architecture choices

- **No FAISS, no LangChain.** Both are compiled/heavier dependencies; a manual in-memory cosine similarity search is completely sufficient at single-document scale, and avoids a class of environment dependency issues encountered on a prior project.
- **Strictly grounded, not hybrid.** The app never falls back to the LLM's general knowledge when the document doesn't cover something — verified honest behavior over surface-level helpfulness.

## Tech stack

Python · Streamlit · Cohere (embeddings) · Groq (generation) · pdfplumber

## Setup

\`\`\`bash
git clone <your-repo-url>
cd rag-document-assistant
python -m venv venv
venv\Scripts\activate  # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
\`\`\`

Create a `.env` file in the project root:
\`\`\`
COHERE_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
\`\`\`

## Running locally

\`\`\`bash
streamlit run app.py
\`\`\`

Opens at `http://localhost:8501`.

## Project structure

\`\`\`
rag-document-assistant/
├── app.py                    # Streamlit UI
├── src/
│   ├── document_loader.py     # Page-aware PDF extraction + chunking
│   ├── embedder.py            # Cohere embeddings + document store
│   ├── retriever.py           # Cosine similarity search
│   └── generator.py           # Groq-based answer + suggestion generation
├── data/                      # Sample PDFs for testing
├── .streamlit/
│   └── config.toml            # Theme config
├── requirements.txt
└── .env                       # API keys (not committed)
\`\`\`

## Known limitations

- Citations are page-level, not section-level — PDFs don't reliably expose heading structure, so page numbers are the honest, achievable granularity.
- Retrieval is a flat in-memory cosine similarity search — fine at single-document scale, but wouldn't scale to a large multi-document corpus without a real vector index.
- Chunking is character-based, not token-based — a reasonable approximation for English text, not an exact token count.