"""Load and extract text from a PDF document."""

from pathlib import Path

import pdfplumber


def extract_pages_from_pdf(file_path) -> list[dict]:
    """Open a PDF and return each page's text with its page number attached.

    Accepts either a file path (str/Path) or an in-memory file-like
    object (e.g. from Streamlit's file_uploader).
    """
    pdf_path = Path(file_path) if isinstance(file_path, (str, Path)) else file_path

    pages: list[dict] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text()
            if not raw_text or not raw_text.strip():
                continue
            pages.append({"page_number": page_number, "text": raw_text.strip()})

    return pages


def extract_text_from_pdf(file_path) -> str:
    """Kept for backward compatibility (quick __main__ testing) - joins
    all page text with no page metadata attached."""
    pages = extract_pages_from_pdf(file_path)
    return "\n\n".join(p["text"] for p in pages)


def chunk_pages(pages: list[dict], chunk_size: int = 800, overlap: int = 150) -> list[dict]:
    """Split each page's text into overlapping chunks independently, so
    every chunk can be traced back to exactly one real page number -
    that's what makes an honest 'Page 4' citation possible.
    """
    chunks: list[dict] = []

    for page in pages:
        text = page["text"]
        page_number = page["page_number"]
        start = 0
        text_length = len(text)

        if text_length == 0:
            continue

        while start < text_length:
            end = start + chunk_size
            chunk_text_value = text[start:end].strip()
            if chunk_text_value:
                chunks.append({"text": chunk_text_value, "page_number": page_number})
            start += chunk_size - overlap

    return chunks


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Kept for backward compatibility / standalone testing - no page
    metadata, same behavior as the original version."""
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    test_pdf = project_root / "data" / "artificial_intelligence_and_machine_learning.pdf"

    pages = extract_pages_from_pdf(test_pdf)
    print(f"Extracted {len(pages)} pages.")

    chunks = chunk_pages(pages)
    print(f"\nSplit into {len(chunks)} chunks.")
    print(f"\nFirst chunk (page {chunks[0]['page_number']}):\n{chunks[0]['text']}")