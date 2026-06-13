"""
ingest.py
Milestone 3: Document ingestion pipeline for GT Course Catalog RAG system.

Usage:
    python ingest.py

Reads all PDFs from DOCS_DIR, extracts and cleans text, chunks by course entry,
embeds with sentence-transformers, and loads into ChromaDB.
"""

import os
import re
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer

# ── Configuration ────────────────────────────────────────────────────────────
DOCS_DIR = r"C:\Users\dseve\Documents\Milestone1codepath"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "gt_courses"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Lines to strip from raw PDF text (nav boilerplate)
BOILERPLATE_PATTERNS = [
    r"Skip to Content",
    r"AZ Index",
    r"Catalog Home",
    r"GA Tech Home",
    r"Georgia Tech logo",
    r"Toggle menu",
    r"Colleges and Schools",
    r"College of Computing",
    r"College of Design",
    r"College of Engineering",
    r"College of Sciences",
    r"Ivan Allen",
    r"Scheller College",
    r"Print Options",
    r"Search catalog",
    r"2025-2026 Edition",
    r"^\s*\d+\s*$",           # lone page numbers
    r"catalog\.gatech\.edu",
    r"gatech\.edu",
]


# ── Step 1: Extract text from PDFs ───────────────────────────────────────────
def extract_text(pdf_path: str) -> str:
    """Extract all text from a PDF file using pdfplumber."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n".join(pages)


# ── Step 2: Clean boilerplate ─────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Remove nav menus, headers, footers, and other boilerplate."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        skip = False
        for pattern in BOILERPLATE_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                skip = True
                break
        if not skip and line.strip():
            cleaned.append(line.strip())
    return "\n".join(cleaned)


# ── Step 3: Chunk by course entry ─────────────────────────────────────────────
def chunk_by_course(text: str, department: str) -> list[dict]:
    """
    Split cleaned text into one chunk per course entry.
    Course entries start with a pattern like: CS 1301, ECE 2026, AE 4373
    """
    # Match lines that start a new course: 2-4 uppercase letters + space + 4 digits
    course_pattern = re.compile(r"^([A-Z]{2,4}\s+\d{4}[A-Z]?)\b", re.MULTILINE)

    matches = list(course_pattern.finditer(text))
    chunks = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()

        if len(chunk_text) < 30:  # skip empty/tiny fragments
            continue

        course_number = match.group(1).strip()
        # Extract title: everything on the first line after the course number
        first_line = chunk_text.split("\n")[0]
        title = first_line[len(course_number):].strip()

        chunks.append({
            "text": chunk_text,
            "department": department,
            "course_number": course_number,
            "course_title": title,
            "id": f"{department}_{course_number.replace(' ', '_')}_{i}",

        })

    return chunks


# ── Step 4: Embed and load into ChromaDB ──────────────────────────────────────
def load_into_chroma(chunks: list[dict], model: SentenceTransformer, collection):
    """Embed chunks and upsert into ChromaDB collection."""
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{
            "department": c["department"],
            "course_number": c["course_number"],
            "course_title": c["course_title"],
        } for c in chunks],
    )
    return len(chunks)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    print("Setting up ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    pdf_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in {DOCS_DIR}")
        return

    total_chunks = 0

    for filename in pdf_files:
        department = filename.replace(".pdf", "").upper()
        path = os.path.join(DOCS_DIR, filename)

        print(f"\nProcessing {filename}...")
        raw = extract_text(path)
        cleaned = clean_text(raw)
        chunks = chunk_by_course(cleaned, department)

        print(f"  Found {len(chunks)} course chunks")

        # Print 2 sample chunks so you can verify quality
        for chunk in chunks[:2]:
            print(f"\n  --- Sample chunk ---")
            print(f"  Course: {chunk['course_number']} {chunk['course_title']}")
            print(f"  Text preview: {chunk['text'][:200]}")

        loaded = load_into_chroma(chunks, model, collection)
        total_chunks += loaded
        print(f"  Loaded {loaded} chunks into ChromaDB")

    print(f"\nDone! Total chunks in database: {total_chunks}")
    print(f"ChromaDB stored at: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
