"""
main.py
Milestone 3: Query interface for GT Course Catalog RAG system.

Usage:
    python main.py

Runs an interactive CLI loop. Type a question, get a grounded answer
with source citations. Type 'quit' to exit.
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "gt_courses"
EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5


# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(query: str, model: SentenceTransformer, collection) -> list[dict]:
    """Embed the query and retrieve top-k most similar course chunks."""
    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({
            "text": doc,
            "department": meta.get("department", ""),
            "course_number": meta.get("course_number", ""),
            "course_title": meta.get("course_title", ""),
        })
    return chunks


# ── Generation ────────────────────────────────────────────────────────────────
def generate(query: str, chunks: list[dict], client: Groq) -> str:
    """Send retrieved chunks + query to Groq LLM and return a grounded answer."""

    context = ""
    for i, chunk in enumerate(chunks, 1):
        context += f"[Source {i}: {chunk['department']} {chunk['course_number']} — {chunk['course_title']}]\n"
        context += chunk["text"] + "\n\n"

    prompt = f"""You are a helpful Georgia Tech course advisor. Answer the student's question using ONLY the course information provided below. Do not use any knowledge outside of these sources.

For every fact you state, cite the source like this: (Source 1) or (Source 2).
If the provided sources don't contain enough information to answer the question, say so clearly.

--- Course Information ---
{context}
--- End of Course Information ---

Student question: {query}

Answer:"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found. Make sure your .env file is set up.")
        return

    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    print("Connecting to ChromaDB...")
    client_db = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client_db.get_or_create_collection(COLLECTION_NAME)

    groq_client = Groq(api_key=api_key)

    count = collection.count()
    if count == 0:
        print("Warning: ChromaDB is empty. Run ingest.py first.")
        return

    print(f"Ready! {count} course chunks loaded.")
    print("Type your question (or 'quit' to exit).\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        chunks = retrieve(query, model, collection)
        answer = generate(query, chunks, groq_client)

        print(f"\nAssistant: {answer}\n")
        print("Sources:")
        for i, chunk in enumerate(chunks, 1):
            print(f"  [{i}] {chunk['department']} {chunk['course_number']} — {chunk['course_title']}")
        print()


if __name__ == "__main__":
    main()
