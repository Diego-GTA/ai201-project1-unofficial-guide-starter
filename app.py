"""
app.py
Milestone 5: Gradio web interface for GT Course Catalog RAG system.

Usage:
    python app.py
Then open http://localhost:7860 in your browser.
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

# Configuration
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "gt_courses"
EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5

# Load models once at startup
print("Loading embedding model...")
model = SentenceTransformer(EMBED_MODEL)

print("Connecting to ChromaDB...")
db_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = db_client.get_or_create_collection(COLLECTION_NAME)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
print(f"Ready! {collection.count()} course chunks loaded.")


def retrieve(query):
    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "department": meta.get("department", ""),
            "course_number": meta.get("course_number", ""),
            "course_title": meta.get("course_title", ""),
            "distance": round(dist, 3),
        })
    return chunks


def generate(query, chunks):
    context = ""
    for i, chunk in enumerate(chunks, 1):
        context += f"[Source {i}: {chunk['department']} {chunk['course_number']} - {chunk['course_title']}]\n"
        context += chunk["text"] + "\n\n"

    prompt = f"""You are a helpful Georgia Tech course advisor. Answer the student's question using ONLY the course information provided below. Do not use any knowledge outside of these sources.

For every fact you state, cite the source like this: (Source 1) or (Source 2).
If the provided sources don't contain enough information to answer the question, say "I don't have enough information on that based on the available course catalog."

--- Course Information ---
{context}
--- End of Course Information ---

Student question: {query}

Answer:"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


def ask(question):
    if not question.strip():
        return "Please enter a question.", ""
    chunks = retrieve(question)
    answer = generate(question, chunks)
    sources = "\n".join(
        f"[{i}] {c['department']} {c['course_number']} — {c['course_title']} (distance: {c['distance']})"
        for i, c in enumerate(chunks, 1)
    )
    return answer, sources


# Gradio UI
with gr.Blocks(title="GT Course Catalog Guide") as demo:
    gr.Markdown("# GT Course Catalog Unofficial Guide")
    gr.Markdown("Ask questions about Georgia Tech courses across CS, ECE, AE, ISyE, and more.")

    with gr.Row():
        inp = gr.Textbox(
            label="Your question",
            placeholder="e.g. What CS courses cover machine learning?",
            lines=2,
        )

    btn = gr.Button("Ask", variant="primary")

    with gr.Row():
        answer = gr.Textbox(label="Answer", lines=10)
        sources = gr.Textbox(label="Retrieved Sources", lines=10)

    btn.click(ask, inputs=inp, outputs=[answer, sources])
    inp.submit(ask, inputs=inp, outputs=[answer, sources])

    gr.Examples(
        examples=[
            ["What CS courses cover machine learning?"],
            ["Which departments offer courses on optimization?"],
            ["What should I take after intro programming if I'm interested in AI?"],
            ["What AE courses are available at the 4000 level?"],
        ],
        inputs=inp,
    )

demo.launch()
