## Domain

Georgia Tech's official course catalog lists every course offered across departments, but it's hard to navigate — you can only browse one department at a time, and there's no way to ask cross-department questions like "which courses cover machine learning?" or "what should I take after finishing intro CS?" This system makes GT course knowledge searchable and answerable across 10 departments, helping students discover courses by topic and plan their academic paths without manually digging through each department page.

## Documents

10 PDF files saved from the Georgia Tech course catalog (2025–2026 edition), one per department:

- `cs.pdf` — CS department: https://catalog.gatech.edu/coursesaz/cs/
- `cse.pdf` — Computational Science & Engineering: https://catalog.gatech.edu/coursesaz/cse/
- `ece.pdf` — Electrical & Computer Engineering: https://catalog.gatech.edu/coursesaz/ece/
- `chem.pdf` — Chemistry: https://catalog.gatech.edu/coursesaz/chem/
- `chbe.pdf` — Chemical & Biomolecular Engineering: https://catalog.gatech.edu/coursesaz/chbe/
- `isye.pdf` — Industrial & Systems Engineering: https://catalog.gatech.edu/coursesaz/isye/
- `cee.pdf` — Civil & Environmental Engineering: https://catalog.gatech.edu/coursesaz/cee/
- `cos.pdf` — College of Sciences: https://catalog.gatech.edu/coursesaz/cos/
- `biol.pdf` — Biology: https://catalog.gatech.edu/coursesaz/biol/
- `ae.pdf` — Aerospace Engineering: https://catalog.gatech.edu/coursesaz/ae/

Each file contains course numbers, titles, credit hours, prerequisites, and full course descriptions for all courses in that department.

## Chunking Strategy

Each course entry (number + title + description) is a natural, self-contained unit, so chunking follows document structure rather than fixed character counts — one chunk per course.

Each chunk contains the course number, title, credit hours, prerequisites, and full description. Chunk size is approximately 100–300 tokens, with no overlap, because each course is logically bounded — there's no risk of a key fact spanning two courses.

This fits the documents because course descriptions are short (50–200 words each) and self-contained. Grouping multiple courses into one chunk would dilute retrieval: a query about "machine learning" might pull a chunk that also covers unrelated courses, giving the LLM noise to work through. Splitting a single course across two chunks would break the connection between its prerequisites and description, making answers incomplete. One course per chunk keeps retrieval precise.

Chunks that are too small would look like: only the course number and title, no description — the LLM can't answer "what does this course cover?" Chunks that are too large would look like: three or four courses merged together — the LLM returns answers that mix information from multiple courses without clearly attributing which fact belongs to which.

## Retrieval Approach

Embedding model: `all-MiniLM-L6-v2` via `sentence-transformers` — runs locally, no API key, no rate limits. Vector store: ChromaDB, also local.

Top-k: 5 chunks per query. Five retrieved courses gives the LLM enough context to compare options and make a recommendation without overwhelming it. Too few (1–2) risks missing relevant courses when a query could match several. Too many (10+) fills the context window with marginally relevant results and increases the chance the LLM mixes up which facts came from which course.

Semantic search works here because students phrase questions differently than catalog text. A student asking "what covers neural networks?" won't match exact words in a description that says "deep learning, backpropagation, and convolutional architectures" — but the embeddings will be close in vector space because the concepts are related.

If deploying for real users with no cost constraint, the tradeoffs to consider would be: `text-embedding-3-large` (OpenAI) for higher accuracy on technical terminology; `bge-large-en` for a stronger open-source option with better retrieval benchmarks; and context length — all-MiniLM-L6-v2 maxes out at 256 tokens, which is fine for single course entries but would be a problem if chunks were larger. Multilingual support is not needed since all catalog content is in English.

## Evaluation Plan

1. **What CS courses cover machine learning?**
   Expected: CS 4641 (Machine Learning) and CS 7641 at minimum, with descriptions mentioning classification, regression, or statistical learning.

2. **What are the prerequisites for CS 3600?**
   Expected: CS 2110 and either MATH 3012 or CS 2051, as listed in the catalog entry for CS 3600.

3. **Which departments offer courses on optimization?**
   Expected: ISyE and CS at minimum (ISyE has multiple optimization courses; CS has CS 4540). Response should cite at least two departments with specific course numbers.

4. **What AE courses are available at the 4000 level?**
   Expected: A list of AE 4XXX courses drawn from the ae.pdf content, with titles and brief descriptions.

5. **What should I take after intro programming if I'm interested in AI?**
   Expected: Should recommend CS 2110 or CS 2340 as a next step, then CS 3600 or CS 4641 for AI — grounded in the prerequisite chains visible in the catalog, not general knowledge.

## Anticipated Challenges

**1. PDF extraction noise.** PDFs saved from the browser may contain navigation menus, page headers, footers, and boilerplate ("Georgia Tech Catalog," "2025–2026 Edition") mixed in with course content. If this text isn't stripped before chunking, some chunks will embed nav text instead of course descriptions, and retrieval will return garbage for those entries.

**2. Course boundary detection.** The catalog doesn't use a perfectly consistent delimiter between courses across all departments. Detecting where one course ends and the next begins will likely require matching on course number patterns (e.g., a line starting with `CS 1234`). If the pattern fails on edge cases — cross-listed courses, special topics sections, or unusual formatting — chunks will either merge multiple courses or split one course mid-description, breaking retrieval for those entries.

## AI Tool Plan

- **PDF ingestion:** I'll give Claude the Documents section of this file and ask it to write a script using `pdfplumber` that opens each PDF, extracts text page by page, strips navigation boilerplate, and saves cleaned plain text to a `.txt` file per department.

- **Chunking:** I'll give Claude the Chunking Strategy section plus a sample of the cleaned text from one department file and ask it to implement a `chunk_documents()` function that splits the text into one chunk per course using course number patterns (e.g., `^[A-Z]+ \d{4}`) as delimiters, with department and course number stored as metadata.

- **Embedding + vector store:** I'll give Claude the Retrieval Approach section and ask it to write a script that embeds each chunk using `all-MiniLM-L6-v2` and upserts into a ChromaDB collection, storing department and course number as metadata fields.

- **Generation:** I'll give Claude the Evaluation Plan section and ask it to write a `query_rag()` function that takes a user question, retrieves the top-5 chunks from ChromaDB, and sends them to Groq's `llama-3.3-70b-versatile` with a prompt that instructs it to answer using only the provided context and to cite the course number and department for every fact it uses.

- **Query interface:** I'll give Claude a description of a simple CLI loop and ask it to write a `main.py` that runs a `while True` input loop, calls `query_rag()`, and prints the answer and sources in a readable format.
