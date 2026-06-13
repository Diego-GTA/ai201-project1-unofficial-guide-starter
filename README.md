# GT Course Catalog Unofficial Guide

A RAG (Retrieval-Augmented Generation) system that makes Georgia Tech's course catalog searchable and answerable. Ask plain-language questions about courses across 10 departments and get grounded, cited answers drawn from the official GT catalog.

## Domain and Document Sources

**Domain:** Georgia Tech course offerings across 10 departments (2025-2026 catalog).

This knowledge is valuable because the official catalog only lets you browse one department at a time with no cross-department search. Students can't easily ask "which departments offer AI courses?" or "what should I take after intro CS if I'm interested in machine learning?" without manually reading through every department page.

**Sources:**

| File | Department | URL |
|------|-----------|-----|
| cs.pdf | Computer Science | https://catalog.gatech.edu/coursesaz/cs/ |
| cse.pdf | Computational Science & Engineering | https://catalog.gatech.edu/coursesaz/cse/ |
| ece.pdf | Electrical & Computer Engineering | https://catalog.gatech.edu/coursesaz/ece/ |
| chem.pdf | Chemistry | https://catalog.gatech.edu/coursesaz/chem/ |
| chbe.pdf | Chemical & Biomolecular Engineering | https://catalog.gatech.edu/coursesaz/chbe/ |
| isye.pdf | Industrial & Systems Engineering | https://catalog.gatech.edu/coursesaz/isye/ |
| cee.pdf | Civil & Environmental Engineering | https://catalog.gatech.edu/coursesaz/cee/ |
| cos.pdf | College of Sciences | https://catalog.gatech.edu/coursesaz/cos/ |
| biol.pdf | Biology | https://catalog.gatech.edu/coursesaz/biol/ |
| ae.pdf | Aerospace Engineering | https://catalog.gatech.edu/coursesaz/ae/ |

## Chunking Strategy

**Approach:** One chunk per course entry. Each chunk contains the course number, title, credit hours, prerequisites, and full description.

**Chunk size:** ~100–300 tokens per chunk. No overlap.

**Why this fits the documents:** Course descriptions are short and self-contained (50–200 words each). Grouping multiple courses into one chunk would dilute retrieval — a query about "machine learning" might pull a chunk covering ML plus unrelated courses. Splitting a single course across chunks would break the connection between its prerequisites and description.

The chunker uses a regex pattern (`^[A-Z]{2,4}\s+\d{4}`) to detect course number boundaries and splits there.

**Sample Chunks:**

1. **AE 4361** (ae.pdf)
   > AE 4361. Space Flight Operations. 3 Credit Hours. Topics of current importance offered in collaboration with an approved aerospace agency or industry partner. Focus on operational aspects of space missions.

2. **CS 7641** (cs.pdf)
   > CS 7641. Machine Learning. 3 Credit Hours. Machine learning techniques for classification, regression, and clustering. Topics include supervised and unsupervised learning, neural networks, and model evaluation.

3. **ISYE 4135** (isye.pdf)
   > ISYE 4135. Optimization Foundations for Machine Learning and Artificial Intelligence. 3 Credit Hours. Covers mathematical optimization methods underlying modern ML and AI systems.

4. **ECE 3251** (ece.pdf)
   > ECE 3251. Optimization for Information Systems. 3 Credit Hours. Second course in a culminating design sequence. Student teams design, build, and demonstrate an electrical engineering system.

5. **CS 3600** (cs.pdf)
   > CS 3600. Introduction to Artificial Intelligence. 3 Credit Hours. Introduction to fundamental concepts of AI including search, knowledge representation, planning, and machine learning.

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers` — runs locally, no API key, no rate limits, 384-dimension embeddings.

**Production tradeoffs:** For a real deployment I would consider:
- **`text-embedding-3-large` (OpenAI):** Higher accuracy on technical terminology, but costs money per call and requires an API key
- **`bge-large-en`:** Stronger open-source alternative with better retrieval benchmarks than MiniLM
- **Context length:** `all-MiniLM-L6-v2` maxes out at 256 tokens — fine for single course entries, but a problem for larger chunks
- **Multilingual:** Not needed here since all catalog content is in English

## Retrieval Test Results

**Query 1:** "What CS courses cover machine learning?"

Top retrieved chunks:
- CS 7641 — Machine Learning (distance: 0.21)
- CS 7651 — Human and Machine Learning (distance: 0.29)
- CSE 7752 — Scientific Machine Learning (distance: 0.31)

*Why relevant:* The query encodes the concept of ML. CS 7641 is titled "Machine Learning" so its embedding is extremely close to the query vector. CS 7651 shares the "machine learning" phrase. Semantic search correctly surfaces these even though the query didn't use GT-specific course numbers.

---

**Query 2:** "Which departments offer courses on optimization?"

Top retrieved chunks:
- ECE 3251 — Optimization for Information Systems (distance: 0.28)
- ISYE 4135 — Optimization Foundations for ML and AI (distance: 0.30)
- ISYE 7683 — Advanced Nonlinear Programming (distance: 0.42)

*Why relevant:* The word "optimization" appears in the titles and descriptions of these chunks, so their embeddings cluster near the query. The system correctly retrieved courses from two different departments, enabling a cross-department answer.

---

**Query 3:** "What are the prerequisites for CS 3600?"

Top retrieved chunks: CS 6301, CS 4455, CS 1301R, CS 4675, CS 4911 (all irrelevant — see failure case)

*Why this failed:* See Failure Case section below.

## Grounded Generation

Grounding is enforced through the prompt template passed to the LLM. The system prompt explicitly states:

> "Answer the student's question using ONLY the course information provided below. Do not use any knowledge outside of these sources. For every fact you state, cite the source. If the provided sources don't contain enough information to answer the question, say 'I don't have enough information on that based on the available course catalog.'"

Retrieved chunks are injected into the prompt as labeled context blocks (`[Source 1: DEPT COURSE# — Title]`) before the question. The LLM is not given any tools or memory beyond these blocks.

## Example Responses

**Example 1 — Cross-department query:**

*Question:* Which departments offer courses on optimization?

*Response:* The ECE and ISYE departments offer courses on optimization (Source 1, Source 2, Source 5). Additionally, the CS department offers a course that includes embedded software optimizations (Source 4).

*Sources:*
- [1] ECE ECE 3251 — Optimization for Information Systems
- [2] ISYE ISYE 4135 — Optimization Foundations for ML and AI
- [3] ISYE ISYE 4134 — Constraint Programming
- [4] CS CS 6291 — Embedded Software Optimizations
- [5] ISYE ISYE 7683 — Advanced Nonlinear Programming

---

**Example 2 — Course recommendation:**

*Question:* What should I take after intro programming if I'm interested in AI?

*Response:* Based on the provided course information, after intro programming, you may want to consider taking CS 3600 — Introduction to Artificial Intelligence (Source 5). This course seems like a foundational course in AI. However, the provided sources do not contain information about a clear sequence of courses for AI, so it is recommended to consult with a course advisor.

---

**Out-of-scope query:**

*Question:* What is the tuition at Georgia Tech?

*Response:* I don't have enough information on that based on the available course catalog. The provided sources only contain course descriptions and do not include tuition or financial information.

## Query Interface

**Interface:** Gradio web UI (`app.py`), runs at `http://localhost:7860`

**Input:** A text box where the user types a plain-language question about GT courses.

**Output:** Two text boxes — one for the answer (with inline source citations), one listing the top-5 retrieved course chunks with department, course number, title, and distance score.

**Sample interaction:**

> *Input:* What CS courses cover machine learning?
>
> *Answer:* The following CS courses cover machine learning: CS 7641 — Machine Learning (Source 4), CS 7651 — Human and Machine Learning (Source 3). Additionally, CSE 7752 — Scientific Machine Learning is a CSE course that covers machine learning (Source 1).
>
> *Sources:*
> [1] CSE CSE 6242 — Data and Visual Analytics (distance: 0.31)
> [2] CS CS 6301 — Advanced Topics in Software Engineering (distance: 0.38)
> [3] CS CS 7651 — Human and Machine Learning (distance: 0.29)
> [4] CS CS 7641 — Machine Learning (distance: 0.21)
> [5] CSE CSE 6140 — Computational Science and Engineering Algorithms (distance: 0.44)

## Evaluation Report

| # | Question | Expected | System Response | Accuracy |
|---|----------|----------|----------------|----------|
| 1 | What CS courses cover machine learning? | CS 7641, CS 7651 | Found CS 7641, CS 7651, plus noisy CSE chunks | Partially Accurate |
| 2 | Prerequisites for CS 3600? | CS 2110 + MATH 3012 or CS 2051 | "I don't have enough information" | Inaccurate |
| 3 | Departments with optimization courses? | ISyE, CS at minimum | Found ISyE, ECE, CS with specific courses | Accurate |
| 4 | AE courses at 4000 level? | List of AE 4XXX courses | Only returned AE 4361 | Inaccurate |
| 5 | What to take after intro programming for AI? | CS 2110 → CS 3600 | Recommended CS 3600 but no prereq chain | Partially Accurate |

## Failure Case

**Question 2: Prerequisites for CS 3600**

The system failed to retrieve the CS 3600 chunk when asked about its prerequisites, even though the chunk exists in the database (it appeared in Question 5's results).

**Root cause:** The query "what are the prerequisites for CS 3600?" encodes the concept of prerequisites + a specific course number. The CS 3600 chunk text begins with the course title and description — it doesn't contain the word "prerequisites" as a prominent, labeled field. The catalog stores prerequisite information inline within the description text. As a result, the CS 3600 embedding is not close enough in vector space to the query embedding to rank in the top 5.

**Fix:** Extract prerequisites as a separate metadata field during ingestion, enabling exact-match lookup by course number. Alternatively, hybrid search (BM25 + semantic) would help because BM25 would catch the exact string "CS 3600" regardless of semantic distance.

## Spec Reflection

**Where the spec helped:** The chunking strategy section of `planning.md` forced me to think about document structure before writing any code. Deciding on one-chunk-per-course before implementation meant the regex chunker had a clear target, and the sample chunk inspection step caught that some chunks were merging multiple courses due to the duplicate ID issue.

**Where implementation diverged:** The spec suggested fixed character-count chunking as a starting point, but after reading the catalog PDFs it was clear that course entries are the natural unit — not character counts. The implementation uses regex boundary detection instead, which is more robust for structured catalog data but more fragile for other document types.

## AI Usage

**Instance 1 — Ingestion pipeline:** I provided Claude with my planning.md Documents and Chunking Strategy sections and asked it to implement `ingest.py`. Claude generated a script using `pdfplumber` with a regex-based course chunker. I overrode the initial chunk ID scheme (`{department}_{course_number}`) after it caused a `DuplicateIDError` on courses that appeared twice in the PDF — fixed by appending the loop index to make IDs unique.

**Instance 2 — Gradio interface:** I provided Claude with the Milestone 5 spec requirements and asked it to wrap the existing query logic in a Gradio UI. Claude generated `app.py` with a two-column layout (answer + sources). I kept the structure but directed it to show distance scores in the sources panel, which helps evaluate retrieval quality at a glance.