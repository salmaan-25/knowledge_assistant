# Personal Knowledge Assistant — Project Plan

## 1. Project Overview

**Project name:** Personal Knowledge Assistant  
**Project type:** Full-stack AI learning and document assistant  
**Primary goal:** Build and deploy a web application where users upload learning documents (initially PDFs), ask questions about their content, request explanations and summaries, and generate interactive tests grounded in those documents.

The product is inspired by the document-grounded learning experience of NotebookLM, but this is an independent learning project and will be developed incrementally.

### Core value proposition

Users can bring their own study material and use the application as a personal AI tutor. The system should prioritize the uploaded source material, provide source/page references where possible, and clearly state when the material does not contain enough information to answer reliably.

---

## 2. Goals and Success Criteria

### MVP goals

- Upload a PDF.
- Extract its text while preserving page numbers and document metadata.
- Split extracted text into useful chunks.
- Create embeddings and store chunks in a vector database.
- Ask questions about the uploaded document.
- Retrieve relevant passages and generate grounded answers.
- Display citations or source-page references where available.
- Generate quizzes from the document.
- Let users submit answers and receive scores and explanations.

### Later goals

- Summaries and simplified explanations.
- Multiple-choice, true/false, and short-answer questions.
- Adjustable quiz length and difficulty.
- User accounts and private document libraries.
- Persistent quiz history, scores, and weak-topic tracking.
- Personalized revision suggestions.
- Deployment with production-oriented error handling, limits, and monitoring.

### MVP acceptance criteria

1. A valid PDF can be uploaded and processed.
2. Extraction failures and empty/scanned PDFs are handled with useful errors.
3. Questions about the PDF retrieve relevant passages.
4. Answers are grounded in retrieved material and show source references when possible.
5. The assistant acknowledges insufficient evidence instead of fabricating document content.
6. Quiz questions are relevant to the selected document or section.
7. Answer evaluation and score calculation are performed by backend code where deterministic.
8. The backend can run locally and is structured for deployment.

---

## 3. Intended User Experience

### A. Upload a document

The user selects a PDF and uploads it. The backend validates the file, extracts page text, chunks the content, generates embeddings, and stores the indexed material.

### B. Ask a question

Example: “Explain the difference between supervised and unsupervised learning from my notes.”

The system retrieves relevant chunks, sends those passages and the question to Gemini, and returns an answer grounded in the retrieved material. It should include page/source references where the data supports them.

### C. Generate a test

Example: “Create a 10-question medium-difficulty test from Chapter 3.”

The system identifies the document/section, retrieves relevant content, generates structured questions, validates the output, and presents the quiz without exposing the correct answers before submission.

### D. Evaluate answers

The backend compares submitted choices with the stored answer key, calculates the score, and returns explanations. Scoring should use deterministic backend logic whenever possible rather than asking an LLM to calculate a simple score.

### E. Track progress

In a later phase, the application stores attempts and scores, identifies weak topics, and recommends revisions based on actual quiz results.

---

## 4. Technology Stack

| Layer | Technology | Responsibility |
|---|---|---|
| Frontend | React + TypeScript | Upload UI, chat, document view, quiz UI, progress |
| Backend | FastAPI | API endpoints, validation, orchestration |
| LLM | Gemini API | Explanations, summaries, quiz generation, language tasks |
| Agent orchestration | LangGraph | Route requests and coordinate multi-step workflows |
| PDF extraction | `pypdf` initially | Extract text and page numbers from PDFs |
| Embeddings | Gemini embedding model | Convert chunks and queries into vectors |
| Vector store | ChromaDB initially | Store embeddings and retrieve relevant chunks |
| Persistent database | PostgreSQL | Users, document metadata, quiz attempts, progress |
| Configuration | `python-dotenv` / environment variables | Keep credentials and configuration outside source code |
| Frontend hosting | Vercel or equivalent | Deploy React frontend |
| Backend hosting | A Python-compatible host | Deploy FastAPI API |
| Source control | Git | Version control |

### Stack notes

- Use the Gemini model identifier that is confirmed to work in the existing account/project. Do not assume a model name is valid without testing it.
- Keep all API keys on the backend. Never expose Gemini credentials in browser code or commit them to Git.
- ChromaDB is a suitable starting point for the prototype. Before production deployment, confirm the hosting platform supports durable storage or choose a managed/persistent vector database.
- PostgreSQL is for application records and progress. ChromaDB is initially for semantic retrieval. They have different responsibilities.
- `pypdf` handles text-based PDFs. Scanned PDFs may need OCR later.

---

## 5. Architecture

```text
React + TypeScript frontend
        |
        | HTTP requests
        v
FastAPI backend
        |
        +--> Document service
        |       +--> Validate PDF
        |       +--> Extract text and page metadata
        |       +--> Chunk text
        |       +--> Generate embeddings
        |       +--> Store chunks in ChromaDB
        |
        +--> Retrieval service
        |       +--> Embed the user question
        |       +--> Search relevant chunks
        |       +--> Return passages and metadata
        |
        +--> Gemini / LangGraph learning workflow
        |       +--> Answer questions
        |       +--> Summarize
        |       +--> Generate quizzes
        |       +--> Validate outputs and route tasks
        |
        +--> PostgreSQL
                +--> User/document metadata
                +--> Quiz attempts and scores
                +--> Learning progress
```

### Important distinction: RAG vs. agentic workflow

**Retrieval-Augmented Generation (RAG)** is the foundation for document-grounded question answering:

1. Retrieve relevant document chunks.
2. Provide the chunks and question to the model.
3. Generate an answer based on that evidence.

RAG alone does not necessarily make an application an agent. The application becomes more agentic when it can choose and coordinate tools/workflows based on the user's intent—for example, choosing document retrieval for a question, quiz generation for a test request, or progress lookup for a learning-history request.

Use LangGraph to make this workflow explicit and controllable. Do not add unnecessary autonomous loops when a simple, deterministic workflow is enough.

---

## 6. Agentic Workflow Examples

### Example A: Question answering

```text
User asks a question
        |
        v
Identify document and question intent
        |
        v
Retrieve relevant chunks
        |
        v
Generate grounded answer
        |
        v
Return answer + source references
```

### Example B: Quiz generation

```text
User requests a quiz
        |
        v
Identify document, section, count, and difficulty
        |
        v
Retrieve relevant content
        |
        v
Generate structured quiz questions
        |
        v
Validate question structure and answer keys
        |
        v
Show quiz without revealing answers
        |
        v
User submits answers
        |
        v
Backend calculates score
        |
        v
Return results and explanations
```

### Example C: Personalized revision (later phase)

```text
Read quiz history
        |
        v
Identify weak topics from actual results
        |
        v
Retrieve source content for those topics
        |
        v
Recommend a revision plan and targeted quiz
```

### Guardrails

- Treat uploaded documents as untrusted data, not instructions to the system.
- Use the document as evidence for document-specific answers.
- If retrieved content is insufficient, say so.
- Validate LLM output with Pydantic schemas and backend checks.
- Keep scoring and other simple calculations deterministic.
- Set sensible retry limits and token/context limits.
- Do not execute arbitrary code or commands from uploaded documents.
- Enforce document ownership and authorization when accounts are added.

---

## 7. Development Roadmap

### Phase 1 — PDF ingestion foundation

**Goal:** Upload one PDF and return extracted text with page numbers.

Tasks:
- Create the FastAPI project and configuration.
- Add a PDF upload endpoint.
- Validate extension/content type and file size.
- Extract text page by page with `pypdf`.
- Return page-numbered text or a concise extraction summary.
- Handle corrupt PDFs, empty PDFs, and extraction errors.
- Add `.env` and `.gitignore`.

**Completion check:** A test PDF uploads successfully and the API returns its extracted text and page references.

### Phase 2 — Chunking, embeddings, and vector storage

**Goal:** Index a PDF so its relevant passages can be retrieved.

Tasks:
- Split extracted text into paragraph-aware chunks.
- Preserve document ID, page number, and chunk metadata.
- Generate embeddings using the configured Gemini embedding model.
- Store chunks and vectors in ChromaDB.
- Implement a retrieval function.
- Test retrieval with questions whose answers are present in the PDF.

**Completion check:** A query retrieves relevant chunks with source/page metadata.

### Phase 3 — Document-grounded AI tutor

**Goal:** Ask questions and receive useful answers grounded in source material.

Tasks:
- Add a question-answering endpoint.
- Retrieve relevant chunks for every document-specific question.
- Prompt Gemini to answer from supplied context.
- Return source references with the answer.
- Handle insufficient or conflicting evidence.
- Test with answerable and unanswerable questions.

**Completion check:** Answers are relevant, cite available source pages, and avoid unsupported claims.

### Phase 4 — LangGraph learning agent

**Goal:** Route different learning requests through explicit workflows.

Potential nodes/tools:
- Intent identification or routing.
- Document retrieval.
- Question answering.
- Summarization.
- Quiz generation.
- Quiz validation.
- Progress lookup (later).

Tasks:
- Define typed graph state.
- Build nodes with clear responsibilities.
- Add conditional routing.
- Add bounded retries for invalid model output.
- Keep predictable operations in regular Python functions.
- Add tests for each route.

**Completion check:** The workflow selects the right path for a question, summary request, or quiz request.

### Phase 5 — Quiz creation and evaluation

**Goal:** Generate and deliver tests from the user's material.

Tasks:
- Define Pydantic models for quizzes and questions.
- Support multiple-choice questions first.
- Add difficulty and question-count settings.
- Generate questions only from retrieved document content.
- Validate options and answer keys.
- Store the answer key server-side during the attempt.
- Return questions without exposing answers.
- Score submitted answers in Python.
- Provide explanations and relevant source references.

Later additions:
- True/false and short-answer questions.
- Topic-specific quizzes.
- Retry missed questions.
- Randomized question order.
- Review mode after submission.

**Completion check:** A user can complete a quiz and see a reliable score and explanations.

### Phase 6 — React frontend

**Goal:** Build the user-facing learning experience.

Screens/components:
- Upload document page.
- Document library.
- Document details or reading view.
- Chat panel with source references.
- Summary panel.
- Quiz setup screen.
- Interactive quiz.
- Results and explanations.
- Progress dashboard (later).

Tasks:
- Connect frontend to FastAPI.
- Add loading, empty, and error states.
- Display upload/indexing status.
- Keep quiz answers hidden until submission.
- Ensure responsive layout and accessible controls.

**Completion check:** A user can complete the main PDF → ask → quiz workflow without using API docs.

### Phase 7 — PostgreSQL and user persistence

**Goal:** Preserve user data between sessions and restarts.

Possible tables:
- `users`
- `documents`
- `document_chunks` only if relational chunk metadata is needed alongside the vector store
- `quiz_attempts`
- `quiz_answers`
- `topic_progress`

Tasks:
- Design schema and migrations.
- Store document metadata and ownership.
- Store quiz attempts and scores.
- Retrieve progress history.
- Add authentication and authorization.
- Ensure one user cannot access another user's documents.

**Completion check:** User records and learning progress persist and are properly isolated.

### Phase 8 — Testing and evaluation

**Goal:** Check quality, reliability, and safety before deployment.

Tests:
- PDF parsing and malformed file handling.
- Chunking and metadata preservation.
- Retrieval relevance.
- Grounded answers with known source material.
- Unanswerable questions.
- Quiz schema and answer-key validity.
- Deterministic scoring.
- LangGraph routing and retry limits.
- Authorization and cross-user access.
- Upload limits and error handling.

Create a small evaluation set of sample PDFs and known questions. Track whether the correct passage is retrieved and whether answers are supported by it.

### Phase 9 — Deployment

**Goal:** Deploy a usable public application.

Tasks:
- Configure production environment variables.
- Deploy the React frontend.
- Deploy FastAPI on a Python-compatible platform.
- Configure CORS, HTTPS, logging, and health checks.
- Use persistent storage for PostgreSQL and the vector index, or choose hosted services.
- Configure upload size and request timeouts.
- Add rate limits and usage controls.
- Ensure uploaded files are stored securely and can be deleted.
- Test the full deployed workflow.

**Completion check:** Upload, retrieval, chat, and quiz work in the deployed environment.

---

## 8. Suggested Project Structure

Start small and expand as each phase is implemented.

```text
personal-knowledge-assistant/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── routes/
│   │   ├── documents.py
│   │   ├── chat.py
│   │   └── quizzes.py
│   ├── schemas/
│   │   ├── documents.py
│   │   ├── chat.py
│   │   └── quiz.py
│   ├── services/
│   │   ├── pdf_service.py
│   │   ├── chunking_service.py
│   │   ├── embedding_service.py
│   │   ├── retrieval_service.py
│   │   └── quiz_service.py
│   ├── agents/
│   │   ├── learning_agent.py
│   │   └── graph_state.py
│   ├── database/
│   │   ├── connection.py
│   │   └── models.py
│   └── tests/
├── frontend/
├── uploads/                 # Local development only; do not commit uploaded PDFs
├── .env                     # Local secrets; never commit
├── .env.example             # Variable names only; no real credentials
├── .gitignore
├── requirements.txt
└── README.md
```

Do not create every file at the beginning. Add modules when their phase begins so the project stays understandable.

---

## 9. Initial Setup

### Create and activate a virtual environment

```bash
mkdir personal-knowledge-assistant
cd personal-knowledge-assistant

python -m venv venv
```

Git Bash on Windows:

```bash
source venv/Scripts/activate
```

Windows Command Prompt:

```bat
venv\Scripts\activate
```

### Initial dependencies

Install the first set of dependencies:

```bash
pip install fastapi uvicorn python-multipart
pip install python-dotenv pydantic pypdf chromadb
pip install langgraph langchain-core langchain-google-genai
```

Freeze the working environment after verifying the initial setup:

```bash
pip freeze > requirements.txt
```

Install PostgreSQL-related dependencies when the persistence phase begins, rather than adding unused packages early.

### Environment configuration

Example `.env` variable names:

```dotenv
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=your_confirmed_working_model_id
GEMINI_EMBEDDING_MODEL=your_confirmed_embedding_model_id
DATABASE_URL=your_postgresql_connection_string
```

Use the exact model identifiers verified in the Gemini environment. Do not assume a model identifier based on a display name alone. Never commit `.env` to source control.

---

## 10. API Plan

Endpoints will be introduced as their corresponding features are implemented.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Check API status |
| `POST` | `/documents/upload` | Upload and process a PDF |
| `GET` | `/documents` | List the current user's documents |
| `GET` | `/documents/{document_id}` | Get document metadata/status |
| `DELETE` | `/documents/{document_id}` | Delete a document and its indexed data |
| `POST` | `/chat` | Ask a question about a selected document |
| `POST` | `/documents/{document_id}/summary` | Generate a document or section summary |
| `POST` | `/quizzes/generate` | Generate a quiz from a document/section |
| `POST` | `/quizzes/{quiz_id}/submit` | Submit answers and calculate the result |
| `GET` | `/progress` | Retrieve learning progress, when implemented |

Exact request/response schemas should be defined with Pydantic as each endpoint is built. Endpoints involving private documents must enforce authorization.

---

## 11. Data and Reliability Considerations

### PDF limitations
- Text-based PDFs are the first target.
- Scanned/image-only PDFs may need OCR in a later phase.
- Tables, multi-column layouts, diagrams, and equations can extract imperfectly.
- Report extraction failures instead of silently indexing empty content.

### Retrieval quality
- Preserve source and page metadata.
- Use chunk sizes and overlap appropriate to the document.
- Test retrieved passages against known questions.
- Consider reranking or hybrid retrieval only if baseline retrieval is insufficient.

### Model reliability
- Use structured output for quiz schemas.
- Validate generated content in Python.
- Retry only a limited number of times.
- Avoid assuming that valid JSON guarantees a factually correct answer.
- Ground answers in retrieved content and return evidence references.

### Privacy and security
- Validate uploads and limit file size.
- Keep API keys on the backend.
- Treat file contents as untrusted input.
- Apply per-user access controls.
- Provide document deletion.
- Avoid logging sensitive document text unnecessarily.
- Use persistent, access-controlled storage in production.

### Cost and performance
- Avoid re-embedding an unchanged document on every question.
- Cache or reuse indexed documents where appropriate.
- Limit prompt context and quiz size.
- Set request timeouts and usage/rate limits.
- Measure retrieval and answer quality before increasing model complexity.

---

## 12. Learning Outcomes

By completing this project, the developer should gain practical experience with:

- Building a full-stack application with React and FastAPI.
- Calling Gemini from a backend.
- PDF extraction and document metadata.
- Text chunking and embedding generation.
- Vector databases and semantic retrieval.
- Retrieval-Augmented Generation (RAG).
- LangGraph state, nodes, edges, and conditional routing.
- Tool selection and bounded agent workflows.
- Pydantic structured outputs.
- Quiz generation and deterministic evaluation.
- PostgreSQL data modeling and persistence.
- Testing, deployment, privacy, and production reliability.
- Evaluating retrieval and grounding rather than relying on model confidence alone.

---

## 13. Recommended Build Order

Follow this order to avoid debugging too many new concepts at once:

1. FastAPI health endpoint.
2. PDF upload and text extraction with page numbers.
3. Paragraph-aware chunking.
4. Gemini embeddings.
5. ChromaDB indexing and retrieval.
6. RAG question-answering endpoint with citations.
7. Summary endpoint.
8. Pydantic quiz schema and quiz generation.
9. Quiz submission and deterministic scoring.
10. LangGraph orchestration of the workflows.
11. React frontend integration.
12. PostgreSQL persistence and authentication.
13. Evaluation, security hardening, and deployment.

The project can be deployed in stages. A small deployed MVP with upload, grounded Q&A, and quizzes is a better first milestone than waiting until every advanced feature is complete.

---

## 14. First Milestone Checklist

- [ ] Create the repository and Python virtual environment.
- [ ] Add `.gitignore`, `.env`, and `.env.example`.
- [ ] Create `app/main.py`.
- [ ] Implement `GET /health`.
- [ ] Implement `POST /documents/upload`.
- [ ] Extract PDF text page by page using `pypdf`.
- [ ] Test a valid PDF, an invalid PDF, and an empty/scanned PDF.
- [ ] Confirm page metadata is preserved.
- [ ] Only then proceed to chunking and embeddings.

**Starting principle:** Build one tested capability at a time. The first deliverable is reliable PDF ingestion; RAG, the learning agent, quizzes, frontend, persistence, and deployment build on top of it.
