# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A full-stack Retrieval-Augmented Generation (RAG) chatbot for querying course materials. Users ask questions through a web interface; the system semantically searches indexed course documents and uses Claude to generate accurate, context-grounded answers with source citations.

**Stack:** Python 3.13 · FastAPI · ChromaDB · Anthropic Claude (`claude-sonnet-4-20250514`) · Sentence Transformers (`all-MiniLM-L6-v2`) · Vanilla JS frontend

**Key flows:**
- **Ingestion** — course `.txt`/`.pdf`/`.docx` files in `docs/` are parsed, chunked, embedded, and stored in ChromaDB on startup
- **Query** — user question → optional ChromaDB semantic search → Claude synthesizes answer → response with sources returned to UI
- **Sessions** — in-memory conversation history (last 2 exchanges) provides multi-turn context

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                    Frontend (/)                      │
│  index.html · script.js · style.css                 │
│  Served as static files by FastAPI                  │
└────────────────────┬────────────────────────────────┘
                     │ POST /api/query
                     ▼
┌─────────────────────────────────────────────────────┐
│                  app.py (FastAPI)                    │
│  /api/query  →  RAGSystem.query()                   │
│  /api/courses →  RAGSystem.get_course_analytics()   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              rag_system.py (Orchestrator)            │
│                                                      │
│  SessionManager ──► conversation history             │
│  AIGenerator    ──► Claude API (1-2 calls)           │
│  ToolManager    ──► CourseSearchTool                 │
│                          │                           │
│                          ▼                           │
│                    VectorStore                       │
│              ┌───────────┴───────────┐              │
│        course_catalog         course_content         │
│      (course metadata,      (text chunks,            │
│       name resolution)       semantic search)        │
│              └───────────────────────┘              │
│                       ChromaDB                       │
└─────────────────────────────────────────────────────┘
```

**Two-call Claude pattern:** The first API call includes the `search_course_content` tool definition with `tool_choice: auto`. If Claude decides to search, the tool result is appended and a second call (no tools) produces the final answer. General-knowledge questions skip the tool entirely and resolve in one call.

**ChromaDB dual-collection design:** `course_catalog` holds one document per course and is used exclusively for fuzzy course-name resolution (vector search → exact title). `course_content` holds all text chunks and is filtered by the resolved title for precise retrieval.

## Commands

> Always use `uv` to manage dependencies and run Python commands. Never use `pip` directly — use `uv add <package>` to add packages and `uv sync` to install.

**Install dependencies:**
```bash
uv sync
```

**Run the application:**
```bash
./run.sh
# or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

**Run a single backend file directly:**
```bash
cd backend && uv run python <file>.py
```

The app is available at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

**Reset the vector database** (forces re-ingestion of all documents):
```bash
rm -rf backend/chroma_db
```

## Architecture

This is a full-stack RAG chatbot. The backend is a single FastAPI process that serves both the API and the frontend static files.

### Request lifecycle

1. `frontend/script.js` — sends `POST /api/query {query, session_id}` and renders the markdown response
2. `backend/app.py` — validates the request and delegates to `RAGSystem.query()`
3. `backend/rag_system.py` — the central orchestrator; fetches conversation history, calls the AI generator, retrieves sources, and saves the exchange back to the session
4. `backend/ai_generator.py` — makes **two** Claude API calls when search is needed: the first lets Claude decide whether to invoke the `search_course_content` tool; if it does, the tool result is injected and a second call produces the final answer. With `tool_choice: auto`, Claude may skip the tool for general-knowledge questions.
5. `backend/search_tools.py` — `CourseSearchTool` executes the vector search and formats results; `ToolManager` registers tools and surfaces `last_sources` to the RAG system after each query.
6. `backend/vector_store.py` — wraps ChromaDB with two persistent collections: `course_catalog` (one doc per course, used for fuzzy course-name resolution) and `course_content` (all text chunks, used for semantic search). Course name matching uses a vector search on `course_catalog` before filtering `course_content`.

### Document ingestion

On server startup (`app.py` startup event), all `.txt/.pdf/.docx` files in `../docs/` are processed. Already-indexed courses (matched by title) are skipped.

`backend/document_processor.py` expects this file format:
```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <title>
Lesson Link: <url>
<lesson body text>

Lesson 1: <title>
...
```
Text is chunked sentence-by-sentence with configurable size and overlap.

### Configuration

All tuneable parameters live in `backend/config.py`:
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — controls chunking granularity
- `MAX_RESULTS` — number of chunks returned per search
- `MAX_HISTORY` — conversation turns kept in memory per session (default: 2)
- `ANTHROPIC_MODEL` — Claude model used for generation
- `CHROMA_PATH` — where ChromaDB persists data (`./chroma_db` relative to `backend/`)

### Known inconsistency

The first chunk of each lesson is prefixed `"Lesson N content: ..."` but the last lesson's chunks are prefixed `"Course X Lesson N content: ..."` — this is a bug in `document_processor.py` lines 186 vs 234.
