# Intelligent Document Q&A Assistant using RAG

A **production-ready Retrieval-Augmented Generation (RAG)** application that lets users upload documents (PDF, DOCX, TXT) and get grounded, cited answers to natural-language questions — powered by FAISS vector search and NVIDIA NIM LLMs.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FastAPI Application                          │
│  POST /upload    POST /query    GET /documents    GET /health        │
└────────────────────────┬────────────────┬────────────────────────────┘
                         │                │
          ┌──────────────▼──────┐  ┌──────▼──────────────────┐
          │  Ingestion Pipeline │  │   Generation Pipeline    │
          │                     │  │                          │
          │  TextExtractor      │  │  RetrieverService        │
          │  ↓ (PDF/DOCX/TXT)   │  │  ↓ embed query          │
          │  DocumentChunker    │  │  FAISSVectorStore.search │
          │  ↓ RecursiveCharSpl │  │  ↓ top-K chunks          │
          │  EmbeddingService   │  │  PromptBuilder           │
          │  ↓ all-MiniLM-L6-v2 │  │  ↓ context + question    │
          │  FAISSVectorStore   │  │  LLMClient (NIM/OpenAI)  │
          │  ↓ IndexFlatIP      │  │  ↓ grounded answer       │
          │  PostgreSQL         │  │  Citations               │
          │  MLflow             │  │  PostgreSQL (history)    │
          └─────────────────────┘  │  MLflow (metrics)        │
                                   └──────────────────────────┘
```

## Features

- **Multi-format document upload** — PDF, DOCX, TXT (up to 50 MB)
- **Recursive text chunking** — configurable size and overlap
- **Dense embeddings** — `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **FAISS vector index** — cosine similarity, persisted to disk
- **Grounded answers** — LLM answers from retrieved context only
- **Source citations** — document name + page number per answer
- **PostgreSQL persistence** — document metadata + full chat history
- **MLflow experiment tracking** — latency, chunk params, model names
- **Docker Compose deployment** — app + PostgreSQL + MLflow

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Docker & Docker Compose
- NVIDIA NIM API key (get one free at https://build.nvidia.com)

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set NVIDIA_API_KEY=nvapi-...
```

### 3. Run with Docker Compose

```bash
docker compose up --build
```

Services started:
| Service | URL |
|---------|-----|
| RAG API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| MLflow UI | http://localhost:5001 |
| PostgreSQL | localhost:5432 |

### 4. Run locally (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Start PostgreSQL separately, then:
uvicorn app.main:app --reload --port 8000
```

---

## API Reference

### `GET /api/v1/health`
Liveness probe.

**Response:**
```json
{"status": "healthy", "version": "1.0.0"}
```

---

### `POST /api/v1/upload`
Upload a document for indexing.

**Request:** `multipart/form-data` with field `file`

**Response:**
```json
{
  "message": "Document uploaded and indexed successfully.",
  "document": {
    "id": "uuid",
    "original_filename": "report.pdf",
    "file_type": "pdf",
    "file_size_bytes": 102400,
    "total_chunks": 47,
    "total_pages": 12,
    "status": "ready",
    "ingestion_latency_seconds": 3.14,
    "created_at": "2024-01-15T10:30:00Z"
  }
}
```

---

### `GET /api/v1/documents`
List all uploaded documents.

**Response:** Array of document objects (same schema as above).

---

### `POST /api/v1/query`
Ask a question about the uploaded documents.

**Request:**
```json
{
  "question": "What are the key findings in the report?",
  "session_id": "user-123",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "The key findings include...",
  "citations": [
    {"document_name": "report.pdf", "page_number": 3, "relevance_score": 0.92},
    {"document_name": "report.pdf", "page_number": 7, "relevance_score": 0.87}
  ],
  "retrieval_latency_seconds": 0.045,
  "response_latency_seconds": 1.23,
  "model_used": "meta/llama-3.1-8b-instruct",
  "session_id": "user-123"
}
```

---

## Project Structure

```
.
├── app/
│   ├── api/
│   │   └── routes.py              # FastAPI route handlers
│   ├── config/
│   │   └── settings.py            # Pydantic Settings (env-based config)
│   ├── database/
│   │   └── connection.py          # SQLAlchemy engine + session
│   ├── models/
│   │   ├── document.py            # Document ORM model
│   │   └── chat.py                # ChatMessage ORM model
│   ├── schemas/
│   │   ├── document.py            # Pydantic request/response schemas
│   │   └── query.py               # Query request/response schemas
│   ├── services/
│   │   ├── ingestion/
│   │   │   ├── extractor.py       # PDF/DOCX/TXT text extraction
│   │   │   ├── chunker.py         # RecursiveCharacterTextSplitter
│   │   │   ├── embedder.py        # sentence-transformers embeddings
│   │   │   ├── vector_store.py    # FAISS index CRUD
│   │   │   └── pipeline.py        # Ingestion orchestrator
│   │   ├── retrieval/
│   │   │   └── retriever.py       # Semantic search service
│   │   └── generation/
│   │       ├── prompt_builder.py  # Prompt assembly
│   │       ├── llm_client.py      # OpenAI-compatible LLM client
│   │       └── generator.py       # RAG answer generation
│   ├── utils/
│   │   ├── mlflow_tracker.py      # MLflow logging helpers
│   │   └── logger.py              # Structured logging setup
│   └── main.py                    # FastAPI app + lifespan
├── tests/
│   ├── conftest.py                # Shared fixtures (SQLite test DB)
│   ├── test_extractor.py
│   ├── test_chunker.py
│   ├── test_vector_store.py
│   ├── test_prompt_builder.py
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
├── .env.example
└── README.md
```

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `NVIDIA_API_KEY` | *(required)* | NVIDIA NIM API key |
| `NVIDIA_BASE_URL` | `https://integrate.api.nvidia.com/v1` | LLM API base URL |
| `LLM_MODEL` | `meta/llama-3.1-8b-instruct` | Model identifier |
| `DATABASE_URL` | `postgresql://raguser:ragpassword@localhost:5432/ragdb` | PostgreSQL URL |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `5` | Chunks retrieved per query |
| `MLFLOW_TRACKING_URI` | `http://localhost:5001` | MLflow server URL |

---

## Running Tests

```bash
pip install -r requirements.txt
pytest
```

Coverage report is printed automatically. No PostgreSQL required — tests use SQLite.

---

## Switching LLM Providers

The `LLMClient` uses the OpenAI SDK and is provider-agnostic. To switch:

```python
# OpenAI
LLMClient(api_key="sk-...", base_url="https://api.openai.com/v1", model="gpt-4o")

# Together AI
LLMClient(api_key="...", base_url="https://api.together.xyz/v1", model="mistralai/Mixtral-8x7B")

# Groq
LLMClient(api_key="...", base_url="https://api.groq.com/openai/v1", model="llama3-8b-8192")

# Local Ollama
LLMClient(api_key="ollama", base_url="http://localhost:11434/v1", model="llama3")
```

Only the three constructor arguments change — zero application code changes required.

---

## MLflow Experiment Tracking

Every upload and query is logged as an MLflow run:

**Ingestion run params/metrics:**
- `embedding_model`, `chunk_size`, `chunk_overlap`
- `total_chunks`, `ingestion_latency_seconds`

**Query run params/metrics:**
- `llm_model`, `top_k`
- `retrieval_latency_seconds`, `response_latency_seconds`, `total_latency_seconds`, `num_sources_cited`

Access the MLflow UI at http://localhost:5001 after starting Docker Compose.

---

## Interview Explanation

**What problem does this solve?**
Organizations store institutional knowledge in PDFs and Word documents. Finding specific information manually takes hours. This system makes any document corpus instantly searchable via natural language.

**How does RAG work here?**
1. Documents are chunked into 512-character overlapping segments.
2. Each chunk is embedded into a 384-dimensional vector using a pre-trained transformer.
3. All vectors are stored in a FAISS flat index (inner-product similarity on L2-normalised vectors ≡ cosine similarity).
4. At query time, the question is embedded and the top-K most similar chunks are retrieved.
5. A prompt is built: `[system: answer only from context] + [retrieved chunks + question]`.
6. The LLM generates a grounded answer. If context doesn't contain the answer, it returns a fixed fallback string.
7. Source citations (filename + page number) accompany every response.

**Why FAISS over a managed vector DB?**
FAISS is free, runs in-process (zero latency overhead for small-to-medium corpora), and is a standard interview topic. The `FAISSVectorStore` class abstracts the index so swapping to Pinecone, Weaviate, or Qdrant is a one-file change.

**Why sentence-transformers/all-MiniLM-L6-v2?**
It's fast (384-dim vs 1536-dim for OpenAI embeddings), runs locally (no API cost), and scores competitively on semantic search benchmarks. The embedding service is injected as a dependency, making it easy to swap for OpenAI or Cohere embeddings.

---

## Resume Description

**Intelligent Document Q&A Assistant | Python · FastAPI · RAG · FAISS · PostgreSQL · MLflow · Docker**

Built a production-ready Retrieval-Augmented Generation system enabling natural-language Q&A over private document corpora. Implemented an end-to-end pipeline: multi-format document ingestion (PDF/DOCX/TXT), recursive text chunking, dense embedding generation with `sentence-transformers/all-MiniLM-L6-v2`, FAISS cosine-similarity search, grounded answer generation via NVIDIA NIM LLMs, and source citations with page-level provenance. Exposed the system as a RESTful FastAPI service backed by PostgreSQL for metadata and chat history persistence. Integrated MLflow for experiment tracking of embedding parameters, retrieval latency, and response latency. Containerised with Docker Compose (app + PostgreSQL + MLflow). Achieved 90%+ test coverage with pytest across unit and API test suites.
