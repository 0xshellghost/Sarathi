# Sarathi — AI-Driven Civic & Legal Empowerment Engine

> Empowering Indian citizens with AI-powered legal awareness and document generation.

Sarathi is a production-ready backend that takes a citizen's plain-language legal problem, classifies it, retrieves relevant Indian law, extracts structured entities, and produces ready-to-render legal document payloads — all through a deterministic, streaming AI pipeline.

**100% self-hosted. Zero paid APIs. Bring your own models.**

---

## Architecture

```
User Input ──► Prompt Injection Guard ──► PII Sanitizer ──► AI Pipeline
                                                               │
                                          ┌────────────────────┘
                                          ▼
                                   ┌─────────────┐
                            Step 1 │ Intent Router│  Classify legal domain
                                   └──────┬──────┘
                                          ▼
                                   ┌─────────────┐
                            Step 2 │ RAG Pipeline │  Retrieve legal clauses (ChromaDB)
                                   └──────┬──────┘
                                          ▼
                                   ┌──────────────────┐
                            Step 3 │ Entity Extractor  │  Extract fields (JSON schema)
                                   └──────┬───────────┘
                                          ▼
                                   ┌──────────────────┐
                            Step 4 │ Output Formatter  │  Compile PDF payload
                                   └──────┬───────────┘
                                          ▼
                                   SSE Stream ──► Frontend
```

**Strictly deterministic** — no autonomous agents, no infinite loops. Each step receives the previous step's output. The pipeline is a straight line.

---

## Tech Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| **Framework** | FastAPI + Pydantic | Async, type-safe, OpenAPI docs |
| **LLM** | Ollama (your model) | Any model you build/fine-tune |
| **Embeddings** | Ollama (your model) | Your custom embedding model |
| **Speech-to-Text** | Your STT endpoint | Pluggable HTTP interface |
| **Vector DB** | ChromaDB | In-process, persistent, zero-latency |
| **Document DB** | MongoDB + Motor | Async session persistence |
| **Streaming** | Server-Sent Events | Real-time token streaming |

---

## Project Structure

```
backend/
├── main.py                        # App factory, lifespan, middleware
├── config.py                      # Pydantic Settings (.env driven)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
│
├── middleware/
│   ├── pii_sanitizer.py           # Redacts Aadhaar, PAN, phone, email
│   └── prompt_injection_guard.py  # Heuristic injection detection
│
├── routers/
│   ├── analyze.py                 # POST /api/v1/action/analyze (SSE)
│   ├── finalize.py                # POST /api/v1/action/finalize
│   ├── sessions.py                # GET /api/v1/sessions/*
│   └── transcribe.py              # POST /api/v1/transcribe
│
├── services/
│   ├── llm_client.py              # Ollama API client (model-agnostic)
│   ├── ai_pipeline.py             # 4-step deterministic orchestrator
│   ├── intent_router.py           # Step 1: Legal domain classification
│   ├── rag_pipeline.py            # Step 2: ChromaDB retrieval
│   ├── entity_extractor.py        # Step 3: JSON extraction + silent retries
│   ├── output_formatter.py        # Step 4: PDF payload compilation
│   └── transcription.py           # Pluggable STT interface
│
├── db/
│   ├── mongodb.py                 # Motor async CRUD
│   └── chromadb_store.py          # Vector store + custom embeddings
│
├── models/
│   ├── domain.py                  # Legal domains, entity schemas
│   ├── requests.py                # Pydantic request models
│   └── responses.py               # Pydantic response + SSE event models
│
└── scripts/
    └── seed_chroma.py             # Seed Indian legal corpus
```

---

## Quick Start

### Prerequisites

- **Python 3.12+**
- **MongoDB** (local or Docker)
- **Ollama** with your custom models loaded

### 1. Install Ollama & Load Your Models

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Load your custom LLM (or use any available model for testing)
ollama pull llama3.1:8b    # Example — replace with your model

# Load your custom embedding model
ollama pull nomic-embed-text  # Example — replace with your model
```

### 2. Start MongoDB

```bash
# Option A: Docker
docker run -d --name sarathi-mongo -p 27017:27017 mongo:7

# Option B: Local installation
mongod --dbpath ./data/db
```

### 3. Set Up the Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your model names and settings
```

### 4. Seed the Legal Corpus

```bash
python scripts/seed_chroma.py
```

This loads 17 sample Indian legal provisions (Model Tenancy Act, Consumer Protection Act, NI Act, etc.) into ChromaDB.

### 5. Run the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API is now live at `http://localhost:8000`. OpenAPI docs at `http://localhost:8000/docs`.

---

## Docker Deployment

```bash
cd backend

# Build and run the full stack
docker compose up -d

# This starts:
#   - FastAPI backend  → localhost:8000
#   - MongoDB          → localhost:27017
#   - Ollama           → localhost:11434
```

For GPU acceleration (NVIDIA), uncomment the GPU section in `docker-compose.yml`.

---

## API Reference

### `POST /api/v1/action/analyze`

Analyze a legal problem. Returns an SSE stream.

```bash
curl -X POST http://localhost:8000/api/v1/action/analyze \
  -H "Content-Type: application/json" \
  -d '{"user_input": "My landlord won'\''t return my 50000 rupee deposit after I moved out 3 months ago"}' \
  --no-buffer
```

**SSE Events:**
| Event Type | Description |
|---|---|
| `{type: "token", text: "..."}` | Streamed explanation tokens |
| `{type: "form_request", schema: {...}}` | Entity form for user to fill |
| `{type: "complete", session_id: "..."}` | Pipeline finished |
| `{type: "error", message: "..."}` | Error (with `recoverable` flag) |

### `POST /api/v1/action/finalize`

Submit user-confirmed form data. Returns a PDF payload.

```json
{
  "session_id": "sess_abc123",
  "case_type": "rent_deposit_dispute",
  "extracted_data": {
    "landlord_name": "Arun Sharma",
    "property_address": "123 MG Road, Bangalore",
    "deposit_amount": 50000
  }
}
```

### `POST /api/v1/transcribe`

Upload audio for speech-to-text transcription.

```bash
curl -X POST http://localhost:8000/api/v1/transcribe \
  -F "audio=@recording.wav"
```

### `GET /api/v1/sessions/history`

List all past sessions.

### `GET /api/v1/sessions/cases/{case_id}`

Retrieve a specific finalized case with its PDF payload.

### `GET /health`

Health check with MongoDB and ChromaDB status.

---

## Supported Legal Domains

| Domain | Description | Key Acts |
|--------|-------------|----------|
| `rent_deposit_dispute` | Tenant-landlord deposit issues | Model Tenancy Act, 2021 |
| `consumer_complaint` | Defective products/services | Consumer Protection Act, 2019 |
| `employment_dispute` | Wages, termination, harassment | Payment of Gratuity Act, Indian Contract Act |
| `property_dispute` | Ownership, encroachment | Transfer of Property Act, 1882 |
| `cheque_bounce` | Dishonoured cheques | NI Act, 1881 (Section 138) |
| `general_legal_query` | Catch-all legal questions | Various |

---

## Security

### PII Sanitization
The middleware automatically detects and redacts in logs:
- **Aadhaar** numbers (12-digit patterns)
- **PAN** cards (ABCDE1234F format)
- **Phone** numbers (+91 / Indian mobile)
- **Email** addresses

Original data passes through to the AI pipeline untouched — only log output is sanitized.

### Prompt Injection Guard
Heuristic-based detection of:
- Instruction override ("ignore previous instructions")
- System prompt extraction attempts
- Role-play / jailbreak attacks (DAN, god mode)
- Delimiter injection (XML tags, markdown fences)

Requests exceeding the threat threshold are rejected with HTTP 422.

---

## Bringing Your Own Models

Sarathi is designed to be **model-agnostic**. You provide three models:

### 1. LLM (Chat Model)
Serve via Ollama or any OpenAI-compatible API:
```bash
# Create an Ollama Modelfile for your custom model
ollama create sarathi-legal -f Modelfile

# Update .env
OLLAMA_MODEL=sarathi-legal
```

### 2. Embedding Model
For ChromaDB vector search:
```bash
# Serve your embedding model via Ollama
ollama create sarathi-embed -f EmbedModelfile

# Update .env
OLLAMA_EMBED_MODEL=sarathi-embed
```

### 3. Speech-to-Text Model (Optional)
Expose your STT model as an HTTP endpoint that accepts `multipart/form-data` with an `audio` field and returns:
```json
{"text": "transcribed text", "language": "hi", "duration": 12.5}
```

```bash
# Update .env
STT_ENDPOINT=http://localhost:9000/transcribe
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `sarathi-legal` | Your LLM model name |
| `OLLAMA_EMBED_MODEL` | `sarathi-embed` | Your embedding model name |
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGODB_DB_NAME` | `sarathi` | Database name |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | ChromaDB storage path |
| `RAG_TOP_K` | `5` | Number of RAG results to retrieve |
| `STT_ENDPOINT` | *(empty)* | Your STT model's HTTP endpoint |
| `LLM_TEMPERATURE` | `0.1` | Low for deterministic legal output |
| `LLM_MAX_RETRIES` | `3` | Silent retry count for entity extraction |
| `CORS_ORIGINS` | `["http://localhost:3000"]` | Allowed frontend origins |
| `INJECTION_THREAT_THRESHOLD` | `0.6` | Prompt injection block threshold |
| `PII_REDACTION_ENABLED` | `true` | Toggle PII log redaction |

---

## License

This project is proprietary. All rights reserved.
