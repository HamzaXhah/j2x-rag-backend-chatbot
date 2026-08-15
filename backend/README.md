# RAG Backend

A FastAPI-based backend for a Retrieval-Augmented Generation (RAG) system that enables document ingestion, vector-based similarity search, and AI-powered question answering.

## Features

- 📄 **Document Processing**: Upload and process PDF, DOCX, TXT, Markdown, CSV, Excel, JSON, and images
- 🌐 **Web Scraping**: Index content from URLs
- 🗄️ **Database Connector**: Query and index data from PostgreSQL, MySQL, SQLite databases
- 🔍 **Vector Search**: Semantic similarity search using PGVector
- 🤖 **AI-Powered Q&A**: Generate answers using Azure OpenAI or OpenAI
- 📦 **Object Storage**: Store files in MinIO (S3-compatible)
- 🔄 **Real-time Updates**: WebSocket support for streaming responses

## Tech Stack

- **Framework**: FastAPI 0.115.5
- **Python**: 3.11
- **Vector Store**: PostgreSQL with pgvector
- **Embeddings**: Azure OpenAI / OpenAI (text-embedding-3-small)
- **LLM**: Azure OpenAI / OpenAI (GPT-4)
- **ORM**: SQLAlchemy 2.x
- **Package Manager**: uv (10-100x faster than pip)

---

## Quick Start

### Option 1: Docker (Recommended)

```bash
# From the project root directory
docker-compose up -d
```

The backend will be available at `http://localhost:8080`

### Option 2: Local Development with uv

#### Prerequisites

- Python 3.11+
- PostgreSQL with pgvector extension
- [uv](https://github.com/astral-sh/uv) package manager

#### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Setup Environment

```bash
cd backend

# Option A: Use the setup script (recommended)
./setup-env.sh          # macOS/Linux
setup-env.bat           # Windows

# Option B: Manual setup with uv sync
uv sync                 # Creates .venv, installs deps, generates uv.lock
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

#### Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings:
# - DATABASE_URL
# - DASHSCOPE_API_KEY
# - DASHSCOPE_BASE_URL for your Alibaba Cloud region
# - MINIO credentials

# Download local BGE Large model weights
uv run python scripts/download_bge_model.py
```

#### Run the Server

```bash
# With uv (recommended)
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8080

# Or with activated venv
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

---

## API Endpoints

### Health Check
```
GET  /api/v1/health          - System health status
```

### Documents
```
POST /api/v1/documents/upload    - Upload a file
POST /api/v1/documents/web       - Index a URL
POST /api/v1/documents/database  - Index database query results
GET  /api/v1/documents/          - List all documents
GET  /api/v1/documents/{id}      - Get document details
DELETE /api/v1/documents/{id}    - Delete a document
```

### Query
```
POST /api/v1/query/              - Query documents (RAG)
POST /api/v1/query/search        - Semantic search only
```

### Data Sources
```
POST /api/v1/datasources/        - Create a data source config
GET  /api/v1/datasources/        - List data sources
PUT  /api/v1/datasources/{id}    - Update data source
DELETE /api/v1/datasources/{id}  - Delete data source
```

---

## Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI application entry
│   ├── api/
│   │   └── endpoints/       # API route handlers
│   │       ├── documents.py
│   │       ├── query.py
│   │       ├── datasource.py
│   │       └── health.py
│   ├── core/
│   │   └── config.py        # Application settings
│   ├── db/
│   │   └── session.py       # Database connection
│   ├── models/
│   │   └── document.py      # SQLAlchemy models
│   ├── schemas/
│   │   └── schemas.py       # Pydantic schemas
│   └── services/
│       ├── document_processor.py  # Document processing
│       ├── vector_store.py        # PGVector operations
│       ├── object_storage.py      # MinIO storage
│       ├── parsers/               # File parsers
│       ├── connectors/            # Database/web connectors
│       └── retrieval/             # RAG retrieval & generation
├── requirements.txt
├── Dockerfile
├── setup-env.sh             # macOS/Linux setup script
├── setup-env.bat            # Windows setup script
└── setup_env.py             # Python setup script
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://...` |
| `DASHSCOPE_API_KEY` | Alibaba Cloud Model Studio API key | - |
| `DASHSCOPE_BASE_URL` | OpenAI-compatible Model Studio endpoint | Singapore shared endpoint |
| `QWEN_MODEL` | Alibaba Cloud chat model ID | `qwen3.7-plus` |
| `BGE_MODEL_NAME` | Hugging Face model used by the downloader | `BAAI/bge-large-en-v1.5` |
| `BGE_MODEL_PATH` | Local BGE model directory | `./models/bge-large-en-v1.5` |
| `BGE_DEVICE` | Inference device: cpu, cuda, or mps | `cpu` |
| `EMBEDDING_MODEL_DIMENSIONS` | BGE vector dimensions | `1024` |
| `MINIO_ENDPOINT` | MinIO server URL | `minio:9000` |
| `MINIO_ACCESS_KEY` | MinIO access key | `minioadmin` |
| `MINIO_SECRET_KEY` | MinIO secret key | `minioadmin` |
| `CHUNK_SIZE` | Text chunk size | `1000` |
| `CHUNK_OVERLAP` | Chunk overlap | `200` |
| `CHUNKING_STRATEGY` | Strategy: recursive, semantic, token, sentence | `recursive` |
| `SEMANTIC_BREAKPOINT_TYPE` | For semantic chunking: percentile, standard_deviation, interquartile, gradient | `percentile` |
| `TOKENIZER_ENCODING` | Encoding for token-based chunking | `cl100k_base` |
| `MIN_CHUNK_SIZE` | Minimum chunk size | `100` |
| `DISABLE_SSL_VERIFICATION` | Disable SSL for dev | `false` |

### Chunking Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `recursive` | Splits by characters with configurable separators | General-purpose, works well for most documents |
| `semantic` | Splits based on semantic similarity using embeddings | Better context preservation, varied document types |
| `token` | Splits based on token count | When working with strict LLM token limits |
| `sentence` | Splits at sentence boundaries | NLP tasks, conversational content |

---

## Development

### Adding Dependencies

```bash
# Add a new package to pyproject.toml and sync
uv add package-name

# Add a dev dependency
uv add --dev package-name

# Sync after manual pyproject.toml edits
uv sync
```

### Running Tests

```bash
uv run pytest
```

### Code Formatting

```bash
uv run black app/
uv run isort app/
```

---

## Docker Build

```bash
# Build development image
docker build --target development -t rag-backend:dev .

# Build production image
docker build --target production -t rag-backend:prod .

# Run container
docker run -p 8080:8080 --env-file .env rag-backend:dev
```

---

## API Documentation

Once running, access the interactive API docs at:
- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

---

## Troubleshooting

### Database Connection Issues
```bash
# Check if PostgreSQL is running
docker-compose ps db

# Check logs
docker-compose logs db
```

### SSL Certificate Errors (Alibaba Cloud Model Studio)
Set `DISABLE_SSL_VERIFICATION=true` in your `.env` file for development.

### Existing vector data
BGE Large vectors are 1024-dimensional. Reset the old vector database and index all documents again when migrating from OpenAI/Azure embeddings.

### Package Installation Slow
Make sure you're using `uv` instead of `pip` for 10-100x faster installs.

---

## License

MIT License - see [LICENSE](../LICENSE) for details.
