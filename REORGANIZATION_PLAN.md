# Repository Reorganization Plan

## 🎯 Goal

Reorganize the repository into a clean, production-ready **RAG Pipeline Monorepo** with:
- **FastAPI** backend for REST API
- **LlamaIndex** for RAG orchestration
- **Weaviate** as vector database
- **OpenAlex fetcher** for data ingestion
- **PDF parsers** (Dolphin/Docling/GROBID) for document processing
- **Clean architecture** with proper separation of concerns

## 📊 Current State Analysis

### Existing Components

✅ **Good to Keep:**
- `openalex_fetcher/` - Professional metadata fetcher with Pydantic models
- `pdf-parser-comparison/` - Working PDF parsers (Dolphin, Docling, GROBID)
- `src/embeddings/` - OpenAI embeddings and chunking
- `src/dolphin/` - PDF processing utilities
- `docs/` - Comprehensive documentation

❌ **Needs Refactoring:**
- Root-level scripts (`fetch_metadata.py`, `download_pdfs.py`) → Move to CLI module
- Mixed concerns in `src/` directory → Reorganize by domain
- Missing LlamaIndex integration
- Missing Weaviate integration
- Old FastAPI placeholder (`src/api/main.py`) → Rebuild properly

## 🏗️ New Architecture

### Directory Structure

```
project-2-rag/
│
├── 📦 backend/                         # FastAPI Backend
│   ├── __init__.py
│   ├── main.py                         # FastAPI app entry point
│   ├── api/                            # API routes
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── health.py              # Health check endpoints
│   │   │   ├── documents.py           # Document upload/management
│   │   │   ├── search.py              # RAG search/query endpoints
│   │   │   └── admin.py               # Admin endpoints
│   │   └── dependencies.py            # FastAPI dependencies
│   ├── models/                         # Pydantic models for API
│   │   ├── __init__.py
│   │   ├── requests.py                # Request models
│   │   ├── responses.py               # Response models
│   │   └── documents.py               # Document models
│   ├── middleware/                     # FastAPI middleware
│   │   ├── __init__.py
│   │   ├── auth.py                    # Authentication
│   │   ├── cors.py                    # CORS handling
│   │   └── logging.py                 # Request logging
│   └── config.py                       # Backend configuration
│
├── 📚 rag/                             # RAG Engine (LlamaIndex)
│   ├── __init__.py
│   ├── engine.py                       # Main RAG engine
│   ├── indexing/                       # Document indexing
│   │   ├── __init__.py
│   │   ├── indexer.py                 # LlamaIndex indexer
│   │   ├── chunking.py                # Document chunking strategies
│   │   └── metadata.py                # Metadata extraction
│   ├── retrieval/                      # Retrieval logic
│   │   ├── __init__.py
│   │   ├── retriever.py               # LlamaIndex retriever
│   │   ├── reranker.py                # Reranking logic
│   │   └── filters.py                 # Query filters
│   ├── generation/                     # Response generation
│   │   ├── __init__.py
│   │   ├── generator.py               # LLM response generation
│   │   └── prompts.py                 # Prompt templates
│   └── config.py                       # RAG configuration
│
├── 🗄️ vectordb/                        # Vector Database
│   ├── __init__.py
│   ├── base.py                         # Abstract vector DB interface
│   ├── weaviate_client.py             # Weaviate implementation
│   ├── schemas/                        # Weaviate schemas
│   │   ├── __init__.py
│   │   ├── document.py                # Document schema
│   │   └── chunk.py                   # Chunk schema
│   └── config.py                       # Vector DB configuration
│
├── 📄 ingestion/                       # Data Ingestion Pipeline
│   ├── __init__.py
│   ├── pipeline.py                     # Main ingestion pipeline
│   ├── sources/                        # Data sources
│   │   ├── __init__.py
│   │   ├── openalex/                  # OpenAlex integration
│   │   │   ├── __init__.py
│   │   │   ├── fetcher.py             # (from openalex_fetcher)
│   │   │   ├── downloader.py
│   │   │   ├── models.py
│   │   │   ├── config.py
│   │   │   └── utils.py
│   │   ├── local.py                   # Local file uploads
│   │   └── s3.py                      # S3 bucket (future)
│   ├── parsers/                        # PDF parsers
│   │   ├── __init__.py
│   │   ├── base.py                    # Abstract parser interface
│   │   ├── dolphin.py                 # Dolphin parser
│   │   ├── docling.py                 # Docling parser
│   │   ├── grobid.py                  # GROBID parser
│   │   └── selector.py                # Parser selection logic
│   └── processors/                     # Post-processing
│       ├── __init__.py
│       ├── cleaner.py                 # Text cleaning
│       ├── enricher.py                # Metadata enrichment
│       └── validator.py               # Document validation
│
├── 🔧 core/                            # Core Utilities
│   ├── __init__.py
│   ├── config.py                       # Global configuration
│   ├── logging.py                      # Logging setup
│   ├── exceptions.py                   # Custom exceptions
│   └── utils.py                        # Shared utilities
│
├── 🗂️ storage/                         # File Storage
│   ├── __init__.py
│   ├── local.py                        # Local filesystem
│   ├── s3.py                          # S3 storage (optional)
│   └── manager.py                      # Storage manager
│
├── 🔌 cli/                             # Command Line Interface
│   ├── __init__.py
│   ├── main.py                         # CLI entry point (Typer)
│   ├── commands/
│   │   ├── __init__.py
│   │   ├── ingest.py                  # Ingestion commands
│   │   ├── index.py                   # Indexing commands
│   │   ├── query.py                   # Query commands
│   │   └── admin.py                   # Admin commands
│   └── utils.py                        # CLI utilities
│
├── 🧪 tests/                           # Test Suite
│   ├── __init__.py
│   ├── conftest.py                     # Pytest fixtures
│   ├── unit/                           # Unit tests
│   │   ├── test_parsers.py
│   │   ├── test_indexing.py
│   │   └── test_retrieval.py
│   ├── integration/                    # Integration tests
│   │   ├── test_api.py
│   │   ├── test_rag_pipeline.py
│   │   └── test_vectordb.py
│   └── e2e/                           # End-to-end tests
│       └── test_full_pipeline.py
│
├── 🐳 docker/                          # Docker Configuration
│   ├── Dockerfile.backend             # Backend container
│   ├── Dockerfile.worker              # Worker container (future)
│   ├── docker-compose.yml             # Local development
│   └── docker-compose.prod.yml        # Production setup
│
├── 📋 config/                          # Configuration Files
│   ├── config.yaml                     # Main configuration
│   ├── logging.yaml                    # Logging configuration
│   └── weaviate/                       # Weaviate configs
│       └── schema.json                 # Weaviate schema
│
├── 📚 docs/                            # Documentation
│   ├── api.md                          # API documentation
│   ├── deployment.md                   # Deployment guide
│   ├── examples.md                     # Usage examples
│   ├── architecture.md                 # Architecture overview
│   └── contributing.md                 # Contribution guide
│
├── 📊 data/                            # Data Directory (gitignored)
│   ├── pdfs/                          # Downloaded PDFs
│   ├── parsed/                        # Parsed documents
│   ├── metadata/                      # Metadata files
│   └── cache/                         # Cache files
│
├── 🚀 scripts/                         # Utility Scripts
│   ├── setup_weaviate.py              # Weaviate initialization
│   ├── migrate_data.py                # Data migration
│   └── benchmark.py                    # Performance benchmarks
│
├── 📝 Root Files
│   ├── README.md                       # Main README
│   ├── requirements.txt                # Python dependencies
│   ├── pyproject.toml                  # Project metadata
│   ├── .env.example                    # Environment template
│   ├── .gitignore                      # Git ignore rules
│   └── Makefile                        # Common commands
│
└── 🗑️ archive/                         # Old code (to be removed)
    ├── download_openalex_pdfs.py       # → cli/commands/ingest.py
    ├── fetch_metadata.py               # → cli/commands/ingest.py
    └── src/                            # Old src/ → refactored
```

## 🔄 Migration Steps

### Phase 1: Core Infrastructure (Week 1)

**Step 1.1: Create New Directory Structure**
- Create all new directories
- Move `data/` and ensure it's in `.gitignore`

**Step 1.2: Set Up Core Module**
- Create `core/config.py` with unified configuration
- Create `core/logging.py` with Loguru setup
- Create `core/exceptions.py` with custom exceptions

**Step 1.3: Migrate OpenAlex Fetcher**
- Move `openalex_fetcher/` → `ingestion/sources/openalex/`
- Keep all existing functionality
- Update imports

### Phase 2: Vector Database Integration (Week 1)

**Step 2.1: Weaviate Setup**
- Create `vectordb/weaviate_client.py`
- Define schemas in `vectordb/schemas/`
- Create Docker Compose with Weaviate

**Step 2.2: Abstract Interface**
- Create `vectordb/base.py` for future DB support
- Implement Weaviate client

### Phase 3: PDF Processing (Week 2)

**Step 3.1: Parser Abstraction**
- Create `ingestion/parsers/base.py`
- Migrate parsers from `pdf-parser-comparison/`
- Create `ingestion/parsers/selector.py` for automatic selection

**Step 3.2: Ingestion Pipeline**
- Create `ingestion/pipeline.py`
- Connect: Download → Parse → Process → Store

### Phase 4: RAG Engine with LlamaIndex (Week 2-3)

**Step 4.1: Indexing**
- Create `rag/indexing/indexer.py` using LlamaIndex
- Integrate with Weaviate
- Create chunking strategies

**Step 4.2: Retrieval**
- Create `rag/retrieval/retriever.py`
- Implement reranking
- Add filters and metadata search

**Step 4.3: Generation**
- Create `rag/generation/generator.py`
- Define prompt templates
- Integrate with OpenAI/Anthropic

### Phase 5: FastAPI Backend (Week 3)

**Step 5.1: API Structure**
- Create `backend/main.py`
- Define all routes in `backend/api/routes/`
- Add middleware (auth, CORS, logging)

**Step 5.2: Request/Response Models**
- Create Pydantic models in `backend/models/`
- Add validation

**Step 5.3: Integration**
- Connect API to RAG engine
- Connect API to ingestion pipeline

### Phase 6: CLI & Testing (Week 4)

**Step 6.1: CLI with Typer**
- Create `cli/main.py`
- Add commands for all operations
- Make it user-friendly

**Step 6.2: Testing**
- Write unit tests
- Write integration tests
- Add E2E tests

**Step 6.3: Docker & Deployment**
- Create production Docker images
- Write deployment guides
- Add monitoring

## 🛠️ Technology Stack

### Backend
- **FastAPI** - Modern, fast web framework
- **Pydantic** - Data validation
- **Uvicorn** - ASGI server

### RAG Framework
- **LlamaIndex** - RAG orchestration
- **OpenAI** - Embeddings & LLM
- **Anthropic** - Alternative LLM

### Vector Database
- **Weaviate** - Primary vector DB
- Docker setup for local development

### PDF Processing
- **Dolphin** - Multimodal understanding
- **Docling** - IBM's parser
- **GROBID** - Scientific papers

### Data Ingestion
- **OpenAlex** - Academic paper metadata
- **Requests** - HTTP client
- **Pandas** - Data processing

### CLI
- **Typer** - CLI framework
- **Rich** - Beautiful terminal output

### Testing
- **Pytest** - Testing framework
- **Pytest-asyncio** - Async testing

## 📦 Updated Dependencies

```txt
# Core
python>=3.10

# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6

# RAG Framework
llama-index==0.9.14
llama-index-vector-stores-weaviate==0.1.0
llama-index-embeddings-openai==0.1.0

# Vector Database
weaviate-client==3.25.3

# LLM Providers
openai==1.3.5
anthropic==0.7.1

# PDF Processing
pdf2image==1.16.3
pypdf==3.17.0
pdfplumber==0.10.3

# Data Processing
pandas==2.1.3
pyarrow==14.0.1
pydantic==2.5.0
pydantic-settings==2.1.0

# Utilities
python-dotenv==1.0.0
loguru==0.7.2
typer[all]==0.9.0
rich==13.7.0

# HTTP
requests==2.31.0
httpx==0.25.1

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
```

## ✅ Success Criteria

- [ ] Clean separation of concerns
- [ ] All tests passing
- [ ] FastAPI backend working
- [ ] LlamaIndex RAG pipeline functional
- [ ] Weaviate integration complete
- [ ] OpenAlex ingestion working
- [ ] PDF parsers integrated
- [ ] CLI commands functional
- [ ] Docker Compose for local development
- [ ] Comprehensive documentation

## 🎯 Next Steps

1. **Review this plan** - Get approval
2. **Create branch** - `git checkout -b refactor/monorepo-structure`
3. **Start Phase 1** - Core infrastructure
4. **Incremental testing** - Test after each phase
5. **Update documentation** - Keep docs in sync

---

**Estimated Timeline**: 3-4 weeks
**Risk Level**: Medium (careful migration needed)
**Benefit**: Clean, maintainable, production-ready codebase