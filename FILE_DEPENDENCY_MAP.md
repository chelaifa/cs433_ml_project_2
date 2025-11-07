# Project-2-RAG: File Dependency Map

## Source Code Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                       │
│                  src/api/main.py (379 lines)                 │
│                                                               │
│  Endpoints:                                                   │
│  ├─ GET  /health                    ────────────┐           │
│  ├─ POST /api/v1/documents/process ─────┐      │           │
│  ├─ GET  /api/v1/documents/{id}/markdown│      │           │
│  ├─ POST /api/v1/query              ─────┼──┐  │           │
│  ├─ GET  /api/v1/documents          ─────┼──┼──┤           │
│  ├─ DELETE /api/v1/documents/{id}   ─────┼──┼──┤           │
│  └─ GET  /api/v1/stats              ─────┼──┼──┤           │
└─────────────────────────────────────────┼──┼──┼─────────────┘
                                          │  │  │
                ┌─────────────────────────┘  │  │
                │                            │  │
                ▼                            │  │
┌────────────────────────────────────────┐  │  │
│   Document Processing (Dolphin)        │  │  │
│   src/dolphin/                         │  │  │
│                                        │  │  │
│  ┌──────────────────────────────────┐ │  │  │
│  │ model.py (259 lines)             │ │  │  │
│  │ DolphinModel class               │ │  │  │
│  │ ├─ _load_model()                 │ │  │  │
│  │ ├─ process_pdf_page()            │ │  │  │
│  │ ├─ process_pdf_document()        │ │  │  │
│  │ ├─ batch_process_pages()         │ │  │  │
│  │ ├─ _extract_citations()          │ │  │  │
│  │ └─ unload()                      │ │  │  │
│  └──────────────────────────────────┘ │  │  │
│                 ↑                      │  │  │
│                 │ uses                 │  │  │
│  ┌──────────────────────────────────┐ │  │  │
│  │ pdf_processor.py (219 lines)     │ │  │  │
│  │ PDFProcessor class               │ │  │  │
│  │ ├─ pdf_to_images()               │ │  │  │
│  │ ├─ preprocess_image()            │ │  │  │
│  │ ├─ extract_metadata()            │ │  │  │
│  │ ├─ extract_text_fallback()       │ │  │  │
│  │ └─ save_images()                 │ │  │  │
│  └──────────────────────────────────┘ │  │  │
│                                        │  │  │
│  citation_extractor.py (placeholder)   │  │  │
└────────────────────────────────────────┘  │  │
                │                           │  │
                │ markdown output           │  │
                ▼                           │  │
┌────────────────────────────────────────┐ │  │
│   Text Processing & Chunking           │ │  │
│   src/embeddings/                      │ │  │
│                                        │ │  │
│  ┌──────────────────────────────────┐ │ │  │
│  │ chunking.py (336 lines)          │ │ │  │
│  │ DocumentChunker class            │ │ │  │
│  │ ├─ semantic_chunking()           │ │ │  │
│  │ ├─ fixed_size_chunking()         │ │ │  │
│  │ ├─ recursive_chunking()          │ │ │  │
│  │ ├─ chunk_with_citations()        │ │ │  │
│  │ └─ _recursive_split()            │ │ │  │
│  └──────────────────────────────────┘ │ │  │
│                │                       │ │  │
│                │ chunks                │ │  │
│                ▼                       │ │  │
│  ┌──────────────────────────────────┐ │ │  │
│  │ openai_embedder.py (229 lines)   │ │ │  │
│  │ OpenAIEmbedder class             │ │ │  │
│  │ ├─ generate_embedding()          │ │ │  │
│  │ ├─ generate_embeddings_batch()   │ │ │  │
│  │ ├─ generate_chunks_with_emb()    │ │ │  │
│  │ ├─ get_embedding_dimension()     │ │ │  │
│  │ ├─ calculate_tokens()            │ │ │  │
│  │ └─ calculate_cost()              │ │ │  │
│  └──────────────────────────────────┘ │ │  │
└────────────────────────────────────────┘ │  │
                │                          │  │
                │ vectors + metadata       │  │
                ▼                          │  │
┌────────────────────────────────────────┐ │  │
│  Vector Database Layer                 │ │  │
│  src/vectordb/                         │ │  │
│                                        │ │  │
│  base.py (abstract base class)         │ │  │
│  pinecone_client.py (not yet impl.)    │ │  │
│  weaviate_client.py (not yet impl.)    │ │  │
└────────────────────────────────────────┘ │  │
                │                          │  │
                │ store / retrieve         │  │
                ▼                          │  │
        ┌──────────────────┐              │  │
        │  Vector Database │              │  │
        │ Pinecone/Weaviate│              │  │
        └──────────────────┘              │  │
                                          │  │
         ┌────────────────────────────────┘  │
         │ retrieval logic                    │
         ▼                                    │
┌────────────────────────────────────────┐  │
│  Retrieval & Reranking                 │  │
│  src/retrieval/ (not yet impl.)        │  │
│                                        │  │
│  retriever.py                          │  │
│  ├─ semantic_search()                  │  │
│  ├─ apply_filters()                    │  │
│  └─ apply_dedup()                      │  │
│                                        │  │
│  reranker.py                           │  │
│  ├─ cross_encoder_rerank()             │  │
│  ├─ bm25_rerank()                      │  │
│  └─ llm_based_rerank()                 │  │
│                                        │  │
│  query_processor.py                    │  │
│  ├─ preprocess_query()                 │  │
│  ├─ expand_query()                     │  │
│  └─ get_query_embedding()              │  │
└────────────────────────────────────────┘  │
                │                           │
                │ context + scores          │
                ▼                           │
         ┌──────────────────┐              │
         │  OpenAI (GPT-4)  │◄─────────────┘
         │  Optional: Resp. │
         │  Generation      │
         └──────────────────┘
                │
                │ final response
                ▼
         ┌──────────────────┐
         │ API Response     │
         │ (JSON with       │
         │  chunks,         │
         │  scores,         │
         │  citations)      │
         └──────────────────┘
```

## Configuration & Deployment Dependencies

```
┌─────────────────────────────────────────────────────────┐
│          Configuration Layer                             │
│                                                          │
│  config/config.yaml (234 lines, YAML)                  │
│  ├─ model.name: "bytedance/dolphin"                     │
│  ├─ model.device: "cuda"                                │
│  ├─ model.batch_size: 4                                 │
│  ├─ pdf_processing.dpi: 300                             │
│  ├─ chunking.strategy: "semantic"                       │
│  ├─ openai.model: "text-embedding-3-large"             │
│  ├─ vectordb.provider: "pinecone"                       │
│  ├─ retrieval.top_k: 50                                 │
│  ├─ reranking.model: "cross-encoder/ms-marco-..."      │
│  ├─ generation.model: "gpt-4-turbo-preview"            │
│  └─ [many more settings...]                             │
│                                                          │
│  config/.env.example (40+ env vars)                     │
│  ├─ VASTAI_API_KEY                                      │
│  ├─ OPENAI_API_KEY                                      │
│  ├─ PINECONE_API_KEY / WEAVIATE_URL                     │
│  ├─ DOLPHIN_MODEL_PATH                                  │
│  └─ [many more...]                                      │
└─────────────────────────────────────────────────────────┘
         │
         ├────────────────────────────────────┐
         │                                    │
         ▼                                    ▼
┌─────────────────────────────┐  ┌──────────────────────┐
│   Docker Deployment         │  │  Vast.ai Integration │
│                             │  │                      │
│  docker/                    │  │  scripts/            │
│  ├─ Dockerfile.dolphin      │  │                      │
│  │  └─ Base: pytorch:2.1.0  │  │  deploy_vastai.py    │
│  │     CUDA 11.8            │  │  ├─ search_instances()
│  │     + deps from req.txt   │  │  ├─ create_instance()
│  │     + Dolphin model       │  │  ├─ wait_for_instance()
│  │     + src/ code           │  │  └─ deploy()
│  │                           │  │
│  ├─ docker-compose.yml      │  │  setup_vectordb.py
│  │  ├─ dolphin-api:8000     │  │  ├─ setup_pinecone()
│  │  │  ├─ GPU: nvidia       │  │  └─ setup_weaviate()
│  │  │  ├─ RAM: 16GB         │  │
│  │  │  └─ volumes: models,  │  │  monitor_health.py
│  │  │      data, src        │  │  └─ check_health()
│  │  │                       │  │
│  │  └─ weaviate:8080        │  │  batch_process.py
│  │     (optional)           │  │  └─ process_pdfs()
│  │                          │  │
│  └─ .dockerignore           │  └──────────────────────┘
│                             │
└─────────────────────────────┘
```

## Data Files & Storage

```
data/
├─ documents/                    # Uploaded PDFs
│  ├─ research_paper_1.pdf
│  ├─ academic_study_2.pdf
│  └─ [...]
│
├─ markdown/                     # Processed markdown
│  ├─ research_paper_1.md
│  └─ [...]
│
├─ temp/                         # Temporary processing
│  ├─ pdf_images_<timestamp>/
│  │  ├─ page_0001.png
│  │  ├─ page_0002.png
│  │  └─ [...]
│  └─ [...]
│
└─ metadata.db                   # Local metadata (SQLite)
   ├─ documents table
   ├─ chunks table
   ├─ embeddings metadata
   └─ citations table

pdf-parser-comparison/
├─ models/dolphin-1.5/           # Dolphin model weights
│  └─ [HuggingFace model files]
│
├─ sample_pdfs/                  # Test PDFs (7 papers)
├─ outputs/                      # Parser outputs (JSON)
├─ reports/                      # HTML reports
└─ [comparison scripts]
```

## External Service Integrations

```
┌──────────────────────────────────────────────────────────┐
│                External Services                         │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │ HuggingFace Hub                                    │ │
│  │ └─ ByteDance/Dolphin-1.5 (model download)          │ │
│  │    Used by: src/dolphin/model.py                   │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │ OpenAI API                                         │ │
│  │ ├─ Embeddings (text-embedding-3-large)             │ │
│  │ │  Used by: src/embeddings/openai_embedder.py     │ │
│  │ │  Cost: $0.13 per 1M tokens                      │ │
│  │ │                                                  │ │
│  │ └─ Generation (gpt-4-turbo, optional)              │ │
│  │    Used by: src/rag/generator.py (not yet impl.)  │ │
│  │    Cost: $0.03-0.06 per 1K tokens                │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Vector Database                                    │ │
│  │ ├─ Option A: Pinecone (Cloud)                      │ │
│  │ │  Used by: src/vectordb/pinecone_client.py       │ │
│  │ │  Cost: ~$70/month (1M vectors)                  │ │
│  │ │                                                  │ │
│  │ └─ Option B: Weaviate (Self-hosted)                │ │
│  │    Used by: src/vectordb/weaviate_client.py       │ │
│  │    Cost: ~$25/month cloud / $0 self-hosted        │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Vast.ai (GPU Compute)                              │ │
│  │ ├─ Instance rental (RTX 4090, A6000, A100)        │ │
│  │ │  Cost: $0.30-2.00 per hour                      │ │
│  │ │  Used by: scripts/deploy_vastai.py              │ │
│  │ │                                                  │ │
│  │ └─ Docker deployment via SSH                       │ │
│  └────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

## Dependency Graph (Key Python Imports)

```
FastAPI Application Entry Point:
  src/api/main.py
  │
  ├─ Imports:
  │  ├─ fastapi
  │  ├─ pydantic
  │  ├─ logging
  │  └─ typing
  │
  ├─ Uses:
  │  ├─ src.dolphin.model.DolphinModel
  │  ├─ src.dolphin.pdf_processor.PDFProcessor
  │  ├─ src.embeddings.chunking.DocumentChunker
  │  ├─ src.embeddings.openai_embedder.OpenAIEmbedder
  │  ├─ src.vectordb (pinecone or weaviate)
  │  ├─ src.retrieval (search & reranking)
  │  └─ src.rag (pipeline orchestration)
  │
  └─ Config:
     └─ config/config.yaml
        └─ config/.env

Third-party Dependencies:
  ├─ torch (PyTorch)
  ├─ transformers (HuggingFace)
  ├─ pdf2image
  ├─ PIL (Pillow)
  ├─ fastapi
  ├─ uvicorn
  ├─ openai
  ├─ pinecone
  ├─ weaviate
  ├─ sentence_transformers
  ├─ rank_bm25
  ├─ faiss
  ├─ redis
  ├─ sqlalchemy
  └─ [40+ more]
```

## Testing & CI/CD (Planned)

```
tests/
├─ test_api.py
│  ├─ test_health_endpoint()
│  ├─ test_document_process()
│  ├─ test_query_endpoint()
│  └─ test_error_handling()
│
├─ test_dolphin.py
│  ├─ test_model_loading()
│  ├─ test_pdf_processing()
│  └─ test_citation_extraction()
│
├─ test_chunking.py
│  ├─ test_semantic_chunking()
│  ├─ test_fixed_size_chunking()
│  └─ test_chunk_with_citations()
│
├─ test_embeddings.py
│  ├─ test_single_embedding()
│  ├─ test_batch_embeddings()
│  └─ test_cost_calculation()
│
└─ test_retrieval.py
   ├─ test_semantic_search()
   ├─ test_reranking()
   └─ test_citation_preservation()

Integration Tests (not yet created):
  ├─ End-to-end PDF processing
  ├─ Full RAG pipeline test
  ├─ Load testing
  └─ Error recovery tests
```

---

This architecture supports:
- **Horizontal scaling**: Multiple instances via load balancer
- **Vertical scaling**: Larger GPU instances on Vast.ai
- **Modular design**: Components can be replaced (e.g., different vector DB)
- **Production deployment**: Docker containers + orchestration
- **Monitoring**: Health checks, logging, metrics
