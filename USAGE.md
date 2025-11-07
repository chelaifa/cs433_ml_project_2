# RAG Pipeline Usage Guide

## Installation

```bash
# Install with UV
uv sync

# Or install in development mode
uv pip install -e .
```

## CLI Commands

### 1. Fetch Metadata from OpenAlex

Fetch academic paper metadata from OpenAlex API using filters:

```bash
# Basic usage - uses default filters (topic:t10856 + open access)
uv run rag-pipeline fetch-metadata

# With email for faster API (polite pool)
uv run rag-pipeline fetch-metadata --email your@email.com

# With custom filters
uv run rag-pipeline fetch-metadata \
  --filter "primary_topic.id:t10856,open_access.is_oa:true" \
  --email your@email.com

# Different topic ID
uv run rag-pipeline fetch-metadata \
  --filter "primary_topic.id:t11234,publication_year:2023"

# Custom output directory
uv run rag-pipeline fetch-metadata \
  --filter "open_access.is_oa:true,publication_year:2023" \
  --output data/openalex \
  --email your@email.com
```

**Options:**
- `--filter, -f`: Comma-separated filters (default: `primary_topic.id:t10856,open_access.is_oa:true`)
  - Examples: `publication_year:2023`, `is_oa:true`, `primary_topic.id:t10856`
- `--output, -o`: Output directory (default: `openalex_data`)
- `--email, -e`: Email for polite pool (faster API access)
- `--per-page`: Results per page, max 200 (default: 200)

### 2. Download PDFs

Download PDFs from the metadata you fetched:

```bash
# Download PDFs from metadata file
uv run rag-pipeline download-pdfs data/openalex/metadata/works.parquet

# With options
uv run rag-pipeline download-pdfs data/openalex/metadata/works.parquet \
  --max 100 \
  --workers 10 \
  --output data/openalex/pdfs
```

**Options:**
- `--max, -m`: Maximum number of PDFs to download
- `--workers, -w`: Number of concurrent downloads (default: 5)
- `--output, -o`: Output directory (default: `data/openalex/pdfs`)

### 3. Parse PDFs with Dolphin

Parse PDFs to markdown using the Dolphin model:

```bash
# Parse all PDFs in a directory
uv run rag-pipeline parse-pdfs data/openalex/pdfs

# With custom options
uv run rag-pipeline parse-pdfs data/openalex/pdfs \
  --output data/parsed \
  --model ByteDance/Dolphin \
  --device cuda
```

**Options:**
- `--output, -o`: Output directory for markdown (default: `data/parsed`)
- `--model, -m`: Dolphin model path (default: `ByteDance/Dolphin`)
- `--device, -d`: Device to use (cuda, cpu, mps) (default: `cuda`)

### 4. Create Embeddings

Generate embeddings from parsed markdown:

```bash
# Create embeddings
uv run rag-pipeline create-embeddings data/parsed

# With custom settings
uv run rag-pipeline create-embeddings data/parsed \
  --output data/embeddings/embeddings.parquet \
  --chunk-size 1024 \
  --overlap 100 \
  --model text-embedding-3-large
```

**Options:**
- `--output, -o`: Output parquet file (default: `data/embeddings/embeddings.parquet`)
- `--chunk-size, -c`: Chunk size in tokens (default: 1024)
- `--overlap`: Overlap between chunks (default: 100)
- `--model, -m`: OpenAI embedding model (default: `text-embedding-3-large`)

### 5. Pipeline Information

View pipeline information:

```bash
uv run rag-pipeline info
```

## Complete Workflow Example

Here's a complete example workflow:

```bash
# 1. Fetch metadata with filters (default saves to openalex_data/)
uv run rag-pipeline fetch-metadata \
  --filter "primary_topic.id:t10856,open_access.is_oa:true" \
  --email your@email.com

# 2. Download the PDFs
uv run rag-pipeline download-pdfs openalex_data/openalex_works.parquet \
  --max 50 \
  --workers 5 \
  --output openalex_data/pdfs

# 3. Parse PDFs to markdown (requires Dolphin model and GPU)
uv run rag-pipeline parse-pdfs openalex_data/pdfs \
  --output data/parsed \
  --device cuda

# 4. Create embeddings (requires OPENAI_API_KEY in .env)
uv run rag-pipeline create-embeddings data/parsed \
  --chunk-size 1024 \
  --model text-embedding-3-large \
  --output data/embeddings/embeddings.parquet
```

## Using as Python Library

You can also import and use the components directly:

```python
from rag_pipeline import MetadataFetcher, PDFDownloader, DolphinModel, OpenAIEmbedder
from rag_pipeline.openalex.config import OpenAlexConfig

# Fetch metadata
config = OpenAlexConfig(email="your@email.com")
fetcher = MetadataFetcher(config)
df = fetcher.fetch_and_save(query="machine learning", limit=100)

# Download PDFs
downloader = PDFDownloader(config)
results = downloader.download_from_metadata("metadata.parquet")

# Parse PDFs
model = DolphinModel(model_path="ByteDance/Dolphin", device="cuda")
markdown = model.process_pdf_document(images)

# Create embeddings
embedder = OpenAIEmbedder(model="text-embedding-3-large")
embeddings = embedder.generate_embeddings_batch(texts)
```

## Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI API key (required for embeddings)
OPENAI_API_KEY=your_openai_api_key

# Optional: OpenAlex email for polite pool
OPENALEX_EMAIL=your@email.com

# Optional: Vector database credentials
WEAVIATE_URL=http://localhost:8080
PINECONE_API_KEY=your_pinecone_key
```

## Data Directory Structure

After running the pipeline, your data directory will look like:

```
data/
├── openalex/
│   ├── metadata/
│   │   └── openalex_works.parquet
│   └── pdfs/
│       ├── paper1.pdf
│       ├── paper2.pdf
│       └── ...
├── parsed/
│   ├── paper1.md
│   ├── paper2.md
│   └── ...
└── embeddings/
    └── embeddings.parquet
```

## Tips

1. **Start small**: Test with `--limit 10` first to verify everything works
2. **Use email**: Provide your email with `--email` for faster API access (polite pool)
3. **Filter wisely**: Use filters like `publication_year:2023,is_oa:true` to get only what you need
4. **Monitor GPU**: Use `nvidia-smi` to monitor GPU usage during parsing
5. **Cost awareness**: OpenAI embeddings cost money - text-embedding-3-large is $0.13 per 1M tokens

## Troubleshooting

### Missing dependencies
```bash
uv sync  # Re-sync dependencies
```

### Import errors
```bash
uv pip install -e .  # Reinstall in editable mode
```

### OpenAI API errors
Check your `.env` file has `OPENAI_API_KEY` set

### GPU not detected
Use `--device cpu` for CPU-only parsing (slower)