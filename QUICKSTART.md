# Quick Start Guide

Get your Dolphin RAG system up and running in minutes.

## Prerequisites

- Python 3.10 or higher
- NVIDIA GPU with 24GB+ VRAM (for Dolphin model)
- Vast.ai account
- OpenAI API key
- Pinecone or Weaviate account

## Step 1: Clone and Setup

```bash
# Navigate to your project directory
cd /Users/eliebruno/Desktop/code/project-2-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Configure Environment

```bash
# Copy example environment file
cp config/.env.example config/.env

# Edit config/.env with your credentials:
# - VASTAI_API_KEY
# - OPENAI_API_KEY
# - PINECONE_API_KEY (or WEAVIATE_URL)
```

## Step 3: Deploy to Vast.ai

```bash
# Option A: Automatic deployment
python scripts/deploy_vastai.py

# Option B: Manual deployment
# 1. Go to https://vast.ai
# 2. Search for GPU instance (RTX 4090 or better)
# 3. Rent instance and note SSH details
# 4. Connect: ssh -p <port> root@<host>
```

## Step 4: Setup Vector Database

```bash
# For Pinecone:
python scripts/setup_vectordb.py

# For Weaviate (if using Docker):
docker-compose -f docker/docker-compose.yml up -d weaviate
```

## Step 5: Deploy Docker Container (on Vast.ai)

After connecting to your Vast.ai instance:

```bash
# Transfer your code
scp -P <port> -r . root@<host>:/workspace/dolphin-rag

# SSH into instance
ssh -p <port> root@<host>

# Navigate to project
cd /workspace/dolphin-rag

# Build and start services
docker-compose -f docker/docker-compose.yml build
docker-compose -f docker/docker-compose.yml up -d

# Check status
docker-compose logs -f dolphin-api
```

## Step 6: Test the API

```bash
# Health check
curl http://localhost:8000/health

# Process a PDF
curl -X POST http://localhost:8000/api/v1/documents/process \
  -H "Authorization: Bearer your_api_key" \
  -F "file=@sample.pdf"

# Query the system
curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?", "top_k": 10}'
```

## Step 7: Use Python SDK

```python
from src.api.client import DolphinRAG  # You'll need to create this

# Initialize client
rag = DolphinRAG(
    base_url="http://vastai-instance-url:8000",
    api_key="your_api_key"
)

# Process a PDF
result = rag.process_pdf("research_paper.pdf")
print(f"Processed: {result['document_id']}")

# Query
results = rag.query("What are the main findings?", generate=True)
print(results['generated_response'])
```

## Common Issues

### GPU Out of Memory
```bash
# Reduce batch size in config/config.yaml
model:
  batch_size: 2  # Down from 4
```

### Connection Timeout
```bash
# Increase timeout in config/config.yaml
resources:
  request_timeout: 600  # Up from 300
```

### Rate Limit Errors
```bash
# Adjust OpenAI rate limits
openai:
  rate_limit:
    requests_per_minute: 500  # Down from 3000
```

## Next Steps

1. **Process Your Documents**: Upload your PDFs and build your knowledge base
2. **Tune Parameters**: Adjust chunking, retrieval, and reranking settings
3. **Integrate**: Use the API in your applications
4. **Scale**: Add more GPU instances or use larger vector database tiers

## Useful Commands

```bash
# View logs
docker logs -f dolphin-rag-api

# Monitor GPU
nvidia-smi -l 1

# Check vector database stats
curl http://localhost:8000/api/v1/stats \
  -H "Authorization: Bearer your_api_key"

# Stop services
docker-compose down

# Restart services
docker-compose restart
```

## Getting Help

- **Documentation**: See `docs/` folder
- **API Reference**: http://localhost:8000/docs (when running)
- **Issues**: Check GitHub issues or Vast.ai Discord

## Production Checklist

Before going to production:

- [ ] Set strong API keys
- [ ] Enable rate limiting
- [ ] Configure monitoring (Prometheus/Sentry)
- [ ] Set up automated backups
- [ ] Implement proper authentication
- [ ] Add request logging
- [ ] Configure CORS properly
- [ ] Set up SSL/TLS
- [ ] Test error handling
- [ ] Load test the system

---

**You're ready to go! 🚀**
