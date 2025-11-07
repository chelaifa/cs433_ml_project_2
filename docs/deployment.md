# Deployment Guide

## Table of Contents
1. [Vast.ai GPU Setup](#vastai-gpu-setup)
2. [Docker Container Deployment](#docker-container-deployment)
3. [Environment Configuration](#environment-configuration)
4. [Vector Database Setup](#vector-database-setup)
5. [API Deployment](#api-deployment)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)

## Vast.ai GPU Setup

### Prerequisites
- Vast.ai account with payment method
- SSH key pair for secure access
- Docker installed on your local machine

### Step 1: Create Vast.ai Account

1. Visit [https://vast.ai](https://vast.ai)
2. Sign up and add payment method
3. Generate API key from dashboard
4. Save API key as environment variable:
   ```bash
   export VASTAI_API_KEY='your_api_key_here'
   ```

### Step 2: Install Vast.ai CLI

```bash
pip install vastai
vastai set api-key $VASTAI_API_KEY
```

### Step 3: Search for Suitable GPU Instances

```bash
# Search for instances with specific requirements
vastai search offers \
  'gpu_ram >= 24 cuda_vers >= 11.8 num_gpus=1 reliability > 0.95' \
  --order 'dph+' \
  --limit 10
```

**Recommended specifications**:
- **GPU**: RTX 4090, A6000, or A100
- **VRAM**: ≥24GB
- **CUDA**: ≥11.8
- **Disk**: ≥100GB
- **Bandwidth**: ≥100 Mbps upload

### Step 4: Rent GPU Instance

```bash
# Rent instance by ID (replace 123456 with actual ID)
vastai create instance 123456 \
  --image pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime \
  --disk 100 \
  --env DOLPHIN_MODEL_PATH=/workspace/models
```

### Step 5: Connect to Instance

```bash
# Get instance details
vastai show instances

# SSH into instance
ssh -p <port> root@<host>
```

### Step 6: Automated Deployment Script

Create `scripts/deploy_vastai.py`:

```python
#!/usr/bin/env python3
"""
Automated Vast.ai deployment script for Dolphin model
"""

import subprocess
import json
import time
import os

class VastAIDeployer:
    def __init__(self, api_key):
        self.api_key = api_key
        os.environ['VASTAI_API_KEY'] = api_key

    def search_instances(self, min_gpu_ram=24, min_cuda_vers=11.8):
        """Search for suitable GPU instances"""
        cmd = [
            'vastai', 'search', 'offers',
            f'gpu_ram >= {min_gpu_ram} cuda_vers >= {min_cuda_vers} num_gpus=1',
            '--order', 'dph+',
            '--limit', '5',
            '--raw'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return json.loads(result.stdout)

    def create_instance(self, offer_id, image='pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime'):
        """Create a new instance"""
        cmd = [
            'vastai', 'create', 'instance', str(offer_id),
            '--image', image,
            '--disk', '100',
            '--env', 'DOLPHIN_MODEL_PATH=/workspace/models',
            '--raw'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return json.loads(result.stdout)

    def wait_for_instance(self, instance_id, timeout=300):
        """Wait for instance to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            cmd = ['vastai', 'show', 'instances', '--raw']
            result = subprocess.run(cmd, capture_output=True, text=True)
            instances = json.loads(result.stdout)

            for inst in instances:
                if inst['id'] == instance_id and inst['actual_status'] == 'running':
                    return inst

            time.sleep(10)

        raise TimeoutError(f"Instance {instance_id} did not start within {timeout}s")

    def deploy(self):
        """Main deployment workflow"""
        print("🔍 Searching for suitable GPU instances...")
        offers = self.search_instances()

        if not offers:
            raise Exception("No suitable instances found")

        best_offer = offers[0]
        print(f"✅ Found instance: {best_offer['gpu_name']} - ${best_offer['dph_total']:.2f}/hour")

        print(f"🚀 Creating instance {best_offer['id']}...")
        instance = self.create_instance(best_offer['id'])
        instance_id = instance['new_contract']

        print(f"⏳ Waiting for instance {instance_id} to be ready...")
        running_instance = self.wait_for_instance(instance_id)

        print(f"✅ Instance ready!")
        print(f"   SSH: ssh -p {running_instance['ssh_port']} root@{running_instance['ssh_host']}")
        print(f"   Direct: {running_instance['direct_port_start']}-{running_instance['direct_port_end']}")

        return running_instance

if __name__ == '__main__':
    api_key = os.getenv('VASTAI_API_KEY')
    if not api_key:
        raise ValueError("VASTAI_API_KEY environment variable not set")

    deployer = VastAIDeployer(api_key)
    instance = deployer.deploy()

    # Save instance details
    with open('config/vastai_instance.json', 'w') as f:
        json.dump(instance, f, indent=2)
```

## Docker Container Deployment

### Dockerfile for Dolphin Model

Create `docker/Dockerfile.dolphin`:

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Dolphin model (adjust based on actual model repository)
RUN mkdir -p /workspace/models && \
    cd /workspace/models && \
    git clone https://huggingface.co/bytedance/dolphin

# Copy application code
COPY src/ /workspace/src/
COPY scripts/ /workspace/scripts/

# Set environment variables
ENV PYTHONPATH=/workspace
ENV DOLPHIN_MODEL_PATH=/workspace/models/dolphin
ENV CUDA_VISIBLE_DEVICES=0

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run API server
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose Configuration

Create `docker/docker-compose.yml`:

```yaml
version: '3.8'

services:
  dolphin-api:
    build:
      context: ..
      dockerfile: docker/Dockerfile.dolphin
    container_name: dolphin-rag-api
    ports:
      - "8000:8000"
    volumes:
      - ../src:/workspace/src
      - ../models:/workspace/models
      - ../data:/workspace/data
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
      - PINECONE_ENVIRONMENT=${PINECONE_ENVIRONMENT}
      - GPU_MEMORY_LIMIT=24GB
      - BATCH_SIZE=4
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
    restart: unless-stopped
    shm_size: '16gb'

  # Optional: Local vector database (Weaviate)
  weaviate:
    image: semitechnologies/weaviate:1.23.0
    ports:
      - "8080:8080"
    environment:
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      ENABLE_MODULES: ''
      CLUSTER_HOSTNAME: 'node1'
    volumes:
      - weaviate_data:/var/lib/weaviate
    restart: unless-stopped

volumes:
  weaviate_data:
```

### Build and Deploy

```bash
# Build Docker image
cd docker
docker-compose build

# Start services
docker-compose up -d

# Check logs
docker-compose logs -f dolphin-api

# Stop services
docker-compose down
```

## Environment Configuration

Create `config/.env.example`:

```bash
# Vast.ai Configuration
VASTAI_API_KEY=your_vastai_api_key

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_MAX_RETRIES=3
OPENAI_TIMEOUT=60

# Vector Database - Pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=us-west1-gcp
PINECONE_INDEX_NAME=dolphin-rag-index

# Vector Database - Weaviate (Alternative)
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=optional_api_key

# Dolphin Model Configuration
DOLPHIN_MODEL_PATH=/workspace/models/dolphin
DOLPHIN_DEVICE=cuda
DOLPHIN_BATCH_SIZE=4
DOLPHIN_MAX_LENGTH=2048

# Document Processing
CHUNK_SIZE=1024
CHUNK_OVERLAP=100
MIN_CHUNK_SIZE=100
MAX_CHUNKS_PER_DOC=1000

# Retrieval Configuration
TOP_K_RETRIEVAL=50
TOP_N_RERANK=10
SIMILARITY_THRESHOLD=0.7

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
API_RELOAD=false

# Logging
LOG_LEVEL=INFO
LOG_FILE=/workspace/logs/app.log

# Resource Limits
GPU_MEMORY_LIMIT=24GB
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=300
```

Create actual `.env` file:
```bash
cp config/.env.example config/.env
# Edit config/.env with your actual credentials
```

## Vector Database Setup

### Option 1: Pinecone Setup

```python
# scripts/setup_vectordb.py

import os
import pinecone
from dotenv import load_dotenv

load_dotenv('config/.env')

def setup_pinecone():
    """Initialize Pinecone index"""

    # Initialize Pinecone
    pinecone.init(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment=os.getenv('PINECONE_ENVIRONMENT')
    )

    index_name = os.getenv('PINECONE_INDEX_NAME')

    # Check if index exists
    if index_name not in pinecone.list_indexes():
        print(f"Creating index: {index_name}")
        pinecone.create_index(
            name=index_name,
            dimension=3072,  # text-embedding-3-large
            metric='cosine',
            pods=1,
            pod_type='p1.x1'
        )
        print(f"✅ Index {index_name} created successfully")
    else:
        print(f"Index {index_name} already exists")

    # Get index stats
    index = pinecone.Index(index_name)
    stats = index.describe_index_stats()
    print(f"Index stats: {stats}")

if __name__ == '__main__':
    setup_pinecone()
```

Run setup:
```bash
python scripts/setup_vectordb.py
```

### Option 2: Weaviate Setup

```python
import weaviate
import os
from dotenv import load_dotenv

load_dotenv('config/.env')

def setup_weaviate():
    """Initialize Weaviate schema"""

    client = weaviate.Client(
        url=os.getenv('WEAVIATE_URL'),
        auth_client_secret=weaviate.AuthApiKey(api_key=os.getenv('WEAVIATE_API_KEY'))
    )

    # Define schema
    schema = {
        "classes": [{
            "class": "Document",
            "description": "A document chunk with embeddings",
            "vectorizer": "none",
            "properties": [
                {
                    "name": "chunk_text",
                    "dataType": ["text"],
                    "description": "The text content of the chunk"
                },
                {
                    "name": "document_id",
                    "dataType": ["string"],
                    "description": "Parent document ID"
                },
                {
                    "name": "document_title",
                    "dataType": ["string"],
                    "description": "Document title"
                },
                {
                    "name": "page_number",
                    "dataType": ["int"],
                    "description": "Page number in original document"
                },
                {
                    "name": "citations",
                    "dataType": ["string[]"],
                    "description": "Citations found in this chunk"
                },
                {
                    "name": "chunk_index",
                    "dataType": ["int"],
                    "description": "Chunk position in document"
                }
            ]
        }]
    }

    # Create schema
    try:
        client.schema.create(schema)
        print("✅ Weaviate schema created successfully")
    except Exception as e:
        print(f"Schema might already exist: {e}")

    # Print schema
    current_schema = client.schema.get()
    print(f"Current schema: {current_schema}")

if __name__ == '__main__':
    setup_weaviate()
```

## API Deployment

### On Vast.ai Instance

1. **SSH into instance**:
   ```bash
   ssh -p <port> root@<host>
   ```

2. **Transfer code**:
   ```bash
   # From local machine
   scp -P <port> -r . root@<host>:/workspace/dolphin-rag
   ```

3. **Build and run**:
   ```bash
   # On Vast.ai instance
   cd /workspace/dolphin-rag
   docker-compose -f docker/docker-compose.yml up -d
   ```

4. **Verify deployment**:
   ```bash
   curl http://localhost:8000/health
   ```

### Exposing API Externally

```bash
# On Vast.ai instance, use port forwarding
# Vast.ai automatically exposes ports, check instance details for public URL
```

## Monitoring and Maintenance

### Health Monitoring Script

```python
# scripts/monitor_health.py

import requests
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_health(url="http://localhost:8000/health"):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            logger.info("✅ API is healthy")
            return True
        else:
            logger.error(f"❌ API returned status {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return False

if __name__ == '__main__':
    while True:
        check_health()
        time.sleep(60)  # Check every minute
```

### Log Monitoring

```bash
# View API logs
docker logs -f dolphin-rag-api

# View last 100 lines
docker logs --tail 100 dolphin-rag-api

# Save logs to file
docker logs dolphin-rag-api > api_logs.txt
```

### Resource Monitoring

```bash
# GPU utilization
nvidia-smi -l 1

# Docker stats
docker stats

# Disk usage
df -h
```

### Backup and Restore

```bash
# Backup vector database (if using Weaviate locally)
docker run --rm --volumes-from weaviate -v $(pwd):/backup ubuntu tar cvf /backup/weaviate_backup.tar /var/lib/weaviate

# Restore
docker run --rm --volumes-from weaviate -v $(pwd):/backup ubuntu tar xvf /backup/weaviate_backup.tar
```

## Troubleshooting

### Common Issues

1. **Out of GPU memory**
   - Reduce `DOLPHIN_BATCH_SIZE`
   - Use model quantization
   - Process documents in smaller batches

2. **Slow PDF processing**
   - Increase GPU instance size
   - Optimize image preprocessing
   - Use multiple worker processes

3. **API timeouts**
   - Increase `REQUEST_TIMEOUT`
   - Implement request queuing
   - Add progress tracking

4. **Vector database connection errors**
   - Check network connectivity
   - Verify API keys
   - Check service status

### Support Resources

- Vast.ai Discord: [https://discord.gg/vastai](https://discord.gg/vastai)
- Pinecone Support: [https://www.pinecone.io/support/](https://www.pinecone.io/support/)
- Weaviate Slack: [https://weaviate.io/slack](https://weaviate.io/slack)

---

**Last Updated**: 2025-11-06
