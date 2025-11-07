# API Documentation

## Overview

The Dolphin RAG system exposes a RESTful API for document processing, embedding generation, and semantic search.

## Base URL

```
http://<vastai-host>:<port>
```

## Authentication

All endpoints require API key authentication via header:

```
Authorization: Bearer <your-api-key>
```

## Endpoints

### 1. Health Check

Check API status and resource availability.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_memory_free": "20.5 GB",
  "model_loaded": true,
  "vectordb_connected": true,
  "timestamp": "2025-11-06T10:30:00Z"
}
```

---

### 2. Process PDF Document

Upload and process a PDF document into markdown with citations.

**Endpoint**: `POST /api/v1/documents/process`

**Content-Type**: `multipart/form-data`

**Parameters**:
- `file` (required): PDF file (max 50MB)
- `extract_citations` (optional): Boolean, default `true`
- `generate_embeddings` (optional): Boolean, default `true`
- `chunk_size` (optional): Integer, default `1024`
- `chunk_overlap` (optional): Integer, default `100`

**Example Request**:
```bash
curl -X POST http://localhost:8000/api/v1/documents/process \
  -H "Authorization: Bearer <your-api-key>" \
  -F "file=@research_paper.pdf" \
  -F "extract_citations=true" \
  -F "generate_embeddings=true"
```

**Response**:
```json
{
  "document_id": "doc_abc123xyz",
  "status": "success",
  "filename": "research_paper.pdf",
  "num_pages": 15,
  "num_chunks": 47,
  "citations": [
    "Smith et al. (2023)",
    "Johnson and Lee (2022)",
    "Brown (2021)"
  ],
  "markdown_url": "/api/v1/documents/doc_abc123xyz/markdown",
  "processing_time_seconds": 45.3,
  "embeddings_generated": true,
  "metadata": {
    "title": "Machine Learning in Healthcare",
    "authors": ["Dr. Jane Doe"],
    "year": 2024,
    "pages": 15
  }
}
```

**Error Responses**:
```json
{
  "error": "File too large",
  "max_size_mb": 50,
  "status_code": 413
}
```

---

### 3. Get Document Markdown

Retrieve processed markdown for a document.

**Endpoint**: `GET /api/v1/documents/{document_id}/markdown`

**Parameters**:
- `document_id` (required): Document identifier

**Example Request**:
```bash
curl http://localhost:8000/api/v1/documents/doc_abc123xyz/markdown \
  -H "Authorization: Bearer <your-api-key>"
```

**Response**:
```json
{
  "document_id": "doc_abc123xyz",
  "markdown": "# Machine Learning in Healthcare\n\n## Abstract\n...",
  "metadata": {
    "title": "Machine Learning in Healthcare",
    "pages": 15,
    "word_count": 8453
  }
}
```

---

### 4. Query RAG System

Semantic search with optional generation.

**Endpoint**: `POST /api/v1/query`

**Content-Type**: `application/json`

**Request Body**:
```json
{
  "query": "What are the main applications of ML in healthcare?",
  "top_k": 10,
  "filters": {
    "document_ids": ["doc_abc123xyz"],
    "min_year": 2020
  },
  "rerank": true,
  "generate_response": true,
  "include_citations": true
}
```

**Response**:
```json
{
  "query": "What are the main applications of ML in healthcare?",
  "results": [
    {
      "chunk_id": "doc_abc123xyz_chunk_12",
      "document_id": "doc_abc123xyz",
      "document_title": "Machine Learning in Healthcare",
      "chunk_text": "Machine learning has three main applications in healthcare: disease diagnosis, treatment recommendation, and patient monitoring...",
      "page_number": 5,
      "similarity_score": 0.92,
      "rerank_score": 0.88,
      "citations": ["Smith et al. (2023)"]
    },
    {
      "chunk_id": "doc_xyz789abc_chunk_8",
      "document_id": "doc_xyz789abc",
      "document_title": "AI in Medical Diagnosis",
      "chunk_text": "Deep learning models have shown remarkable accuracy in medical image analysis...",
      "page_number": 3,
      "similarity_score": 0.87,
      "rerank_score": 0.85,
      "citations": ["Johnson and Lee (2022)"]
    }
  ],
  "generated_response": "Machine learning has several key applications in healthcare:\n\n1. **Disease Diagnosis**: ML models analyze medical images and patient data to detect diseases early (Smith et al., 2023).\n\n2. **Treatment Recommendation**: AI systems suggest personalized treatment plans based on patient characteristics and outcomes data.\n\n3. **Patient Monitoring**: Real-time monitoring systems use ML to predict patient deterioration and alert clinicians (Johnson and Lee, 2022).",
  "num_results": 2,
  "processing_time_ms": 234
}
```

---

### 5. Batch Process Documents

Process multiple PDF documents in a batch.

**Endpoint**: `POST /api/v1/documents/batch`

**Content-Type**: `multipart/form-data`

**Parameters**:
- `files` (required): Multiple PDF files (max 10 files, 50MB each)

**Example Request**:
```bash
curl -X POST http://localhost:8000/api/v1/documents/batch \
  -H "Authorization: Bearer <your-api-key>" \
  -F "files=@paper1.pdf" \
  -F "files=@paper2.pdf" \
  -F "files=@paper3.pdf"
```

**Response**:
```json
{
  "batch_id": "batch_xyz789",
  "status": "processing",
  "total_documents": 3,
  "estimated_completion_minutes": 5,
  "status_url": "/api/v1/batch/batch_xyz789/status"
}
```

---

### 6. Get Batch Status

Check status of batch processing job.

**Endpoint**: `GET /api/v1/batch/{batch_id}/status`

**Response**:
```json
{
  "batch_id": "batch_xyz789",
  "status": "completed",
  "total_documents": 3,
  "completed": 3,
  "failed": 0,
  "documents": [
    {
      "document_id": "doc_aaa111",
      "filename": "paper1.pdf",
      "status": "success",
      "num_chunks": 32
    },
    {
      "document_id": "doc_bbb222",
      "filename": "paper2.pdf",
      "status": "success",
      "num_chunks": 45
    },
    {
      "document_id": "doc_ccc333",
      "filename": "paper3.pdf",
      "status": "success",
      "num_chunks": 28
    }
  ],
  "started_at": "2025-11-06T10:00:00Z",
  "completed_at": "2025-11-06T10:04:30Z"
}
```

---

### 7. List Documents

Get list of all processed documents.

**Endpoint**: `GET /api/v1/documents`

**Query Parameters**:
- `limit` (optional): Number of results (default: 50, max: 100)
- `offset` (optional): Pagination offset (default: 0)
- `sort_by` (optional): Sort field (default: `created_at`)
- `order` (optional): Sort order (`asc` or `desc`, default: `desc`)

**Example Request**:
```bash
curl "http://localhost:8000/api/v1/documents?limit=10&sort_by=created_at&order=desc" \
  -H "Authorization: Bearer <your-api-key>"
```

**Response**:
```json
{
  "documents": [
    {
      "document_id": "doc_abc123xyz",
      "filename": "research_paper.pdf",
      "title": "Machine Learning in Healthcare",
      "num_pages": 15,
      "num_chunks": 47,
      "created_at": "2025-11-06T10:30:00Z",
      "citations_count": 23
    }
  ],
  "total": 150,
  "limit": 10,
  "offset": 0
}
```

---

### 8. Delete Document

Remove document and its embeddings from the system.

**Endpoint**: `DELETE /api/v1/documents/{document_id}`

**Response**:
```json
{
  "document_id": "doc_abc123xyz",
  "status": "deleted",
  "chunks_removed": 47,
  "embeddings_removed": 47
}
```

---

### 9. Get Statistics

Retrieve system statistics.

**Endpoint**: `GET /api/v1/stats`

**Response**:
```json
{
  "total_documents": 150,
  "total_chunks": 6847,
  "total_embeddings": 6847,
  "total_citations": 2341,
  "storage_used_gb": 12.5,
  "avg_processing_time_seconds": 38.2,
  "documents_processed_today": 12,
  "queries_today": 245
}
```

---

### 10. Reindex Document

Regenerate embeddings for a document.

**Endpoint**: `POST /api/v1/documents/{document_id}/reindex`

**Response**:
```json
{
  "document_id": "doc_abc123xyz",
  "status": "reindexing",
  "estimated_completion_minutes": 2
}
```

---

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Invalid or missing API key |
| 404 | Not Found - Document or resource not found |
| 413 | Payload Too Large - File exceeds size limit |
| 422 | Unprocessable Entity - Invalid file format |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable - GPU or service overloaded |

## Rate Limits

- **Free tier**: 10 requests/minute, 100 requests/day
- **Pro tier**: 100 requests/minute, 10,000 requests/day
- **Enterprise**: Custom limits

## Webhooks

Configure webhooks to receive notifications for async operations.

**Endpoint**: `POST /api/v1/webhooks`

**Request Body**:
```json
{
  "url": "https://your-domain.com/webhook",
  "events": ["document.processed", "batch.completed"],
  "secret": "your_webhook_secret"
}
```

**Webhook Payload Example**:
```json
{
  "event": "document.processed",
  "timestamp": "2025-11-06T10:30:00Z",
  "data": {
    "document_id": "doc_abc123xyz",
    "status": "success",
    "num_chunks": 47
  },
  "signature": "sha256=..."
}
```

## SDK Examples

### Python

```python
import requests

class DolphinRAGClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def process_pdf(self, pdf_path):
        with open(pdf_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{self.base_url}/api/v1/documents/process",
                headers=self.headers,
                files=files
            )
        return response.json()

    def query(self, query_text, top_k=10):
        data = {"query": query_text, "top_k": top_k, "rerank": True}
        response = requests.post(
            f"{self.base_url}/api/v1/query",
            headers=self.headers,
            json=data
        )
        return response.json()

# Usage
client = DolphinRAGClient("http://localhost:8000", "your_api_key")

# Process PDF
result = client.process_pdf("paper.pdf")
print(f"Processed document: {result['document_id']}")

# Query
results = client.query("What is machine learning?")
for r in results['results']:
    print(f"- {r['chunk_text'][:100]}...")
```

### JavaScript/TypeScript

```typescript
class DolphinRAGClient {
  constructor(
    private baseUrl: string,
    private apiKey: string
  ) {}

  async processPDF(file: File): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/api/v1/documents/process`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`
      },
      body: formData
    });

    return response.json();
  }

  async query(queryText: string, topK: number = 10): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/v1/query`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        query: queryText,
        top_k: topK,
        rerank: true
      })
    });

    return response.json();
  }
}

// Usage
const client = new DolphinRAGClient('http://localhost:8000', 'your_api_key');

// Process PDF
const fileInput = document.querySelector('input[type="file"]');
const result = await client.processPDF(fileInput.files[0]);
console.log('Processed document:', result.document_id);

// Query
const results = await client.query('What is machine learning?');
results.results.forEach(r => {
  console.log(`- ${r.chunk_text.substring(0, 100)}...`);
});
```

### cURL Examples

**Process a PDF**:
```bash
curl -X POST http://localhost:8000/api/v1/documents/process \
  -H "Authorization: Bearer your_api_key" \
  -F "file=@paper.pdf"
```

**Query the system**:
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "top_k": 10,
    "rerank": true
  }'
```

## Streaming Responses

For long-running operations, use Server-Sent Events (SSE):

**Endpoint**: `GET /api/v1/documents/{document_id}/process/stream`

```javascript
const eventSource = new EventSource(
  'http://localhost:8000/api/v1/documents/doc_123/process/stream'
);

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`Progress: ${data.progress}% - ${data.status}`);
};
```

---

**Last Updated**: 2025-11-06
