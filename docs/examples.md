# Usage Examples

## Table of Contents
1. [Quick Start](#quick-start)
2. [Processing PDFs](#processing-pdfs)
3. [Querying the System](#querying-the-system)
4. [Advanced Retrieval](#advanced-retrieval)
5. [Batch Processing](#batch-processing)
6. [Python SDK Examples](#python-sdk-examples)
7. [Integration Examples](#integration-examples)

## Quick Start

### 1. Process Your First PDF

```python
import requests

API_URL = "http://localhost:8000"
API_KEY = "your_api_key"

# Upload and process a PDF
with open("research_paper.pdf", "rb") as f:
    response = requests.post(
        f"{API_URL}/api/v1/documents/process",
        headers={"Authorization": f"Bearer {API_KEY}"},
        files={"file": f}
    )

result = response.json()
print(f"✅ Processed: {result['document_id']}")
print(f"📄 Pages: {result['num_pages']}")
print(f"🧩 Chunks: {result['num_chunks']}")
print(f"📚 Citations: {len(result['citations'])}")
```

### 2. Query the System

```python
# Search for information
query_response = requests.post(
    f"{API_URL}/api/v1/query",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "query": "What are the main findings?",
        "top_k": 10,
        "rerank": True,
        "generate_response": True
    }
)

results = query_response.json()
print(f"\n🔍 Query: {results['query']}")
print(f"\n📝 Generated Response:\n{results['generated_response']}")
print(f"\n📚 Sources:")
for i, result in enumerate(results['results'][:3], 1):
    print(f"{i}. {result['document_title']} (p. {result['page_number']})")
    print(f"   Score: {result['rerank_score']:.2f}")
```

## Processing PDFs

### Basic PDF Processing

```python
import os
from pathlib import Path

def process_pdf(pdf_path: str, api_url: str, api_key: str):
    """Process a single PDF document"""

    with open(pdf_path, 'rb') as f:
        response = requests.post(
            f"{api_url}/api/v1/documents/process",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": f},
            data={
                "extract_citations": "true",
                "generate_embeddings": "true",
                "chunk_size": 1024,
                "chunk_overlap": 100
            }
        )

    if response.status_code == 200:
        result = response.json()
        print(f"✅ Successfully processed: {result['filename']}")
        return result
    else:
        print(f"❌ Error: {response.json()}")
        return None

# Usage
result = process_pdf("paper.pdf", API_URL, API_KEY)
```

### Processing with Custom Settings

```python
def process_pdf_custom(
    pdf_path: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 100,
    extract_citations: bool = True
):
    """Process PDF with custom settings"""

    with open(pdf_path, 'rb') as f:
        response = requests.post(
            f"{API_URL}/api/v1/documents/process",
            headers={"Authorization": f"Bearer {API_KEY}"},
            files={"file": f},
            data={
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "extract_citations": str(extract_citations).lower()
            }
        )

    return response.json()

# Large documents - smaller chunks for better retrieval
result = process_pdf_custom(
    "large_paper.pdf",
    chunk_size=512,
    chunk_overlap=50
)

# Technical documents - larger chunks for context
result = process_pdf_custom(
    "technical_manual.pdf",
    chunk_size=2048,
    chunk_overlap=200
)
```

### Retrieving Processed Markdown

```python
def get_markdown(document_id: str):
    """Retrieve markdown for a processed document"""

    response = requests.get(
        f"{API_URL}/api/v1/documents/{document_id}/markdown",
        headers={"Authorization": f"Bearer {API_KEY}"}
    )

    if response.status_code == 200:
        data = response.json()
        return data['markdown']
    return None

# Get and save markdown
markdown = get_markdown("doc_abc123xyz")
with open("output.md", "w", encoding="utf-8") as f:
    f.write(markdown)
```

## Querying the System

### Simple Query

```python
def simple_query(query_text: str):
    """Perform a simple semantic search"""

    response = requests.post(
        f"{API_URL}/api/v1/query",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"query": query_text, "top_k": 10}
    )

    return response.json()

results = simple_query("What is machine learning?")
for result in results['results']:
    print(f"📄 {result['document_title']}")
    print(f"📍 Page {result['page_number']}")
    print(f"📊 Score: {result['similarity_score']:.3f}")
    print(f"📝 {result['chunk_text'][:200]}...\n")
```

### Query with Reranking

```python
def query_with_reranking(query_text: str, top_k: int = 50, top_n: int = 10):
    """Query with reranking for better results"""

    response = requests.post(
        f"{API_URL}/api/v1/query",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "query": query_text,
            "top_k": top_k,  # Initial retrieval
            "rerank": True,
            "generate_response": False  # Just retrieval
        }
    )

    return response.json()

results = query_with_reranking("applications of deep learning in healthcare")

print(f"Found {len(results['results'])} relevant chunks:")
for i, result in enumerate(results['results'][:5], 1):
    print(f"\n{i}. {result['document_title']} (Page {result['page_number']})")
    print(f"   Similarity: {result['similarity_score']:.3f}")
    print(f"   Rerank: {result['rerank_score']:.3f}")
    print(f"   Text: {result['chunk_text'][:150]}...")
```

### Query with Response Generation

```python
def query_with_generation(query_text: str):
    """Query and generate a comprehensive response"""

    response = requests.post(
        f"{API_URL}/api/v1/query",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "query": query_text,
            "top_k": 50,
            "rerank": True,
            "generate_response": True,
            "include_citations": True
        }
    )

    return response.json()

# Ask a complex question
results = query_with_generation(
    "What are the ethical considerations in using AI for medical diagnosis?"
)

print(f"🔍 Query: {results['query']}\n")
print(f"📝 Response:\n{results['generated_response']}\n")
print(f"📚 Based on {len(results['results'])} sources:")
for i, source in enumerate(results['results'][:3], 1):
    print(f"{i}. {source['document_title']} (p. {source['page_number']})")
```

## Advanced Retrieval

### Filtered Search

```python
def filtered_query(query_text: str, filters: dict):
    """Search with metadata filters"""

    response = requests.post(
        f"{API_URL}/api/v1/query",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "query": query_text,
            "top_k": 20,
            "filters": filters
        }
    )

    return response.json()

# Search only in specific documents
results = filtered_query(
    "neural networks",
    filters={"document_ids": ["doc_123", "doc_456"]}
)

# Search papers from recent years
results = filtered_query(
    "climate change impacts",
    filters={"min_year": 2020}
)

# Search specific authors
results = filtered_query(
    "transformer architecture",
    filters={"authors": ["Vaswani", "Devlin"]}
)
```

### Multi-Query Search

```python
def multi_query_search(queries: list[str]):
    """Search multiple queries and combine results"""

    all_results = []

    for query in queries:
        response = requests.post(
            f"{API_URL}/api/v1/query",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"query": query, "top_k": 10}
        )
        all_results.extend(response.json()['results'])

    # Remove duplicates based on chunk_id
    unique_results = {r['chunk_id']: r for r in all_results}.values()

    # Sort by score
    sorted_results = sorted(
        unique_results,
        key=lambda x: x['similarity_score'],
        reverse=True
    )

    return sorted_results

# Search related concepts
results = multi_query_search([
    "deep learning architectures",
    "convolutional neural networks",
    "transformer models"
])

print(f"Found {len(results)} unique chunks across all queries")
```

## Batch Processing

### Process Multiple PDFs

```python
def batch_process_pdfs(pdf_files: list[str]):
    """Process multiple PDFs in a batch"""

    files = [
        ('files', (os.path.basename(pdf), open(pdf, 'rb'), 'application/pdf'))
        for pdf in pdf_files
    ]

    response = requests.post(
        f"{API_URL}/api/v1/documents/batch",
        headers={"Authorization": f"Bearer {API_KEY}"},
        files=files
    )

    # Close file handles
    for _, (_, file_obj, _) in files:
        file_obj.close()

    return response.json()

# Process a folder of PDFs
pdf_folder = Path("papers")
pdf_files = list(pdf_folder.glob("*.pdf"))

batch_result = batch_process_pdfs([str(f) for f in pdf_files])
print(f"Batch ID: {batch_result['batch_id']}")
print(f"Processing {batch_result['total_documents']} documents")
print(f"ETA: {batch_result['estimated_completion_minutes']} minutes")
```

### Monitor Batch Progress

```python
import time

def wait_for_batch(batch_id: str, poll_interval: int = 10):
    """Wait for batch processing to complete"""

    while True:
        response = requests.get(
            f"{API_URL}/api/v1/batch/{batch_id}/status",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )

        status = response.json()
        print(f"Progress: {status['completed']}/{status['total_documents']}")

        if status['status'] == 'completed':
            print("✅ Batch processing completed!")
            return status
        elif status['status'] == 'failed':
            print("❌ Batch processing failed!")
            return status

        time.sleep(poll_interval)

# Process and wait
batch_result = batch_process_pdfs(pdf_files)
final_status = wait_for_batch(batch_result['batch_id'])

# Print results
print(f"\nResults:")
for doc in final_status['documents']:
    status_icon = "✅" if doc['status'] == 'success' else "❌"
    print(f"{status_icon} {doc['filename']}: {doc['num_chunks']} chunks")
```

## Python SDK Examples

### Complete SDK Implementation

```python
from typing import Optional, List, Dict
import requests
from pathlib import Path

class DolphinRAG:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def process_pdf(
        self,
        pdf_path: str,
        chunk_size: int = 1024,
        chunk_overlap: int = 100
    ) -> Dict:
        """Process a PDF document"""
        with open(pdf_path, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/api/v1/documents/process",
                headers=self.headers,
                files={"file": f},
                data={
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap
                }
            )
        response.raise_for_status()
        return response.json()

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        rerank: bool = True,
        generate: bool = False,
        filters: Optional[Dict] = None
    ) -> Dict:
        """Query the RAG system"""
        payload = {
            "query": query_text,
            "top_k": top_k,
            "rerank": rerank,
            "generate_response": generate
        }
        if filters:
            payload["filters"] = filters

        response = requests.post(
            f"{self.base_url}/api/v1/query",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def get_markdown(self, document_id: str) -> str:
        """Get processed markdown for a document"""
        response = requests.get(
            f"{self.base_url}/api/v1/documents/{document_id}/markdown",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()['markdown']

    def list_documents(
        self,
        limit: int = 50,
        offset: int = 0
    ) -> Dict:
        """List all processed documents"""
        response = requests.get(
            f"{self.base_url}/api/v1/documents",
            headers=self.headers,
            params={"limit": limit, "offset": offset}
        )
        response.raise_for_status()
        return response.json()

    def delete_document(self, document_id: str) -> Dict:
        """Delete a document and its embeddings"""
        response = requests.delete(
            f"{self.base_url}/api/v1/documents/{document_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

# Usage
rag = DolphinRAG("http://localhost:8000", "your_api_key")

# Process document
doc = rag.process_pdf("paper.pdf")
print(f"Processed: {doc['document_id']}")

# Query
results = rag.query("What is deep learning?", generate=True)
print(results['generated_response'])

# Get markdown
markdown = rag.get_markdown(doc['document_id'])
with open("output.md", "w") as f:
    f.write(markdown)
```

## Integration Examples

### Streamlit App

```python
import streamlit as st
from dolphin_rag import DolphinRAG

st.title("📚 Document Q&A System")

# Initialize client
rag = DolphinRAG(
    st.secrets["api_url"],
    st.secrets["api_key"]
)

# File upload
uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
if uploaded_file:
    with st.spinner("Processing document..."):
        # Save temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Process
        result = rag.process_pdf("temp.pdf")
        st.success(f"Processed: {result['num_chunks']} chunks")
        st.session_state['doc_id'] = result['document_id']

# Query interface
if 'doc_id' in st.session_state:
    query = st.text_input("Ask a question:")
    if query:
        with st.spinner("Searching..."):
            results = rag.query(query, generate=True)

        st.subheader("Answer")
        st.write(results['generated_response'])

        st.subheader("Sources")
        for i, result in enumerate(results['results'][:3], 1):
            with st.expander(f"Source {i}: {result['document_title']}"):
                st.write(f"Page: {result['page_number']}")
                st.write(f"Score: {result['rerank_score']:.3f}")
                st.write(result['chunk_text'])
```

### Flask API Wrapper

```python
from flask import Flask, request, jsonify
from dolphin_rag import DolphinRAG

app = Flask(__name__)
rag = DolphinRAG("http://localhost:8000", "your_api_key")

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    file.save('temp.pdf')
    result = rag.process_pdf('temp.pdf')
    return jsonify(result)

@app.route('/search', methods=['POST'])
def search():
    query = request.json['query']
    results = rag.query(query, generate=True)
    return jsonify(results)

if __name__ == '__main__':
    app.run(port=5000)
```

---

**Last Updated**: 2025-11-06
