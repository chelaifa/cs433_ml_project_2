"""RAG Pipeline for Academic Papers."""

__version__ = "1.0.1"
__author__ = "Elie Bruno"
__description__ = "RAG pipeline for academic papers with OpenAlex and Dolphin"

# Make key classes easily importable
from .openalex.fetcher import MetadataFetcher
from .openalex.downloader import PDFDownloader

# RAG components
from .rag.chunking import DocumentChunker
from .rag.openai_embedder import OpenAIEmbedder

__all__ = [
    "MetadataFetcher",
    "PDFDownloader",
    "DocumentChunker",
    "OpenAIEmbedder",
]

# PDF parsing exports (optional - only available with [pdf] extra)
# Import these directly when needed:
#   from rag_pipeline.pdf_parsing import DolphinModel, PDFParsingPipeline, PDFParsingConfig
