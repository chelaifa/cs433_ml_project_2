        # PDF Parsing Pipeline Design

**Date**: 2025-11-07
**Status**: Approved

## Overview

A new `rag_pipeline/pdf_parsing/` sub-package that converts PDF documents to structured Markdown using the ByteDance Dolphin multimodal model. The design follows a pipeline-based architecture with modular, independently testable components.

## Architecture

### Pipeline-Based Architecture

The system uses a configurable pipeline of processors:

```
PDF → ImageExtractor → LayoutParser → ElementRecognizer → MarkdownConverter
```

Each processor:
- Implements an abstract interface
- Is independently testable and configurable
- Uses Pydantic for input/output validation
- Can be swapped or extended

### Package Structure

```
rag_pipeline/pdf_parsing/
├── __init__.py              # Public API exports
├── config.py                # Pydantic configuration models
├── models.py                # Pydantic data models
├── models/                  # Dolphin model weights (~796MB)
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── core/
│   ├── __init__.py
│   ├── interfaces.py        # Abstract base classes/protocols
│   ├── pipeline.py          # Main pipeline orchestrator
│   └── exceptions.py        # Custom exceptions
├── processors/
│   ├── __init__.py
│   ├── base.py              # Base processor class
│   ├── image_extractor.py   # PDF → Images
│   ├── layout_parser.py     # Images → Layout detection
│   ├── element_recognizer.py # Layout → Element content
│   └── markdown_converter.py # Elements → Markdown
├── model/
│   ├── __init__.py
│   ├── dolphin.py           # DOLPHIN model wrapper
│   └── loader.py            # Model loading utilities
├── utils/
│   ├── __init__.py
│   ├── image_utils.py       # Image processing helpers
│   ├── coordinate_utils.py  # Coordinate transformations
│   └── visualization.py     # Layout visualization
└── factories/
    ├── __init__.py
    └── processor_factory.py # Creates processors based on config
```

## Data Flow

1. **PDF Input**: Accept PDF file path
2. **Image Extraction**: Convert PDF pages to PIL Images (using pymupdf)
3. **Layout Detection**: Detect layout elements (text, tables, equations, figures, code)
4. **Element Recognition**: Extract content from each element using Dolphin model
5. **Markdown Generation**: Convert parsed elements to structured Markdown with metadata

## Key Design Decisions

### 1. Pydantic Models

All configuration and data structures use Pydantic for:
- Runtime validation
- Type safety
- Clear documentation
- Easy serialization

**Configuration Models**:
- `DolphinModelConfig`: Model loading and inference settings
- `ProcessingConfig`: Document processing parameters
- `OutputConfig`: Output directory and format settings
- `PDFParsingConfig`: Main configuration

**Data Models**:
- `BoundingBox`: Element coordinates
- `ImageDimensions`: Image size information
- `LayoutElement`: Detected layout element
- `ParsedElement`: Fully parsed element with content
- `PageResult`: Results for single page
- `DocumentResult`: Complete document results

### 2. Abstract Interfaces

Three main interfaces define the architecture:

**`Processor[InputT, OutputT]`**: Generic processor interface
- `process(input_data)`: Process single input
- `process_batch(inputs)`: Process batch for efficiency
- `validate_input(input_data)`: Optional validation

**`DocumentParser`**: Document-level parsing interface
- `parse_document(document_path)`: Parse complete document
- `parse_page(image, page_num)`: Parse single page

**`ModelWrapper`**: ML model wrapper interface
- `load()`, `unload()`, `is_loaded()`: Lifecycle management
- `infer()`, `infer_batch()`: Inference methods

### 3. Factory Pattern

`ProcessorFactory` creates processors based on configuration, enabling:
- Easy testing with mock processors
- Runtime configuration of pipeline stages
- Extensibility for new processor types

## Component Details

### ImageExtractor
- **Input**: Path to PDF file
- **Output**: List of PIL Images
- **Logic**: Uses pymupdf to convert PDF pages to high-resolution images (896px longest dimension)

### LayoutParser
- **Input**: PIL Image
- **Output**: List of LayoutElements with bounding boxes and labels
- **Logic**: Uses Dolphin model with "Parse the reading order of this document" prompt

### ElementRecognizer
- **Input**: Tuple of (Image, List[LayoutElement])
- **Output**: List of ParsedElements with extracted content
- **Logic**:
  - Crops each element from image
  - Groups by type (tables, equations, code, text, figures)
  - Batch processes each group with type-specific prompts
  - Saves figures to disk

### MarkdownConverter
- **Input**: DocumentResult
- **Output**: None (writes files)
- **Logic**:
  - Converts parsed elements to Markdown
  - Saves JSON results
  - Saves Markdown files
  - Organizes output directories

### DolphinModel
- Wraps the HuggingFace VisionEncoderDecoderModel
- Handles device management (CUDA/CPU)
- Implements batch inference
- Supports both single and batch processing

## Features Implemented

From Dolphin demo_page.py:
- ✅ Two-stage parsing (layout → elements)
- ✅ Multi-page PDF support
- ✅ Element type detection (text, tables, equations, code, figures)
- ✅ Batch processing for efficiency
- ✅ Figure extraction and saving
- ✅ Markdown generation
- ✅ JSON output with metadata
- ✅ Coordinate transformations (padded ↔ original)
- ✅ Layout visualization (optional)

Enhanced features:
- ✅ Pydantic models for type safety
- ✅ Abstract interfaces for extensibility
- ✅ Factory pattern for flexibility
- ✅ Better error handling
- ✅ Configuration management
- ✅ Modular, testable architecture

## Usage Example

```python
from pathlib import Path
from rag_pipeline.pdf_parsing import PDFParsingPipeline, PDFParsingConfig, OutputConfig

# Configure pipeline
config = PDFParsingConfig(
    output=OutputConfig(output_dir=Path("./output"))
)

# Create pipeline
pipeline = PDFParsingPipeline(config)

# Parse document
result = pipeline.parse_document(Path("document.pdf"))

# Access results
print(f"Parsed {result.total_pages} pages")
for page in result.pages:
    print(f"Page {page.page_number}: {len(page.elements)} elements")
```

## Testing Strategy

1. **Unit Tests**: Each processor independently
2. **Integration Tests**: Full pipeline with sample PDFs
3. **Model Tests**: Mock DolphinModel for fast testing
4. **Validation Tests**: Pydantic model validation

## Future Enhancements

- Async processing for large document batches
- OCR integration for scanned documents
- Table extraction improvements
- Citation extraction
- Progress callbacks
- Streaming processing for large PDFs

## Model Weights

- **Location**: `rag_pipeline/pdf_parsing/models/`
- **Size**: ~796MB
- **Source**: Copied from `/Users/eliebruno/Desktop/code/Dolphin/hf_model/`
- **Model**: bytedance/dolphin (VisionEncoderDecoderModel)

## Dependencies

- torch
- transformers
- pymupdf
- Pillow
- opencv-python
- numpy
- pydantic