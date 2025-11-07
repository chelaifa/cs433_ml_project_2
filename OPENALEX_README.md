# OpenAlex Metadata Fetcher & PDF Downloader

Professional Python toolkit for fetching OpenAlex metadata and downloading PDFs with clean architecture, Pydantic models, and comprehensive logging.

## 🌟 Features

- **Two-stage workflow**: Fetch metadata once, filter and download multiple times
- **Parquet storage**: Efficient columnar storage for ~8K works (~10-20MB)
- **Pydantic models**: Type-safe data validation and serialization
- **Loguru logging**: Beautiful, structured logging with rotation
- **Configuration management**: Environment variables + pydantic-settings
- **Clean architecture**: Separation of concerns with services, models, and utilities
- **Advanced filtering**: Filter by source, citations, year, OA status, DOI, and more
- **Resume capability**: Skip already downloaded PDFs
- **Progress tracking**: Real-time progress with ETA calculations
- **Error handling**: Robust error handling with detailed logging
- **Metadata exploration**: Built-in tools to analyze your dataset

## 📦 Installation

```bash
# Dependencies already in requirements.txt
pip install -r requirements.txt

# Optional: Create .env file for configuration
cp .env.example .env
# Edit .env with your settings
```

## 🚀 Quick Start

### 1. Fetch All Metadata (Run Once)

```bash
# Fetch all works and save to Parquet
python fetch_metadata.py

# With your email for faster API access (recommended)
python fetch_metadata.py --email your@email.com

# Custom output directory
python fetch_metadata.py --output-dir my_data
```

**Output:**
- `openalex_data/openalex_works.parquet` - All metadata (~10-20MB for 8K works)
- `openalex_data/summary_stats.json` - Summary statistics
- `openalex_data/summary_stats.txt` - Human-readable summary

### 2. Explore the Metadata

```bash
# Analyze the dataset
python explore_metadata.py

# Show top 20 items in rankings
python explore_metadata.py --top 20
```

**Output:**
- Basic statistics (total works, memory usage)
- Open access breakdown by status
- PDF availability statistics
- Top sources and source types
- Publication year distribution
- Citation statistics
- Most cited papers
- Topics, languages, and more

### 3. Download PDFs (Run with Filters)

```bash
# Download all PDFs with available URLs
python download_pdfs.py

# Only arXiv papers
python download_pdfs.py --only-arxiv

# Only highly cited papers (10+ citations)
python download_pdfs.py --min-citations 10

# Recent papers (2020 onwards)
python download_pdfs.py --year-from 2020

# Gold OA only
python download_pdfs.py --oa-status gold

# Combine multiple filters
python download_pdfs.py --only-arxiv --min-citations 5 --year-from 2018

# Limit number of downloads
python download_pdfs.py --only-arxiv --limit 100

# Only papers with DOI
python download_pdfs.py --has-doi

# Only papers without DOI
python download_pdfs.py --no-doi
```

**Output:**
- `openalex_data/pdfs/` - Downloaded PDF files
- `openalex_data/metadata/` - Individual JSON metadata (if enabled)
- `openalex_data/download_stats.json` - Download statistics

## 📂 Project Structure

```
project-2-rag/
├── openalex_fetcher/          # Main package
│   ├── __init__.py
│   ├── models.py              # Pydantic models for data structures
│   ├── config.py              # Configuration with pydantic-settings
│   ├── fetcher.py             # Metadata fetcher service
│   ├── downloader.py          # PDF downloader service
│   └── utils.py               # Utility functions
├── fetch_metadata.py          # CLI: Fetch metadata from API
├── download_pdfs.py           # CLI: Download PDFs from parquet
├── explore_metadata.py        # CLI: Analyze metadata
├── requirements.txt           # Python dependencies
├── .env.example               # Example environment configuration
└── OPENALEX_README.md         # This file
```

## 🎯 Usage Examples

### Example 1: arXiv Papers Only

```bash
# Fetch metadata (run once)
python fetch_metadata.py --email your@email.com

# Explore what we have
python explore_metadata.py

# Download only arXiv papers
python download_pdfs.py --only-arxiv
```

### Example 2: Highly Cited Recent Papers

```bash
# Fetch metadata (run once)
python fetch_metadata.py

# Download highly cited papers from last 5 years
python download_pdfs.py --min-citations 10 --year-from 2019
```

### Example 3: Custom Filters

```bash
# Fetch metadata with custom topic
python fetch_metadata.py --filters "primary_topic.id:t10856,open_access.is_oa:true"

# Download PubMed Central papers only
python download_pdfs.py --only-pmc
```

### Example 4: Working with Parquet in Python

```python
import pandas as pd

# Load the metadata
df = pd.read_parquet("openalex_data/openalex_works.parquet")

print(f"Total works: {len(df):,}")

# Filter for arXiv papers with 10+ citations
arxiv_cited = df[
    (df["best_oa_source"].str.contains("arXiv", case=False, na=False)) &
    (df["cited_by_count"] >= 10)
]
print(f"arXiv papers with 10+ citations: {len(arxiv_cited):,}")

# Get papers from 2020-2023
recent = df[df["publication_year"].between(2020, 2023)]
print(f"Papers from 2020-2023: {len(recent):,}")

# Papers with PDF URLs
has_pdf = df[df["has_any_pdf"] == True]
print(f"Papers with PDF URLs: {len(has_pdf):,}")

# Top authors
authors = df["first_author"].value_counts().head(10)
print("\nTop 10 first authors:")
print(authors)

# Save filtered subset
arxiv_cited.to_parquet("openalex_data/arxiv_highly_cited.parquet")
```

## ⚙️ Configuration

### Method 1: Environment Variables (Recommended)

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
OPENALEX_EMAIL=your.email@example.com
OPENALEX_OUTPUT_DIR=openalex_data
OPENALEX_REQUEST_DELAY=0.1
OPENALEX_DOWNLOAD_DELAY=0.5
OPENALEX_LOG_LEVEL=INFO
```

### Method 2: Command Line Arguments

```bash
python fetch_metadata.py \
    --email your@email.com \
    --output-dir my_data \
    --log-level DEBUG
```

### Method 3: Python Code

```python
from openalex_fetcher.config import OpenAlexConfig
from openalex_fetcher.fetcher import MetadataFetcher

config = OpenAlexConfig(
    email="your@email.com",
    output_dir="my_data",
    log_level="DEBUG"
)

fetcher = MetadataFetcher(config)
df = fetcher.run()
```

## 🔧 Advanced Usage

### Custom Filtering in Python

```python
import pandas as pd
from openalex_fetcher.config import OpenAlexConfig
from openalex_fetcher.downloader import PDFDownloader

# Load metadata
df = pd.read_parquet("openalex_data/openalex_works.parquet")

# Apply custom filters
filtered = df[
    (df["cited_by_count"] >= 50) &
    (df["publication_year"] >= 2018) &
    (df["has_any_pdf"] == True)
]

# Save filtered data
filtered.to_parquet("openalex_data/filtered.parquet")

# Download PDFs from filtered data
config = OpenAlexConfig()
downloader = PDFDownloader(config)
stats = downloader.run(parquet_file="openalex_data/filtered.parquet")

print(f"Downloaded: {stats.pdfs_downloaded}")
```

### Accessing Full JSON

```python
import pandas as pd
import json

df = pd.read_parquet("openalex_data/openalex_works.parquet")

# Get full JSON for first work
work_json = json.loads(df.iloc[0]["full_json"])

# Access all fields
print(work_json["abstract_inverted_index"])
print(work_json["authorships"])
print(work_json["referenced_works"])
```

## 📊 Data Schema

### Main Fields in Parquet

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | OpenAlex ID URL |
| `openalex_id` | str | Short ID (e.g., W123456789) |
| `doi` | str | DOI |
| `title` | str | Paper title |
| `publication_year` | int | Publication year |
| `type` | str | Work type (article, book, etc.) |
| `is_oa` | bool | Is open access |
| `oa_status` | str | OA status (gold, green, etc.) |
| `best_oa_pdf_url` | str | Best PDF URL |
| `has_any_pdf` | bool | Has any PDF URL |
| `cited_by_count` | int | Citation count |
| `topic_name` | str | Primary topic |
| `first_author` | str | First author name |
| `author_names` | str | All authors (pipe-separated) |
| `keywords` | str | Keywords (pipe-separated) |
| `full_json` | str | Complete OpenAlex JSON |

[See `openalex_fetcher/models.py` for complete schema]

## 🎛️ Download Filtering Options

```
--only-arxiv              Only arXiv papers
--only-pmc                Only PubMed Central papers
--source NAME             Filter by specific source name
--min-citations N         Minimum citation count
--max-citations N         Maximum citation count
--year-from YYYY          Minimum publication year
--year-to YYYY            Maximum publication year
--oa-status STATUS        Filter by OA status (gold/green/hybrid/bronze/diamond)
--has-doi                 Only papers with DOI
--no-doi                  Only papers without DOI
--limit N                 Limit number of PDFs to download
--no-skip-existing        Re-download existing PDFs
--download-delay SEC      Delay between downloads
```

## 📈 Performance

### Metadata Fetching
- **Speed**: ~50-100 works/second with polite pool
- **Time**: ~2-5 minutes for 8,000 works
- **Storage**: ~10-20 MB parquet file

### PDF Downloads
- **Success rate**: ~40-60% (depends on filters)
- **Speed**: ~2-5 PDFs/second (with rate limiting)
- **Time**: ~30-60 minutes for 1,000 PDFs

## 🐛 Troubleshooting

### "No works fetched"
- Check your filters with `--filters`
- Verify internet connection
- Check OpenAlex API status

### "403 Forbidden" errors
- These are normal for some publishers
- Try filtering by source: `--only-arxiv` or `--only-pmc`

### Import errors
- Ensure you're running from project root
- Check all dependencies installed: `pip install -r requirements.txt`

### Out of memory
- Disable full JSON: `python fetch_metadata.py --no-full-json`
- Process in batches using `--limit`

## 📝 Logging

Logs are written to console by default. To save to file:

```bash
python fetch_metadata.py --log-file openalex.log --log-level DEBUG
```

Or set in `.env`:

```env
OPENALEX_LOG_FILE=openalex.log
OPENALEX_LOG_LEVEL=DEBUG
```

## 🤝 Contributing

This is a professional, production-ready codebase with:
- ✅ Type hints everywhere
- ✅ Pydantic models for validation
- ✅ Clean architecture (models, services, utils)
- ✅ Comprehensive error handling
- ✅ Structured logging with Loguru
- ✅ Configuration management
- ✅ Docstrings for all public APIs

## 📚 Resources

- [OpenAlex API Documentation](https://docs.openalex.org/)
- [OpenAlex Works Endpoint](https://docs.openalex.org/api-entities/works)
- [OpenAlex Filters](https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/filter-entity-lists)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Loguru Documentation](https://loguru.readthedocs.io/)

## 📄 License

This project is for research and educational purposes. Respect OpenAlex's terms of service and the licenses of downloaded papers.

## 🎉 Credits

Built with:
- [OpenAlex](https://openalex.org/) - Free, open catalog of scholarly papers
- [Pydantic](https://pydantic.dev/) - Data validation using Python type hints
- [Loguru](https://loguru.readthedocs.io/) - Python logging made simple
- [Pandas](https://pandas.pydata.org/) - Data analysis library
- [PyArrow](https://arrow.apache.org/docs/python/) - Parquet file support

---

**Pro Tip**: Always fetch metadata first with `fetch_metadata.py`, then explore with `explore_metadata.py`, and finally download selectively with `download_pdfs.py` using filters! 🚀
