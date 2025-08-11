# hpact-edtech

## Overview

This project demonstrates a modular pipeline for extracting, tagging, embedding, and storing knowledge graph triples from web content using LangChain, OpenAI, Qdrant, and AWS Neptune.

## Folder Structure

```
src/
  main.py             # Entry point
  pipeline.py         # Pipeline orchestration
  web_crawler.py      # Content extraction
tests/
  test_pipeline.py    # Pipeline unit tests
  test_web_crawler.py # Web crawler tests
requirements.txt      # Python dependencies
Dockerfile            # Container build instructions
README.md             # Documentation
```

## Setup

Set required environment variables:

```bash
export OPENAI_API_KEY=sk-xxx
export QDRANT_URL=http://localhost:6333
export QDRANT_COLLECTION=edtech_docs
export NEPTUNE_ENDPOINT=your-neptune-endpoint
export NEPTUNE_PORT=8182
```

## Build and Run

### Docker

Build and run using Docker:

```bash
docker build -t hpact-edtech .
docker run --env OPENAI_API_KEY --env QDRANT_URL --env QDRANT_COLLECTION --env NEPTUNE_ENDPOINT --env NEPTUNE_PORT hpact-edtech
# or pass URL as argument
docker run hpact-edtech https://example.com
```

### Local

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline:

```bash
python src/main.py https://example.com
```

## Testing

Run all tests:

```bash
pytest
```

## Tech Stack

- **LangChain**: Orchestrates AI and DB operations.
- **OpenAI GPT-3.5 Turbo**: Tagging and triple extraction.
- **OpenAI Embeddings**: Vector generation.
- **Qdrant**: Vector database.
- **AWS Neptune**: Knowledge graph database.
- **trafilatura / BeautifulSoup**: Web content extraction.

## Requirements

See `requirements.txt` for details.
