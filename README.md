# LangChain Web Content Processing Pipeline

## Overview

A modular Python application that implements a sophisticated web content extraction, tagging, embedding, and knowledge graph storage pipeline using LangChain integrations. The system crawls websites, extracts meaningful content, generates structured tags using OpenAI's language models, creates embeddings, and stores everything in a vector database for semantic search.

## Architecture

The application follows a modular architecture with clear separation of concerns:

- **Web Crawler**: Sophisticated content extraction using trafilatura and BeautifulSoup
- **LangChain Pipeline**: Orchestrates tagging, embedding, and knowledge graph storage
- **Knowledge Graph**: Vector database storage for semantic search and retrieval
- **CLI Interface**: Command-line tool for processing URLs and searching content

## Folder Structure

```
├── src/                    # Application source code
│   ├── web_crawler.py     # Sophisticated web crawler with LangChain Document generation
│   ├── pipeline.py        # LangChain pipeline for processing and storage
│   └── main.py           # Entry point for URL processing and CLI interface
├── tests/                 # Unit tests
│   ├── test_web_crawler.py    # Tests for web crawler with mocked HTTP requests
│   └── test_pipeline.py       # Tests for pipeline with mocked LangChain components
├── requirements.txt       # Python dependencies
├── README.md             # This documentation
└── .env.example          # Example environment configuration
```

## Key Components

### Web Crawler (`src/web_crawler.py`)
- Uses **trafilatura** for intelligent content extraction
- Falls back to **BeautifulSoup** for robust HTML parsing
- Generates **LangChain Document** objects with rich metadata
- Extracts titles, descriptions, keywords, and Open Graph data
- Handles various content types and edge cases

### LangChain Pipeline (`src/pipeline.py`)
- **Content Tagging**: Uses OpenAI GPT models for structured content analysis
- **Embeddings**: Generates semantic embeddings using OpenAI's embedding models
- **Knowledge Graph Storage**: Stores processed content in Qdrant vector database
- **Semantic Search**: Enables similarity search across processed documents
- **Modular Design**: Each processing step is independently configurable

### Main Entry Point (`src/main.py`)
- **CLI Interface**: Process single URLs, batch processing, or interactive mode
- **Search Functionality**: Query the knowledge graph for similar content
- **Configuration**: All settings read from environment variables
- **Output Formats**: JSON export and console output

## Prerequisites

- Python 3.8+
- OpenAI API Key
- Qdrant vector database (local or remote)

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd hpact-edtech
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file:
   ```bash
   # Required
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Qdrant Configuration
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   QDRANT_COLLECTION_NAME=langchain_knowledge_graph
   
   # OpenAI Configuration (optional, defaults provided)
   OPENAI_MODEL=gpt-3.5-turbo
   OPENAI_TEMPERATURE=0.1
   OPENAI_EMBEDDING_MODEL=text-embedding-3-large
   EMBEDDING_SIZE=3072
   
   # Web Crawler Configuration (optional)
   WEB_CRAWLER_TIMEOUT=10
   WEB_CRAWLER_MAX_CONTENT_LENGTH=50000
   ```

4. **Start Qdrant (if running locally):**
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

## Usage

### Command Line Interface

1. **Process a single URL:**
   ```bash
   python src/main.py --url https://example.com/article
   ```

2. **Process multiple URLs from file:**
   ```bash
   # Create urls.txt with one URL per line
   python src/main.py --urls-file urls.txt
   ```

3. **Search the knowledge graph:**
   ```bash
   python src/main.py --search "machine learning algorithms"
   ```

4. **Interactive mode:**
   ```bash
   python src/main.py
   # Then enter URLs or 'search <query>' commands
   ```

5. **Export results to JSON:**
   ```bash
   python src/main.py --url https://example.com --output results.json
   ```

### Programmatic Usage

```python
from src.web_crawler import create_web_crawler
from src.pipeline import create_pipeline

# Initialize components
crawler = create_web_crawler()
pipeline = create_pipeline()

# Process a URL
document = crawler.crawl_url("https://example.com/article")
result = pipeline.process_document(document)

print(f"Processed document: {result['doc_id']}")
print(f"Generated tags: {result['tags']['primary_tags']}")

# Search for similar content
similar_docs = pipeline.search_similar_documents("artificial intelligence", limit=5)
```

## Features

### Sophisticated Web Crawling
- **Intelligent Content Extraction**: Uses trafilatura for main content identification
- **Robust Parsing**: BeautifulSoup fallback for reliable HTML processing
- **Metadata Extraction**: Titles, descriptions, keywords, Open Graph data
- **Content Filtering**: Removes navigation, ads, and boilerplate content
- **Error Handling**: Graceful degradation with multiple extraction strategies

### Advanced Content Analysis
- **Structured Tagging**: JSON-formatted tags with categories, topics, and sentiment
- **Content Classification**: Automatic categorization by type and complexity
- **Key Concept Extraction**: Identifies important terms and concepts
- **Summarization**: Generates concise content summaries
- **Multilingual Support**: Works with content in various languages

### Knowledge Graph Storage
- **Vector Database**: High-performance semantic storage in Qdrant
- **Rich Metadata**: Preserves all extracted information and analysis
- **Semantic Search**: Find similar content using natural language queries
- **Scalable Storage**: Handles large collections of processed documents
- **Efficient Retrieval**: Fast similarity search with configurable limits

### LangChain Integration
- **Document Objects**: Native LangChain Document format throughout pipeline
- **Chain Composition**: Modular processing chains for different tasks
- **Model Flexibility**: Easy switching between different OpenAI models
- **Prompt Engineering**: Optimized prompts for content analysis tasks
- **Error Handling**: Robust error handling and fallback strategies

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_web_crawler.py -v
pytest tests/test_pipeline.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Coverage

- **Web Crawler Tests**: Mock HTTP requests, test extraction methods, error handling
- **Pipeline Tests**: Mock LangChain components, test processing workflow
- **Integration Tests**: End-to-end workflow testing with mocked external services
- **Error Scenarios**: Comprehensive error handling and edge case testing

## Configuration

All configuration is handled through environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | Required | OpenAI API key for language models |
| `QDRANT_HOST` | localhost | Qdrant server host |
| `QDRANT_PORT` | 6333 | Qdrant server port |
| `QDRANT_COLLECTION_NAME` | langchain_knowledge_graph | Collection name for documents |
| `OPENAI_MODEL` | gpt-3.5-turbo | OpenAI model for content analysis |
| `OPENAI_TEMPERATURE` | 0.1 | Model temperature for consistent results |
| `OPENAI_EMBEDDING_MODEL` | text-embedding-3-large | Embedding model |
| `EMBEDDING_SIZE` | 3072 | Embedding vector dimensions |
| `WEB_CRAWLER_TIMEOUT` | 10 | HTTP request timeout in seconds |
| `WEB_CRAWLER_MAX_CONTENT_LENGTH` | 50000 | Maximum content length to process |

## Performance & Scalability

- **Efficient Processing**: Optimized content extraction and processing pipeline
- **Batch Operations**: Support for processing multiple URLs efficiently  
- **Memory Management**: Controlled content length and smart truncation
- **Error Recovery**: Robust error handling without stopping batch processing
- **Scalable Storage**: Qdrant vector database handles large document collections
- **Concurrent Processing**: Can be extended for parallel URL processing

## Examples

### Processing News Articles
```bash
python src/main.py --url https://news.site.com/article/123
```
Output includes structured tags for topics, sentiment, complexity, and key concepts.

### Building a Knowledge Base
```bash
# Process a list of documentation URLs
python src/main.py --urls-file documentation_urls.txt --output knowledge_base.json
```

### Semantic Search
```bash
python src/main.py --search "machine learning best practices" --limit 10
```
Returns similar documents with relevance scores and content previews.

## Contributing

1. Follow the modular architecture patterns established in the codebase
2. Add comprehensive tests for new functionality
3. Update documentation for new features or configuration options
4. Use environment variables for all configuration
5. Follow LangChain patterns for document processing and chain composition

## License

[Add your license information here]


