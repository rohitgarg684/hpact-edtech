# HPACT EdTech - LangChain Document Processing & Knowledge Management

[![FastAPI](https://img.shields.io/badge/FastAPI-0.116.1-009688.svg?style=flat&logo=FastAPI)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.27-1C3C3C.svg?style=flat)](https://langchain.readthedocs.io)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5--Turbo-412991.svg?style=flat&logo=openai)](https://openai.com)

## Overview

A next-generation Python microservice that leverages **LangChain** as the orchestrator for advanced document processing and knowledge management workflows. The system integrates state-of-the-art web crawling, AI-powered content analysis, vector database storage, and knowledge graph construction.

### Key Features

- 🚀 **LangChain Orchestration**: Complete workflow management using LangChain
- 🕷️ **Advanced Web Crawler**: State-of-the-art content extraction with intelligent parsing
- 🧠 **OpenAI Integration**: GPT-3.5 Turbo for tagging, text-embedding-3-large for embeddings  
- 🔍 **Vector Search**: Qdrant integration for semantic similarity search
- 📊 **Knowledge Graph**: AWS Neptune for entity relationships and semantic connections
- 🎯 **Smart Content Analysis**: Automated tagging, categorization, and theme extraction
- ⚡ **Async Processing**: High-performance batch processing capabilities
- 🔧 **Environment Configuration**: Pure environment variable configuration (no config files)

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Crawler   │───▶│   LangChain      │───▶│   OpenAI        │
│   (Beautiful    │    │   Orchestrator   │    │   GPT-3.5 Turbo │
│   Soup + Smart  │    │                  │    │   + Embeddings  │
│   Extraction)   │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                        ┌───────▼───────┐
                        │   Document    │
                        │   Processing  │
                        │   Pipeline    │
                        └───────┬───────┘
                                │
           ┌────────────────────┼────────────────────┐
           ▼                    ▼                    ▼
  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
  │     Qdrant      │  │   AWS Neptune   │  │   Enhanced      │
  │  Vector Store   │  │ Knowledge Graph │  │   Metadata      │
  │  (Similarity    │  │  (Entities &    │  │   & Tagging     │
  │   Search)       │  │ Relationships)  │  │                 │
  └─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Prerequisites

- **Docker** (recommended)
- **Python 3.11+**
- **OpenAI API Key** (required)
- **Qdrant Instance** (local or cloud)
- **AWS Neptune Cluster** (optional, for knowledge graph features)

## Environment Variables

Create a `.env` file or set these environment variables:

### Required
```bash
# OpenAI Configuration (Required)
OPENAI_API_KEY=your_openai_api_key_here

# Qdrant Configuration (Required)
QDRANT_HOST=localhost                    # or your Qdrant cloud endpoint
QDRANT_PORT=6333
QDRANT_API_KEY=your_qdrant_api_key      # Optional for cloud instances
QDRANT_COLLECTION=hpact_documents        # Optional, defaults to "hpact_documents"
```

### Optional (for Knowledge Graph features)
```bash
# AWS Neptune Configuration (Optional)
NEPTUNE_ENDPOINT=your-neptune-cluster.cluster-xyz.us-east-1.neptune.amazonaws.com
NEPTUNE_PORT=8182
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_SESSION_TOKEN=your_session_token     # Optional for STS

# Neptune IAM Role (Alternative to access keys)
NEPTUNE_IAM_ROLE=your_neptune_iam_role   # Optional
```

## Quick Start

### Option 1: Docker (Recommended)

1. **Clone and build:**
   ```bash
   git clone <repository-url>
   cd hpact-edtech
   docker build -t hpact-edtech .
   ```

2. **Run with environment variables:**
   ```bash
   docker run --env-file .env -p 8000:8000 hpact-edtech
   ```

### Option 2: Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the server:**
   ```bash
   # With environment variables
   export OPENAI_API_KEY=your_key_here
   export QDRANT_HOST=localhost
   
   # Start server
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

## API Endpoints

### Core Processing

#### `POST /process-url/`
Process a single URL through the complete LangChain pipeline.

```bash
curl -X POST "http://localhost:8000/process-url/" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'
```

**Response:**
```json
{
  "url": "https://example.com/article",
  "status": "completed",
  "processing_steps": {
    "crawling": {"duration": 2.1, "documents_extracted": 3},
    "tagging": {"duration": 1.8, "documents_tagged": 3},
    "embedding": {"duration": 1.2, "embeddings_generated": 3},
    "vector_storage": {"duration": 0.5, "documents_stored": 3},
    "knowledge_graph": {"duration": 3.2, "extracted_nodes": 15}
  },
  "summary": {
    "documents_processed": 3,
    "unique_tags": 12,
    "knowledge_graph_nodes": 15
  }
}
```

#### `POST /process-multiple/`
Process multiple URLs in batches.

```bash
curl -X POST "http://localhost:8000/process-multiple/" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://example1.com",
      "https://example2.com"
    ],
    "batch_size": 5
  }'
```

### Search & Retrieval

#### `GET /search/`
Search processed content using vector similarity.

```bash
curl "http://localhost:8000/search/?query=machine+learning&k=5&include_graph=true"
```

**Response:**
```json
{
  "query": "machine learning",
  "vector_results": [
    {
      "document": {
        "content": "Article about machine learning...",
        "metadata": {"source_url": "https://example.com", "tags": ["AI", "ML"]}
      },
      "similarity_score": 0.92
    }
  ],
  "graph_context": [...]
}
```

### System Monitoring

#### `GET /health/`
Comprehensive health check for all services.

```bash
curl http://localhost:8000/health/
```

#### `GET /stats/`
Get processing statistics and service status.

```bash
curl http://localhost:8000/stats/
```

### Legacy Compatibility

#### `POST /tag-and-embed/` (Deprecated)
Legacy endpoint for backward compatibility. Redirects to new pipeline.

## Advanced Features

### 1. Smart Web Crawling
- **Intelligent Content Extraction**: BeautifulSoup + HTML2Text for clean content
- **Recursive Crawling**: Follow links within domain boundaries
- **Content Filtering**: Automatic filtering of short/irrelevant content
- **Metadata Enhancement**: Extract titles, descriptions, and page metadata

### 2. AI-Powered Content Analysis
- **Multi-dimensional Tagging**: Topics, categories, themes, content types
- **Semantic Embeddings**: High-dimensional vector representations
- **Context Awareness**: Understanding document relationships and dependencies
- **Quality Assessment**: Content quality and relevance scoring

### 3. Vector Database Operations
- **Similarity Search**: Find semantically similar content
- **Metadata Filtering**: Search by tags, categories, or custom attributes
- **Batch Operations**: Efficient bulk storage and retrieval
- **Collection Management**: Dynamic collection creation and management

### 4. Knowledge Graph Construction
- **Entity Extraction**: Identify people, organizations, locations, concepts
- **Relationship Mapping**: Discover and store semantic relationships
- **Graph Traversal**: Find related entities and connection paths
- **Contextual Queries**: Search based on entity relationships

## Configuration Examples

### Local Development Setup
```bash
# .env file for local development
OPENAI_API_KEY=sk-your-openai-key
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### Production Cloud Setup
```bash
# .env file for production
OPENAI_API_KEY=sk-your-openai-key
QDRANT_HOST=your-cluster.qdrant.cloud
QDRANT_PORT=6333
QDRANT_API_KEY=your-qdrant-cloud-key
NEPTUNE_ENDPOINT=your-neptune-cluster.amazonaws.com
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
```

## Performance & Scaling

### Batch Processing
- **Concurrent Processing**: Process multiple URLs simultaneously
- **Rate Limiting**: Built-in rate limiting to respect API quotas
- **Error Handling**: Graceful failure handling with detailed error reporting
- **Progress Tracking**: Real-time processing status updates

### Resource Management
- **Memory Efficient**: Streaming processing for large documents
- **Connection Pooling**: Efficient database connection management
- **Async Operations**: Non-blocking I/O for improved throughput
- **Configurable Limits**: Customizable processing limits and timeouts

## Monitoring & Observability

### Health Checks
- **Service Status**: Monitor all integrated services
- **Connectivity Tests**: Verify external service connections
- **Performance Metrics**: Track processing times and success rates

### Logging
- **Structured Logging**: JSON-formatted logs for easy parsing
- **Error Tracking**: Detailed error reporting with stack traces
- **Performance Monitoring**: Processing time and resource usage tracking

## Error Handling

The system provides comprehensive error handling:

- **Network Errors**: Retry mechanisms for transient failures
- **API Rate Limits**: Automatic backoff and retry strategies
- **Data Validation**: Input validation with helpful error messages
- **Service Failures**: Graceful degradation when services are unavailable

## Development & Testing

### Running Tests
```bash
# Install development dependencies
pip install -r requirements.txt

# Run health check
curl http://localhost:8000/health/

# Test with sample URL
curl -X POST "http://localhost:8000/process-url/" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://httpbin.org/html"}'
```

### Custom Extensions
The modular architecture allows easy extension:
- **Custom Crawlers**: Add specialized crawlers for specific sites
- **Additional AI Models**: Integrate other language models
- **Custom Storage**: Add support for additional databases
- **Processing Plugins**: Create custom processing steps

## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   ```
   Solution: Ensure OPENAI_API_KEY is set correctly
   ```

2. **Qdrant Connection Failed**
   ```
   Solution: Verify QDRANT_HOST and QDRANT_PORT, or service will use in-memory storage
   ```

3. **Neptune Connection Error**
   ```
   Solution: Neptune is optional. Check AWS credentials and endpoint configuration
   ```

4. **Memory Issues**
   ```
   Solution: Reduce batch_size or implement document chunking for large content
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality  
5. Submit a pull request

---

**Built with ❤️ using LangChain, FastAPI, and modern AI technologies**
