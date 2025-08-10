# EdTech Content Processing System

A comprehensive modular LangChain pipeline for web content processing, automated tagging, embedding generation, and vector storage. This system combines advanced web crawling, OpenAI-powered content analysis, and Qdrant vector database for educational technology applications.

## 🏗️ Architecture

The system follows a modular architecture with clear separation of concerns:

```
src/
├── main.py           # FastAPI application entry point
├── pipeline.py       # Modular LangChain processing pipeline
├── web_crawler.py    # Enhanced web crawler with trafilatura/BeautifulSoup
└── __init__.py       # Package initialization
```

### Core Components

#### 1. **Web Crawler** (`web_crawler.py`)
- **Primary Method**: Trafilatura for high-quality content extraction
- **Fallback Method**: BeautifulSoup for edge cases and custom content
- **Features**: 
  - Intelligent content extraction
  - Metadata preservation
  - Error handling and validation
  - Configurable content length limits

#### 2. **Processing Pipeline** (`pipeline.py`)
- **Content Tagging**: LangChain + OpenAI GPT-3.5-turbo for intelligent categorization
- **Text Splitting**: Recursive character text splitter for optimal chunk sizes
- **Embedding Generation**: OpenAI text-embedding-3-large (3072 dimensions)
- **Vector Storage**: Qdrant for similarity search and retrieval
- **Experiment Tracking**: Optional Neptune integration

#### 3. **Main Application** (`main.py`)
- **FastAPI Framework**: Modern async web framework
- **RESTful API**: Clean endpoints for processing and management
- **Background Tasks**: Efficient batch processing
- **Health Monitoring**: Comprehensive health checks
- **Error Handling**: Robust error management

## 🚀 Quick Start

### Prerequisites

- **Docker** (recommended for deployment)
- **Python 3.11+** (for local development)
- **OpenAI API Key** (required)
- **Qdrant Instance** (can use Docker or cloud)

### Environment Variables

Create a `.env` file in the root directory:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=content_vectors

# Optional Configuration
OPENAI_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-3-large
MAX_CONTENT_LENGTH=10000
PORT=8000
HOST=0.0.0.0

# Optional Neptune Experiment Tracking
NEPTUNE_PROJECT=your_neptune_project
NEPTUNE_API_TOKEN=your_neptune_token
```

## 🐳 Docker Deployment (Recommended)

### 1. Build the Container

```bash
docker build -t edtech-content-processor .
```

### 2. Start Dependencies

```bash
# Start Qdrant vector database
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant
```

### 3. Run the Application

```bash
docker run --env-file .env -p 8000:8000 edtech-content-processor
```

### 4. Docker Compose (Complete Setup)

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_storage:/qdrant/storage:z
    
  edtech-processor:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      - qdrant
    environment:
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
```

Run with:
```bash
docker-compose up -d
```

## 💻 Local Development

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd hpact-edtech

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Development Server

```bash
# Start Qdrant (in separate terminal)
docker run -p 6333:6333 qdrant/qdrant

# Start the application
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Alternative: Direct Python Execution

```bash
python -m src.main
```

## 📚 API Usage

### Interactive Documentation

Once running, access the interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Core Endpoints

#### 1. Process Single URL

```bash
curl -X POST "http://localhost:8000/process" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'
```

**Response:**
```json
{
  "url": "https://example.com/article",
  "title": "Article Title",
  "tags": ["education", "technology", "AI"],
  "content_length": 2500,
  "num_chunks": 3,
  "processing_time": 2.5,
  "success": true
}
```

#### 2. Batch Processing

```bash
curl -X POST "http://localhost:8000/process-batch" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://example.com/article1",
      "https://example.com/article2",
      "https://example.com/article3"
    ]
  }'
```

#### 3. Health Check

```bash
curl "http://localhost:8000/health"
```

#### 4. Test Web Crawler

```bash
curl "http://localhost:8000/crawl?url=https://example.com"
```

#### 5. View Configuration

```bash
curl "http://localhost:8000/config"
```

## 🧪 Testing

### Manual Testing

#### Test Web Crawler
```bash
cd /path/to/project
python -c "
from src.web_crawler import WebCrawler
crawler = WebCrawler()
result = crawler.crawl('https://en.wikipedia.org/wiki/Artificial_intelligence')
print(f'Success: {result.success}')
print(f'Content length: {len(result.content)}')
print(f'Title: {result.title}')
crawler.close()
"
```

#### Test Pipeline Components
```bash
python -c "
import os
os.environ['OPENAI_API_KEY'] = 'your-key-here'
from src.pipeline import create_pipeline
pipeline = create_pipeline()
result = pipeline.process_url('https://en.wikipedia.org/wiki/Machine_learning')
print(f'Success: {result.success}')
print(f'Tags: {result.tags}')
print(f'Chunks: {len(result.chunks)}')
pipeline.close()
"
```

### Build Verification

```bash
# Test Docker build
docker build -t edtech-test .

# Test Python imports
python -c "
from src import ContentProcessingPipeline, WebCrawler
print('All imports successful!')
"

# Test API startup (without keys)
python -c "
import sys
sys.path.append('.')
from src.main import app
print('FastAPI app created successfully!')
"
```

### Integration Testing

```bash
# Start the application
docker-compose up -d

# Wait for startup
sleep 10

# Test health endpoint
curl -f http://localhost:8000/health

# Test crawl endpoint (no OpenAI key needed)
curl -f "http://localhost:8000/crawl?url=https://httpbin.org/html"

echo "Integration tests passed!"
```

## 🔧 Configuration

### Pipeline Configuration

Customize the processing pipeline through environment variables or the `ProcessingConfig` class:

```python
from src.pipeline import ProcessingConfig, create_pipeline

config = ProcessingConfig(
    openai_model="gpt-4",                    # OpenAI model for tagging
    embedding_model="text-embedding-3-large", # Embedding model
    embedding_dimension=3072,                # Vector dimensions
    max_chunk_size=1500,                     # Text chunk size
    chunk_overlap=300,                       # Chunk overlap
    collection_name="custom_vectors",        # Qdrant collection
    max_content_length=15000                 # Max content to process
)

pipeline = create_pipeline(config=config)
```

### Web Crawler Configuration

```python
from src.web_crawler import WebCrawler

crawler = WebCrawler(
    max_content_length=8000,    # Max content extraction
    timeout=60                  # Request timeout
)
```

## 🏗️ Development Guidelines

### Code Structure

- **Modular Design**: Each component is independently testable
- **Type Hints**: Full type annotations for better IDE support
- **Docstrings**: Comprehensive documentation for all functions
- **Error Handling**: Graceful degradation and informative errors
- **Logging**: Structured logging for monitoring and debugging

### Adding New Features

1. **Create Feature Branch**: `git checkout -b feature/new-feature`
2. **Implement Changes**: Follow existing patterns and conventions
3. **Add Documentation**: Update docstrings and README if needed
4. **Test Changes**: Verify build and functionality
5. **Submit PR**: Include description of changes and testing done

### Code Quality

The codebase follows these principles:
- **Single Responsibility**: Each class/function has one clear purpose
- **Dependency Injection**: Easy testing and configuration
- **Error Boundaries**: Errors are contained and handled gracefully
- **Resource Management**: Proper cleanup of resources (sessions, clients)

## 🚨 Troubleshooting

### Common Issues

#### 1. OpenAI API Key Issues
```
Error: OpenAI API key not configured
```
**Solution**: Ensure `OPENAI_API_KEY` is set in your environment or `.env` file.

#### 2. Qdrant Connection Issues
```
Error: Failed to connect to Qdrant
```
**Solution**: 
- Check if Qdrant is running: `docker ps | grep qdrant`
- Verify `QDRANT_HOST` and `QDRANT_PORT` settings
- The system will fall back to in-memory mode if Qdrant is unavailable

#### 3. Memory Issues with Large Content
```
Error: Content processing failed due to memory limits
```
**Solution**: Reduce `MAX_CONTENT_LENGTH` or `max_chunk_size` in configuration.

#### 4. Trafilatura Extraction Fails
**Info**: System automatically falls back to BeautifulSoup extraction.

### Debug Mode

Enable verbose logging:
```bash
export PYTHONPATH=/path/to/project
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from src.main import app
"
```

### Health Monitoring

Monitor application health:
```bash
# Check API health
curl http://localhost:8000/health

# Check Qdrant health
curl http://localhost:6333/health

# View logs
docker logs <container-id>
```

## 📄 Dependencies

### Core Dependencies
- **FastAPI**: Modern web framework
- **LangChain**: LLM orchestration framework
- **OpenAI**: GPT models and embeddings
- **Qdrant**: Vector database
- **Trafilatura**: Content extraction
- **BeautifulSoup**: HTML parsing fallback

### Optional Dependencies
- **Neptune**: Experiment tracking
- **Docker**: Containerization
- **Uvicorn**: ASGI server

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation at `/docs`
3. Open an issue on GitHub
4. Check logs for detailed error information

---

**Built with ❤️ for educational technology applications**