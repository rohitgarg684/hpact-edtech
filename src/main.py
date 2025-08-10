"""
Main Application Entry Point for the EdTech Content Processing System

This module provides the main FastAPI application that orchestrates web crawling,
content processing, and vector storage using the modular LangChain pipeline.
It offers both individual URL processing and batch processing capabilities.
"""

import os
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field
from dotenv import load_dotenv

# Import our pipeline components
from .pipeline import ContentProcessingPipeline, ProcessingConfig, ProcessingResult, create_pipeline
from .web_crawler import WebCrawler

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline: Optional[ContentProcessingPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for the FastAPI app."""
    global pipeline
    
    # Startup
    logger.info("Initializing content processing pipeline...")
    config = ProcessingConfig(
        openai_model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        collection_name=os.getenv("QDRANT_COLLECTION", "content_vectors"),
        max_content_length=int(os.getenv("MAX_CONTENT_LENGTH", "10000"))
    )
    
    neptune_project = os.getenv("NEPTUNE_PROJECT")
    pipeline = create_pipeline(config=config, neptune_project=neptune_project)
    
    logger.info("Pipeline initialized successfully")
    yield
    
    # Shutdown
    logger.info("Shutting down pipeline...")
    if pipeline:
        pipeline.close()
    logger.info("Pipeline shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="EdTech Content Processing API",
    description="A modular LangChain pipeline for web content processing, tagging, and vector storage",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API
class UrlInput(BaseModel):
    """Input model for single URL processing."""
    url: HttpUrl = Field(..., description="URL to process")
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com/article"
            }
        }


class UrlBatchInput(BaseModel):
    """Input model for batch URL processing."""
    urls: List[HttpUrl] = Field(..., min_items=1, max_items=10, description="List of URLs to process")
    
    class Config:
        json_schema_extra = {
            "example": {
                "urls": [
                    "https://example.com/article1",
                    "https://example.com/article2"
                ]
            }
        }


class ProcessingResponse(BaseModel):
    """Response model for processing results."""
    url: str
    title: Optional[str]
    tags: List[str]
    content_length: int
    num_chunks: int
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    
    @classmethod
    def from_result(cls, result: ProcessingResult) -> "ProcessingResponse":
        """Create response from processing result."""
        return cls(
            url=result.url,
            title=result.title,
            tags=result.tags,
            content_length=len(result.content),
            num_chunks=len(result.chunks),
            processing_time=result.processing_time,
            success=result.success,
            error_message=result.error_message
        )


class BatchProcessingResponse(BaseModel):
    """Response model for batch processing."""
    total_urls: int
    successful: int
    failed: int
    results: List[ProcessingResponse]
    total_processing_time: float


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    pipeline_initialized: bool
    services: dict


# Dependency to get pipeline
def get_pipeline() -> ContentProcessingPipeline:
    """Get the global pipeline instance."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipeline


# API Endpoints
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "EdTech Content Processing API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    
    Returns the status of the API and its dependencies.
    """
    pipeline_initialized = pipeline is not None
    
    services = {
        "openai_api_key": bool(os.getenv("OPENAI_API_KEY")),
        "qdrant_configured": bool(os.getenv("QDRANT_HOST")),
        "neptune_configured": bool(os.getenv("NEPTUNE_PROJECT"))
    }
    
    status = "healthy" if pipeline_initialized else "unhealthy"
    
    return HealthResponse(
        status=status,
        pipeline_initialized=pipeline_initialized,
        services=services
    )


@app.post("/process", response_model=ProcessingResponse, tags=["processing"])
async def process_url(
    input_data: UrlInput,
    pipeline: ContentProcessingPipeline = Depends(get_pipeline)
) -> ProcessingResponse:
    """
    Process a single URL through the complete pipeline.
    
    This endpoint:
    1. Crawls the provided URL to extract content
    2. Generates relevant tags using OpenAI
    3. Splits content into manageable chunks
    4. Generates embeddings for each chunk
    5. Stores vectors in Qdrant for similarity search
    
    Args:
        input_data: URL input containing the URL to process
        
    Returns:
        ProcessingResponse: Processing results including tags, metrics, and status
        
    Raises:
        HTTPException: If processing fails or URL is invalid
    """
    try:
        logger.info(f"Processing URL: {input_data.url}")
        result = pipeline.process_url(str(input_data.url))
        
        if not result.success:
            raise HTTPException(
                status_code=400,
                detail=f"Processing failed: {result.error_message}"
            )
        
        response = ProcessingResponse.from_result(result)
        logger.info(f"Successfully processed URL: {input_data.url}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to process URL {input_data.url}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-batch", response_model=BatchProcessingResponse, tags=["processing"])
async def process_batch_urls(
    input_data: UrlBatchInput,
    background_tasks: BackgroundTasks,
    pipeline: ContentProcessingPipeline = Depends(get_pipeline)
) -> BatchProcessingResponse:
    """
    Process multiple URLs in batch through the pipeline.
    
    This endpoint processes multiple URLs concurrently for better performance.
    Failed URLs will not stop the processing of successful ones.
    
    Args:
        input_data: Batch input containing list of URLs to process
        background_tasks: FastAPI background tasks for async processing
        
    Returns:
        BatchProcessingResponse: Batch processing results with individual URL results
        
    Raises:
        HTTPException: If batch processing setup fails
    """
    try:
        urls = [str(url) for url in input_data.urls]
        logger.info(f"Processing batch of {len(urls)} URLs")
        
        import time
        start_time = time.time()
        
        # Process all URLs
        results = pipeline.process_urls(urls)
        
        # Calculate metrics
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        total_processing_time = time.time() - start_time
        
        # Convert results
        response_results = [ProcessingResponse.from_result(r) for r in results]
        
        response = BatchProcessingResponse(
            total_urls=len(urls),
            successful=successful,
            failed=failed,
            results=response_results,
            total_processing_time=total_processing_time
        )
        
        logger.info(f"Batch processing complete: {successful}/{len(urls)} successful")
        return response
        
    except Exception as e:
        logger.error(f"Failed to process batch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/crawl", tags=["utilities"])
async def crawl_url_only(url: HttpUrl):
    """
    Crawl a URL and return raw content (without processing).
    
    This endpoint is useful for testing the web crawler functionality
    without running the full processing pipeline.
    
    Args:
        url: URL to crawl
        
    Returns:
        dict: Raw crawl results including content and metadata
    """
    try:
        crawler = WebCrawler()
        result = crawler.crawl(str(url))
        crawler.close()
        
        return {
            "url": result.url,
            "title": result.title,
            "content_length": len(result.content),
            "content_preview": result.content[:500] + "..." if len(result.content) > 500 else result.content,
            "metadata": result.metadata,
            "success": result.success,
            "error_message": result.error_message
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config", tags=["utilities"])
async def get_configuration():
    """
    Get current pipeline configuration.
    
    Returns:
        dict: Current pipeline configuration settings
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    config_dict = {
        "openai_model": pipeline.config.openai_model,
        "embedding_model": pipeline.config.embedding_model,
        "embedding_dimension": pipeline.config.embedding_dimension,
        "max_chunk_size": pipeline.config.max_chunk_size,
        "chunk_overlap": pipeline.config.chunk_overlap,
        "collection_name": pipeline.config.collection_name,
        "max_content_length": pipeline.config.max_content_length
    }
    
    return {
        "configuration": config_dict,
        "environment": {
            "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
            "qdrant_host": os.getenv("QDRANT_HOST", "localhost"),
            "qdrant_port": os.getenv("QDRANT_PORT", "6333"),
            "neptune_project": os.getenv("NEPTUNE_PROJECT", "Not configured")
        }
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return {"error": "Not found", "status_code": 404}


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {str(exc)}")
    return {"error": "Internal server error", "status_code": 500}


if __name__ == "__main__":
    """Run the application with uvicorn for development."""
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting application on {host}:{port}")
    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )