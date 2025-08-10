from fastapi import FastAPI, HTTPException, BackgroundTasks
from app.models.schemas import UrlInput
from app.services.document_processor import LangChainDocumentProcessor
from typing import List, Optional
import logging
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HPACT EdTech - LangChain Document Processing API",
    description="Advanced document processing and knowledge management using LangChain orchestration",
    version="2.0.0"
)

# Initialize the main LangChain document processor
try:
    document_processor = LangChainDocumentProcessor()
    logger.info("Successfully initialized LangChain Document Processor")
except Exception as e:
    logger.error(f"Failed to initialize document processor: {str(e)}")
    document_processor = None


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "HPACT EdTech - LangChain Document Processing API",
        "version": "2.0.0",
        "description": "Advanced document processing with web crawling, OpenAI tagging/embedding, Qdrant vector storage, and Neptune knowledge graphs",
        "endpoints": {
            "process_url": "/process-url/",
            "process_multiple": "/process-multiple/", 
            "search": "/search/",
            "health": "/health/",
            "stats": "/stats/"
        }
    }


@app.post("/process-url/")
async def process_url(input_data: UrlInput):
    """
    Process a single URL through the complete LangChain pipeline:
    1. Advanced web crawling and content extraction
    2. Document tagging using OpenAI 3.5 Turbo
    3. Embedding generation using OpenAI Large Language Model  
    4. Vector storage in Qdrant
    5. Knowledge graph extraction and storage in Neptune
    """
    if not document_processor:
        raise HTTPException(
            status_code=503, 
            detail="Document processor not available. Check service configuration."
        )
    
    try:
        result = await document_processor.process_url(input_data.url)
        
        if result["status"] == "failed":
            raise HTTPException(status_code=400, detail=result.get("error", "Processing failed"))
        
        return result
        
    except Exception as e:
        logger.error(f"URL processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/process-multiple/")
async def process_multiple_urls(urls: List[str], batch_size: Optional[int] = 5):
    """
    Process multiple URLs in batches through the LangChain pipeline.
    
    Args:
        urls: List of URLs to process
        batch_size: Number of URLs to process concurrently (default: 5)
    """
    if not document_processor:
        raise HTTPException(
            status_code=503,
            detail="Document processor not available. Check service configuration."
        )
    
    if not urls or len(urls) == 0:
        raise HTTPException(status_code=400, detail="URLs list cannot be empty")
    
    if len(urls) > 50:  # Limit to prevent abuse
        raise HTTPException(status_code=400, detail="Maximum 50 URLs allowed per request")
    
    try:
        results = await document_processor.process_multiple_urls(urls, batch_size)
        
        # Summary statistics
        successful = sum(1 for r in results if r["status"] == "completed")
        failed = len(results) - successful
        
        return {
            "total_urls": len(urls),
            "successful": successful,
            "failed": failed,
            "batch_size": batch_size,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Multiple URL processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


@app.get("/search/")
async def search_content(
    query: str,
    k: Optional[int] = 5,
    include_graph: Optional[bool] = False
):
    """
    Search for similar content using vector similarity and optionally knowledge graph context.
    
    Args:
        query: Search query text
        k: Number of results to return (default: 5, max: 20)
        include_graph: Include knowledge graph context in results
    """
    if not document_processor:
        raise HTTPException(
            status_code=503,
            detail="Document processor not available. Check service configuration."
        )
    
    if not query or len(query.strip()) < 3:
        raise HTTPException(status_code=400, detail="Query must be at least 3 characters long")
    
    if k > 20:
        k = 20  # Limit results
    
    try:
        results = document_processor.search_similar_content(
            query=query,
            k=k,
            include_graph_context=include_graph
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Content search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/health/")
async def health_check():
    """
    Comprehensive health check for all services in the LangChain pipeline.
    """
    if not document_processor:
        return {
            "status": "unhealthy",
            "error": "Document processor not initialized"
        }
    
    try:
        health_status = await document_processor.health_check()
        
        # Determine overall health
        service_statuses = [
            service.get("status", "unknown") 
            for service in health_status.get("services", {}).values()
        ]
        
        if all(status in ["healthy", "disabled"] for status in service_statuses):
            overall_status = "healthy"
        elif any(status == "healthy" for status in service_statuses):
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "details": health_status
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.get("/stats/")
async def get_statistics():
    """
    Get comprehensive statistics about processed content and service status.
    """
    if not document_processor:
        raise HTTPException(
            status_code=503,
            detail="Document processor not available. Check service configuration."
        )
    
    try:
        stats = document_processor.get_processing_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"Statistics retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Statistics failed: {str(e)}")


# Legacy endpoint for backward compatibility (deprecated)
@app.post("/tag-and-embed/")
async def tag_and_embed_legacy(input_data: UrlInput):
    """
    Legacy endpoint for backward compatibility. 
    Redirects to the new LangChain-based processing pipeline.
    """
    logger.warning("Using deprecated endpoint /tag-and-embed/. Please use /process-url/ instead.")
    
    # Process using new pipeline
    result = await process_url(input_data)
    
    # Return simplified response for backward compatibility
    summary = result.get("summary", {})
    return {
        "url": input_data.url,
        "tags": summary.get("unique_tags", 0),
        "embedding_dim": result.get("processing_steps", {}).get("embedding", {}).get("embedding_dimension", 0),
        "documents_processed": summary.get("documents_processed", 0),
        "status": result.get("status", "unknown")
    }
