"""
Qdrant Client Service

This module provides the QdrantService class for interacting with Qdrant vector database,
and includes a FastAPI health check endpoint.

The module exports:
- QdrantService: Class for vector database operations
- router: FastAPI router with health check endpoint

Usage:
    To include the health check endpoint in your FastAPI application:
    
    ```python
    from fastapi import FastAPI
    from app.services.qdrant_client_service import router
    
    app = FastAPI()
    app.include_router(router)
    ```
    
    This will add a /health endpoint that returns {"status": "ok"}
"""

import os
from fastapi import APIRouter
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance

COLLECTION_NAME = "openai_tagged_data"
EMBEDDING_SIZE = 3072  # for text-embedding-3-large

class QdrantService:
    def __init__(self):
        host = os.environ.get("QDRANT_HOST")
        port = os.environ.get("QDRANT_PORT")
        if host and port:
            self.client = QdrantClient(host=host, port=int(port))
        else:
            # Use in-memory Qdrant if environment variables are missing
            self.client = QdrantClient(":memory:")
        self.init_collection()

    def init_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_SIZE,
                    distance=Distance.COSINE
                )
            )

    def upsert_point(self, url, text, tags, embedding):
        point = PointStruct(
            id=None,
            vector=embedding,
            payload={
                "url": url,
                "text": text,
                "tags": tags,
            }
        )
        self.client.upsert(collection_name=COLLECTION_NAME, points=[point])


# FastAPI Router for health check endpoint
# Usage: Include this router in your FastAPI application with app.include_router(router)
router = APIRouter()


@router.get("/health")
def health_check():
    """
    Health check endpoint for the Qdrant service.
    
    Returns:
        dict: JSON response indicating service health status.
        
    Example:
        GET /health
        Response: {"status": "ok"}
    """
    return {"status": "ok"}