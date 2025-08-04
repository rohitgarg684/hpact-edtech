import os
from fastapi import APIRouter
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance

COLLECTION_NAME = "openai_tagged_data"
EMBEDDING_SIZE = 3072  # for text-embedding-3-large

router = APIRouter()

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

@router.get("/health", tags=["health"])
def health_check():
    """
    Health check endpoint for the Qdrant service.
    Returns 200 OK if the service is up.
    """
    return {"status": "ok"}