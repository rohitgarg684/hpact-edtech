import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance

COLLECTION_NAME = "openai_tagged_data"
EMBEDDING_SIZE = 3072  # for text-embedding-3-large

class QdrantService:
    def __init__(self):
        self.client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333"))
        )
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
