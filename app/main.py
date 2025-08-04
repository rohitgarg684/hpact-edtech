from fastapi import FastAPI, HTTPException
from app.models.schemas import UrlInput
from app.services.url_fetcher import URLFetcher
from app.services.openai_tagger import OpenAITagger
from app.services.openai_embedder import OpenAIEmbedder
from app.services.qdrant_client_service import QdrantService

app = FastAPI()

# Initialize services
url_fetcher = URLFetcher()
tagger = OpenAITagger()
embedder = OpenAIEmbedder()
qdrant_service = QdrantService()

@app.post("/tag-and-embed/")
def tag_and_embed(input_data: UrlInput):
    # 1. Fetch content from URL
    try:
        text = url_fetcher.fetch(input_data.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

    # 2. Tag using GPT-3.5 Turbo
    try:
        tags_text = tagger.get_tags(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI Tagging failed: {e}")

    # 3. Generate embedding
    try:
        embedding = embedder.get_embedding(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI Embedding failed: {e}")

    # 4. Store in Qdrant
    try:
        qdrant_service.upsert_point(input_data.url, text, tags_text, embedding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant upsert failed: {e}")

    return {"url": input_data.url, "tags": tags_text, "embedding_dim": len(embedding)}

# ... rest of your code ...
