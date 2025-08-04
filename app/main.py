from fastapi import FastAPI

app = FastAPI()

from app.services.qdrant_client_service import router as qdrant_router
app.include_router(qdrant_router)

# ... rest of your code ...