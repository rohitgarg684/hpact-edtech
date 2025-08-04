import os
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-large"

class OpenAIEmbedder:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_embedding(self, text: str):
        emb_resp = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return emb_resp.data[0].embedding
