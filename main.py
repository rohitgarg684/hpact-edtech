import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.graph_stores import NeptuneGraphStore
from pipeline import process_documents

# Environment variables (replace with your secrets manager in production)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
NEPTUNE_ENDPOINT = os.environ.get("NEPTUNE_ENDPOINT")
NEPTUNE_PORT = os.environ.get("NEPTUNE_PORT")

def main():
    docs_path = "data/documents"
    loader = DirectoryLoader(docs_path)
    documents = loader.load()
    process_documents(
        documents,
        openai_api_key=OPENAI_API_KEY,
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        neptune_endpoint=NEPTUNE_ENDPOINT,
        neptune_port=NEPTUNE_PORT
    )

if __name__ == "__main__":
    main()