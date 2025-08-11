"""
pipeline.py

Orchestrates tagging, embedding, and knowledge graph storage using LangChain integrations.
Reads configuration from environment variables.

Functions:
- tag_documents(docs): Tags documents using OpenAI LLM.
- embed_and_store(docs): Embeds documents and stores them in Qdrant.
- build_knowledge_graph(docs): Extracts triples using LLM and stores them in Neptune.
"""

import os
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.graph_stores import NeptuneGraphStore

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "edtech_docs")
NEPTUNE_ENDPOINT = os.environ.get("NEPTUNE_ENDPOINT")
NEPTUNE_PORT = int(os.environ.get("NEPTUNE_PORT", "8182"))

def tag_documents(docs):
    """
    Tags each document using OpenAI LLM.

    Args:
        docs (list): List of LangChain Document objects.

    Returns:
        list: Documents with 'tags' attribute added.
    """
    llm = OpenAI(model="gpt-3.5-turbo")
    for doc in docs:
        doc.tags = llm.predict(f"Tag this document:\n{doc.page_content}")
    return docs

def embed_and_store(docs):
    """
    Embeds documents and stores them in Qdrant vector DB.

    Args:
        docs (list): List of LangChain Document objects.

    Returns:
        Qdrant: Qdrant vector database instance.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectordb = Qdrant.from_documents(
        docs, embeddings, url=QDRANT_URL, collection_name=QDRANT_COLLECTION
    )
    return vectordb

def build_knowledge_graph(docs):
    """
    Extracts triples using LLM and stores them in Neptune knowledge graph DB.

    Args:
        docs (list): List of LangChain Document objects.

    Returns:
        NeptuneGraphStore: Neptune graph store instance.
    """
    llm = OpenAI(model="gpt-3.5-turbo")
    graph_store = NeptuneGraphStore(endpoint=NEPTUNE_ENDPOINT, port=NEPTUNE_PORT)
    for doc in docs:
        triples = llm.predict(f"Extract knowledge graph triples:\n{doc.page_content}")
        graph_store.add_triples(triples)
    return graph_store
