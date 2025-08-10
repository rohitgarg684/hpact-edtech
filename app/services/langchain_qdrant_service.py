import os
from typing import List, Dict, Any, Optional
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
import logging
import uuid

logger = logging.getLogger(__name__)


class LangChainQdrantService:
    """
    LangChain-based Qdrant vector store service for storing and retrieving
    document embeddings with advanced search capabilities.
    """
    
    def __init__(self):
        """Initialize Qdrant vector store through LangChain."""
        # Read configuration from environment variables
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")  # Optional for cloud instances
        self.collection_name = os.getenv("QDRANT_COLLECTION", "hpact_documents")
        
        # OpenAI API key for embeddings
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=openai_api_key
        )
        
        # Initialize Qdrant client
        self._init_qdrant_client()
        
        # Initialize vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.embeddings
        )
        
        logger.info(f"Initialized LangChain Qdrant service with collection: {self.collection_name}")

    def _init_qdrant_client(self):
        """Initialize Qdrant client with proper configuration."""
        try:
            # Initialize client based on available configuration
            if self.qdrant_api_key:
                # Cloud/authenticated instance
                self.client = QdrantClient(
                    host=self.qdrant_host,
                    port=self.qdrant_port,
                    api_key=self.qdrant_api_key,
                    timeout=30
                )
            else:
                # Local instance or in-memory
                if self.qdrant_host == "localhost" or self.qdrant_host == "127.0.0.1":
                    try:
                        self.client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
                    except Exception:
                        logger.warning("Failed to connect to local Qdrant, using in-memory storage")
                        self.client = QdrantClient(":memory:")
                else:
                    self.client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
            
            logger.info(f"Successfully connected to Qdrant at {self.qdrant_host}:{self.qdrant_port}")
            
        except Exception as e:
            logger.warning(f"Failed to connect to Qdrant: {str(e)}, using in-memory storage")
            self.client = QdrantClient(":memory:")

    async def store_documents(
        self, 
        documents: List[Document], 
        embeddings: List[List[float]], 
        tags_data: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Store documents with embeddings and metadata in Qdrant vector store.
        
        Args:
            documents: List of Document objects
            embeddings: List of embedding vectors
            tags_data: List of dictionaries containing tags and metadata
            
        Returns:
            List of document IDs stored in Qdrant
        """
        try:
            if len(documents) != len(embeddings) or len(documents) != len(tags_data):
                raise ValueError("Documents, embeddings, and tags_data must have the same length")
            
            # Prepare documents with enhanced metadata
            enhanced_docs = []
            doc_ids = []
            
            for i, (doc, embedding, tags) in enumerate(zip(documents, embeddings, tags_data)):
                doc_id = str(uuid.uuid4())
                doc_ids.append(doc_id)
                
                # Enhanced metadata combining original, tags, and processing info
                enhanced_metadata = {
                    **doc.metadata,
                    'doc_id': doc_id,
                    'tags': tags.get('tagging_result', {}).get('tags', []),
                    'categories': tags.get('tagging_result', {}).get('categories', []),
                    'content_type': tags.get('tagging_result', {}).get('content_type', 'document'),
                    'themes': tags.get('tagging_result', {}).get('themes', []),
                    'summary': tags.get('tagging_result', {}).get('summary', ''),
                    'source_url': tags.get('source_url', doc.metadata.get('source', 'unknown')),
                    'content_length': len(doc.page_content),
                    'processing_timestamp': str(uuid.uuid4().time_low)  # Simple timestamp
                }
                
                # Create enhanced document
                enhanced_doc = Document(
                    page_content=doc.page_content,
                    metadata=enhanced_metadata
                )
                enhanced_docs.append(enhanced_doc)
            
            # Store documents using LangChain vector store
            await self._async_add_documents(enhanced_docs, embeddings, doc_ids)
            
            logger.info(f"Successfully stored {len(documents)} documents in Qdrant")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Failed to store documents in Qdrant: {str(e)}")
            raise

    async def _async_add_documents(
        self, 
        documents: List[Document], 
        embeddings: List[List[float]], 
        ids: List[str]
    ):
        """Async wrapper for adding documents to vector store."""
        # LangChain QdrantVectorStore doesn't have native async support
        # So we'll use the sync method in a way that's compatible
        self.vector_store.add_documents(documents=documents, ids=ids)

    def search_similar_documents(
        self, 
        query: str, 
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query: Search query text
            k: Number of similar documents to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of similar Document objects
        """
        try:
            # Perform similarity search
            if filter_dict:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vector_store.similarity_search(query=query, k=k)
            
            logger.info(f"Found {len(results)} similar documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search documents: {str(e)}")
            return []

    def search_with_score(
        self, 
        query: str, 
        k: int = 5,
        score_threshold: float = 0.7
    ) -> List[tuple]:
        """
        Search for similar documents with similarity scores.
        
        Args:
            query: Search query text
            k: Number of similar documents to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of (Document, score) tuples
        """
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                score_threshold=score_threshold
            )
            
            logger.info(f"Found {len(results)} documents with scores >= {score_threshold}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search documents with scores: {str(e)}")
            return []

    def search_by_tags(
        self, 
        tags: List[str], 
        k: int = 10
    ) -> List[Document]:
        """
        Search documents by tags using metadata filtering.
        
        Args:
            tags: List of tags to search for
            k: Number of documents to return
            
        Returns:
            List of Document objects matching the tags
        """
        try:
            # Create filter for tags (documents containing any of the specified tags)
            tag_filter = {"tags": {"$in": tags}}
            
            # Since we need a query for similarity search, use the first tag as query
            query = tags[0] if tags else "content"
            
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=tag_filter
            )
            
            logger.info(f"Found {len(results)} documents with tags: {tags}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search by tags: {str(e)}")
            return []

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                'collection_name': self.collection_name,
                'points_count': collection_info.points_count,
                'vectors_count': collection_info.vectors_count,
                'status': collection_info.status.value,
                'config': {
                    'vector_size': collection_info.config.params.vectors.size if collection_info.config.params.vectors else 'unknown',
                    'distance': collection_info.config.params.vectors.distance.value if collection_info.config.params.vectors else 'unknown'
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {str(e)}")
            return {'error': str(e)}

    def delete_documents(self, document_ids: List[str]) -> bool:
        """
        Delete documents by their IDs.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.vector_store.delete(ids=document_ids)
            logger.info(f"Successfully deleted {len(document_ids)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            return False

    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete the entire collection and recreate it
            self.client.delete_collection(self.collection_name)
            
            # Recreate collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=3072,  # text-embedding-3-large dimension
                    distance=Distance.COSINE
                )
            )
            
            # Reinitialize vector store
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embeddings=self.embeddings
            )
            
            logger.info(f"Successfully cleared collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {str(e)}")
            return False