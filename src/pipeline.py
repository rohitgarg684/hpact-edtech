"""
LangChain pipeline for web content extraction, tagging, embedding, and knowledge graph storage.
Orchestrates the entire processing pipeline using LangChain integrations.
"""

import os
import uuid
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance


class LangChainPipeline:
    """
    A modular pipeline that processes web content through tagging, embedding,
    and knowledge graph storage using LangChain integrations.
    """
    
    def __init__(self):
        """Initialize the pipeline with LangChain components and configuration from environment variables."""
        # OpenAI configuration from environment
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # LangChain components
        self.chat_model = ChatOpenAI(
            api_key=self.openai_api_key,
            model=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
            temperature=float(os.getenv('OPENAI_TEMPERATURE', '0.1'))
        )
        
        self.embeddings = OpenAIEmbeddings(
            api_key=self.openai_api_key,
            model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-large')
        )
        
        # Qdrant configuration from environment
        self.qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
        self.qdrant_port = int(os.getenv('QDRANT_PORT', '6333'))
        self.collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'langchain_knowledge_graph')
        
        # Initialize Qdrant client
        try:
            self.qdrant_client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        except Exception:
            # Fallback to in-memory if connection fails
            self.qdrant_client = QdrantClient(":memory:")
        
        # Initialize collection
        self._init_qdrant_collection()
        
        # Create tagging prompt template
        self.tagging_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_tagging_system_prompt()),
            ("user", "Content to tag:\n\nTitle: {title}\nURL: {url}\nContent: {content}")
        ])
        
        # Create tagging chain
        self.tagging_chain = self.tagging_prompt | self.chat_model | StrOutputParser()
    
    def _get_tagging_system_prompt(self) -> str:
        """Get the system prompt for content tagging."""
        return """You are an intelligent content tagger and analyzer. Your task is to analyze web content and provide structured tags and insights.

Please provide your response in the following JSON format:
{
    "primary_tags": ["tag1", "tag2", "tag3"],
    "categories": ["category1", "category2"],
    "topics": ["topic1", "topic2", "topic3"],
    "content_type": "article|blog|news|documentation|other",
    "sentiment": "positive|negative|neutral",
    "complexity": "beginner|intermediate|advanced",
    "key_concepts": ["concept1", "concept2", "concept3"],
    "summary": "Brief summary of the content"
}

Focus on:
- Relevant and specific tags that describe the content accurately
- Clear categorization based on subject matter
- Key topics and concepts mentioned
- Content characteristics and sentiment
- A concise summary of the main points"""
    
    def _init_qdrant_collection(self):
        """Initialize the Qdrant collection for knowledge graph storage."""
        try:
            collections = [c.name for c in self.qdrant_client.get_collections().collections]
            if self.collection_name not in collections:
                # Get embedding dimension
                embedding_size = int(os.getenv('EMBEDDING_SIZE', '3072'))  # text-embedding-3-large
                
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=embedding_size,
                        distance=Distance.COSINE
                    )
                )
        except Exception as e:
            print(f"Warning: Failed to initialize Qdrant collection: {e}")
    
    def generate_tags(self, document: Document) -> Dict[str, Any]:
        """
        Generate structured tags for a document using LangChain.
        
        Args:
            document: LangChain Document to tag
            
        Returns:
            Dictionary containing structured tags and metadata
        """
        try:
            # Prepare input for tagging chain
            chain_input = {
                "title": document.metadata.get('title', 'No title'),
                "url": document.metadata.get('url', 'Unknown URL'),
                "content": document.page_content[:2000]  # Limit content for tagging
            }
            
            # Generate tags using LangChain
            tags_response = self.tagging_chain.invoke(chain_input)
            
            # Try to parse JSON response
            try:
                import json
                tags_data = json.loads(tags_response)
            except json.JSONDecodeError:
                # Fallback if response isn't valid JSON
                tags_data = {
                    "primary_tags": self._extract_simple_tags(tags_response),
                    "categories": ["general"],
                    "topics": [],
                    "content_type": "article",
                    "sentiment": "neutral",
                    "complexity": "intermediate",
                    "key_concepts": [],
                    "summary": tags_response[:200] + "..." if len(tags_response) > 200 else tags_response
                }
            
            return tags_data
            
        except Exception as e:
            # Return basic tags if tagging fails
            return {
                "primary_tags": ["web-content"],
                "categories": ["general"],
                "topics": [],
                "content_type": "article",
                "sentiment": "neutral",
                "complexity": "intermediate",
                "key_concepts": [],
                "summary": f"Content from {document.metadata.get('url', 'unknown URL')}",
                "tagging_error": str(e)
            }
    
    def _extract_simple_tags(self, text: str) -> List[str]:
        """Extract simple tags from unstructured text response."""
        # Simple heuristic to extract tags from text
        words = text.lower().split()
        potential_tags = [word.strip('.,!?":;') for word in words if len(word) > 3]
        return potential_tags[:5]  # Return first 5 potential tags
    
    def generate_embeddings(self, document: Document) -> List[float]:
        """
        Generate embeddings for a document using LangChain OpenAI embeddings.
        
        Args:
            document: LangChain Document to embed
            
        Returns:
            List of floats representing the document embedding
        """
        try:
            # Combine title and content for embedding
            text_to_embed = f"{document.metadata.get('title', '')}\n\n{document.page_content}"
            
            # Generate embedding using LangChain
            embedding = self.embeddings.embed_query(text_to_embed)
            
            return embedding
            
        except Exception as e:
            raise Exception(f"Failed to generate embeddings: {str(e)}")
    
    def store_in_knowledge_graph(self, document: Document, tags: Dict[str, Any], embedding: List[float]) -> str:
        """
        Store the processed document in the knowledge graph (Qdrant vector database).
        
        Args:
            document: Original LangChain Document
            tags: Generated tags and metadata
            embedding: Document embedding vector
            
        Returns:
            Unique identifier for the stored document
        """
        try:
            # Generate unique ID for the document
            doc_id = str(uuid.uuid4())
            
            # Prepare payload combining original metadata with generated tags
            payload = {
                # Original document metadata
                **document.metadata,
                # Generated tags and analysis
                "tags": tags,
                "content": document.page_content[:1000],  # Store first 1000 chars
                "content_length": len(document.page_content),
                "processing_timestamp": os.getenv('PROCESSING_TIMESTAMP', ''),
                "doc_id": doc_id
            }
            
            # Create point for Qdrant
            point = PointStruct(
                id=doc_id,
                vector=embedding,
                payload=payload
            )
            
            # Store in Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            return doc_id
            
        except Exception as e:
            raise Exception(f"Failed to store in knowledge graph: {str(e)}")
    
    def process_document(self, document: Document) -> Dict[str, Any]:
        """
        Process a complete document through the full pipeline.
        
        Args:
            document: LangChain Document to process
            
        Returns:
            Dictionary containing processing results and metadata
        """
        try:
            # Step 1: Generate tags
            tags = self.generate_tags(document)
            
            # Step 2: Generate embeddings
            embedding = self.generate_embeddings(document)
            
            # Step 3: Store in knowledge graph
            doc_id = self.store_in_knowledge_graph(document, tags, embedding)
            
            # Return processing results
            return {
                "doc_id": doc_id,
                "url": document.metadata.get('url'),
                "title": document.metadata.get('title'),
                "tags": tags,
                "embedding_dimension": len(embedding),
                "content_length": len(document.page_content),
                "status": "success"
            }
            
        except Exception as e:
            return {
                "doc_id": None,
                "url": document.metadata.get('url'),
                "title": document.metadata.get('title'),
                "status": "error",
                "error": str(e)
            }
    
    def search_similar_documents(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents in the knowledge graph.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            
        Returns:
            List of similar documents with metadata
        """
        try:
            # Generate embedding for query
            query_embedding = self.embeddings.embed_query(query)
            
            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                with_payload=True
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    "doc_id": result.id,
                    "score": result.score,
                    "url": result.payload.get('url'),
                    "title": result.payload.get('title'),
                    "tags": result.payload.get('tags', {}),
                    "content_preview": result.payload.get('content', '')[:200]
                })
            
            return results
            
        except Exception as e:
            raise Exception(f"Failed to search similar documents: {str(e)}")


def create_pipeline() -> LangChainPipeline:
    """
    Factory function to create a LangChainPipeline instance.
    
    Returns:
        Configured LangChainPipeline instance
    """
    return LangChainPipeline()