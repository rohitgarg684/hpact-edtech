import os
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import logging

logger = logging.getLogger(__name__)


class LangChainOpenAIService:
    """
    LangChain-based OpenAI service for document tagging and embedding
    using OpenAI 3.5 Turbo for tagging and Large Language Model for embeddings.
    """
    
    def __init__(self):
        """Initialize OpenAI services through LangChain."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize ChatOpenAI for tagging with GPT-3.5 Turbo
        self.chat_model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.3,
            openai_api_key=api_key
        )
        
        # Initialize OpenAI Embeddings with text-embedding-3-large
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=api_key
        )
        
        # Create tagging prompt template
        self.tagging_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an intelligent content analyzer that reads documents and provides relevant tags and categories. 
            
            Your task is to analyze the given content and provide:
            1. 5-8 relevant topical tags
            2. 2-3 category classifications
            3. Content type identification
            4. Key themes or subjects
            
            Format your response as a structured JSON with the following keys:
            - tags: list of specific topical tags
            - categories: list of broad categories
            - content_type: single content type (e.g., "article", "documentation", "blog_post", "news")
            - themes: list of main themes or subjects
            - summary: brief one-sentence summary
            
            Be concise and accurate. Focus on the most relevant and distinctive aspects of the content."""),
            
            HumanMessage(content="Analyze this content and provide structured tags and categorization:\n\n{content}")
        ])
        
        # Create tagging chain
        self.tagging_chain = self.tagging_prompt | self.chat_model

    async def tag_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Tag multiple documents using OpenAI 3.5 Turbo via LangChain.
        
        Args:
            documents: List of Document objects to tag
            
        Returns:
            List of dictionaries containing tags and metadata for each document
        """
        tagged_docs = []
        
        for i, doc in enumerate(documents):
            try:
                # Truncate content if too long (keep within token limits)
                content = doc.page_content[:4000] if len(doc.page_content) > 4000 else doc.page_content
                
                # Generate tags using the chain
                response = await self.tagging_chain.ainvoke({"content": content})
                
                # Parse response (assuming it's JSON-like)
                try:
                    import json
                    tags_data = json.loads(response.content)
                except json.JSONDecodeError:
                    # Fallback parsing if not perfect JSON
                    tags_data = {
                        "tags": self._extract_tags_from_text(response.content),
                        "categories": ["general"],
                        "content_type": "document",
                        "themes": ["content"],
                        "summary": response.content[:100] + "..." if len(response.content) > 100 else response.content
                    }
                
                # Combine with original document metadata
                enhanced_doc = {
                    'document_index': i,
                    'original_metadata': doc.metadata,
                    'content': content,
                    'tagging_result': tags_data,
                    'source_url': doc.metadata.get('source_url', doc.metadata.get('source', 'unknown'))
                }
                
                tagged_docs.append(enhanced_doc)
                logger.info(f"Successfully tagged document {i+1}/{len(documents)}")
                
            except Exception as e:
                logger.error(f"Failed to tag document {i}: {str(e)}")
                # Add fallback tags
                tagged_docs.append({
                    'document_index': i,
                    'original_metadata': doc.metadata,
                    'content': doc.page_content[:4000],
                    'tagging_result': {
                        "tags": ["content", "document"],
                        "categories": ["general"],
                        "content_type": "document",
                        "themes": ["general_content"],
                        "summary": "Document content analysis failed"
                    },
                    'source_url': doc.metadata.get('source_url', doc.metadata.get('source', 'unknown')),
                    'tagging_error': str(e)
                })
        
        return tagged_docs

    async def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """
        Generate embeddings for documents using OpenAI Large Language Model via LangChain.
        
        Args:
            documents: List of Document objects to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Extract text content from documents
            texts = [doc.page_content for doc in documents]
            
            # Generate embeddings using LangChain OpenAIEmbeddings
            embeddings = await self.embeddings.aembed_documents(texts)
            
            logger.info(f"Successfully generated embeddings for {len(documents)} documents")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            # Return zero embeddings as fallback
            return [[0.0] * 3072 for _ in documents]  # 3072 is the dimension for text-embedding-3-large

    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query text.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector
        """
        try:
            embedding = await self.embeddings.aembed_query(query)
            return embedding
        except Exception as e:
            logger.error(f"Failed to embed query: {str(e)}")
            return [0.0] * 3072

    def tag_single_document(self, content: str) -> Dict[str, Any]:
        """
        Synchronous version of document tagging for single content.
        
        Args:
            content: Text content to tag
            
        Returns:
            Dictionary containing tags and metadata
        """
        try:
            # Truncate content if too long
            content = content[:4000] if len(content) > 4000 else content
            
            # Generate tags using the chain (sync version)
            response = self.tagging_chain.invoke({"content": content})
            
            # Parse response
            try:
                import json
                tags_data = json.loads(response.content)
            except json.JSONDecodeError:
                tags_data = {
                    "tags": self._extract_tags_from_text(response.content),
                    "categories": ["general"],
                    "content_type": "document", 
                    "themes": ["content"],
                    "summary": response.content[:100] + "..." if len(response.content) > 100 else response.content
                }
            
            return tags_data
            
        except Exception as e:
            logger.error(f"Failed to tag content: {str(e)}")
            return {
                "tags": ["content"],
                "categories": ["general"],
                "content_type": "document",
                "themes": ["general_content"],
                "summary": "Content analysis failed",
                "error": str(e)
            }

    def embed_single_text(self, text: str) -> List[float]:
        """
        Synchronous version of text embedding.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Failed to embed text: {str(e)}")
            return [0.0] * 3072

    def _extract_tags_from_text(self, text: str) -> List[str]:
        """
        Fallback method to extract tags from unstructured text response.
        
        Args:
            text: Response text from the model
            
        Returns:
            List of extracted tags
        """
        # Simple extraction logic - look for common patterns
        import re
        
        # Look for lists or comma-separated items
        tag_patterns = [
            r'tags?[:\s]*([^\n]+)',
            r'keywords?[:\s]*([^\n]+)',
            r'categories?[:\s]*([^\n]+)'
        ]
        
        tags = []
        for pattern in tag_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Split by common delimiters
                items = re.split(r'[,;|]+', match)
                tags.extend([item.strip().strip('"\'') for item in items if item.strip()])
        
        # Clean and deduplicate
        tags = list(set([tag.lower() for tag in tags if len(tag.strip()) > 2]))[:10]
        
        return tags if tags else ["content", "document"]