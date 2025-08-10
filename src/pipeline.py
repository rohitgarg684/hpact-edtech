"""
Modular LangChain Pipeline for Content Processing

This module provides a comprehensive LangChain-based pipeline for processing
web content, including tagging, embedding generation, and vector storage.
It integrates with OpenAI, Qdrant, and Neptune for end-to-end content analysis.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# LangChain imports
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Vector store and storage
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance

# Local imports
from .web_crawler import WebCrawler, CrawlResult

# Optional Neptune for experiment tracking
try:
    import neptune
    NEPTUNE_AVAILABLE = True
except ImportError:
    NEPTUNE_AVAILABLE = False
    neptune = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for the content processing pipeline."""
    openai_model: str = "gpt-3.5-turbo"
    embedding_model: str = "text-embedding-3-large"
    embedding_dimension: int = 3072
    max_chunk_size: int = 1000
    chunk_overlap: int = 200
    collection_name: str = "content_vectors"
    max_content_length: int = 10000


@dataclass 
class ProcessingResult:
    """Result of the content processing pipeline."""
    url: str
    title: Optional[str]
    content: str
    tags: List[str]
    chunks: List[str]
    embeddings: List[List[float]]
    metadata: Dict[str, Any]
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class ContentTagger:
    """LangChain-based content tagger using OpenAI."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the content tagger.
        
        Args:
            model_name (str): OpenAI model to use for tagging
        """
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0.3,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Enhanced tagging prompt
        self.tagging_prompt = PromptTemplate(
            input_variables=["content", "title"],
            template="""You are an expert content analyst. Analyze the following content and provide relevant tags.

Title: {title}
Content: {content}

Instructions:
- Generate 5-8 highly relevant tags that capture the main topics, themes, and concepts
- Focus on educational content, technology, and subject matter domains
- Use specific, descriptive tags rather than generic ones
- Separate tags with commas
- Include both broad categories and specific topics

Tags:"""
        )
        
        self.tagging_chain = LLMChain(
            llm=self.llm,
            prompt=self.tagging_prompt,
            verbose=False
        )
    
    def generate_tags(self, content: str, title: Optional[str] = None) -> List[str]:
        """
        Generate tags for the given content.
        
        Args:
            content (str): Content to tag
            title (Optional[str]): Title of the content
            
        Returns:
            List[str]: List of generated tags
        """
        try:
            # Truncate content if too long
            truncated_content = content[:2000] + "..." if len(content) > 2000 else content
            
            response = self.tagging_chain.run(
                content=truncated_content,
                title=title or "No title"
            )
            
            # Parse tags from response
            tags = [tag.strip() for tag in response.split(',')]
            tags = [tag for tag in tags if tag and len(tag) > 1]
            
            logger.info(f"Generated {len(tags)} tags: {tags[:3]}...")
            return tags
            
        except Exception as e:
            logger.error(f"Tag generation failed: {str(e)}")
            return ["untagged", "processing_error"]


class ContentSplitter:
    """Text splitter for creating manageable chunks."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the content splitter.
        
        Args:
            chunk_size (int): Maximum size of each chunk
            chunk_overlap (int): Overlap between consecutive chunks
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def split_content(self, content: str) -> List[str]:
        """
        Split content into chunks.
        
        Args:
            content (str): Content to split
            
        Returns:
            List[str]: List of content chunks
        """
        try:
            chunks = self.text_splitter.split_text(content)
            logger.info(f"Split content into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Content splitting failed: {str(e)}")
            return [content]  # Return original content as single chunk


class EmbeddingGenerator:
    """LangChain-based embedding generator."""
    
    def __init__(self, model_name: str = "text-embedding-3-large"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name (str): OpenAI embedding model to use
        """
        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            return []


class VectorStore:
    """Qdrant vector store manager."""
    
    def __init__(self, collection_name: str = "content_vectors", embedding_dimension: int = 3072):
        """
        Initialize the vector store.
        
        Args:
            collection_name (str): Name of the Qdrant collection
            embedding_dimension (int): Dimension of the embedding vectors
        """
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        
        # Initialize Qdrant client
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        
        try:
            self.client = QdrantClient(host=host, port=port)
            logger.info(f"Connected to Qdrant at {host}:{port}")
        except Exception:
            logger.warning("Using in-memory Qdrant client")
            self.client = QdrantClient(":memory:")
        
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Ensure the collection exists."""
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in collections:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Failed to ensure collection: {str(e)}")
    
    def store_vectors(self, url: str, chunks: List[str], embeddings: List[List[float]], 
                     metadata: Dict[str, Any]) -> bool:
        """
        Store vectors in Qdrant.
        
        Args:
            url (str): Source URL
            chunks (List[str]): Text chunks
            embeddings (List[List[float]]): Embedding vectors
            metadata (Dict[str, Any]): Additional metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point = PointStruct(
                    id=None,  # Let Qdrant generate IDs
                    vector=embedding,
                    payload={
                        "url": url,
                        "chunk_index": i,
                        "chunk_text": chunk,
                        "metadata": metadata,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                points.append(point)
            
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"Stored {len(points)} vectors for URL: {url}")
            return True
            
        except Exception as e:
            logger.error(f"Vector storage failed: {str(e)}")
            return False


class ExperimentTracker:
    """Neptune experiment tracking (optional)."""
    
    def __init__(self, project_name: Optional[str] = None):
        """
        Initialize experiment tracker.
        
        Args:
            project_name (Optional[str]): Neptune project name
        """
        self.run = None
        if NEPTUNE_AVAILABLE and project_name:
            try:
                self.run = neptune.init_run(project=project_name)
                logger.info("Neptune experiment tracking initialized")
            except Exception as e:
                logger.warning(f"Neptune initialization failed: {str(e)}")
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics to Neptune."""
        if self.run:
            try:
                for key, value in metrics.items():
                    self.run[key] = value
            except Exception as e:
                logger.warning(f"Failed to log metrics: {str(e)}")
    
    def stop(self):
        """Stop the Neptune run."""
        if self.run:
            try:
                self.run.stop()
            except Exception as e:
                logger.warning(f"Failed to stop Neptune run: {str(e)}")


class ContentProcessingPipeline:
    """
    Main pipeline for processing web content with LangChain components.
    
    This pipeline combines web crawling, content tagging, text splitting,
    embedding generation, and vector storage into a unified workflow.
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None, 
                 neptune_project: Optional[str] = None):
        """
        Initialize the content processing pipeline.
        
        Args:
            config (Optional[ProcessingConfig]): Pipeline configuration
            neptune_project (Optional[str]): Neptune project for experiment tracking
        """
        self.config = config or ProcessingConfig()
        
        # Initialize components
        self.web_crawler = WebCrawler(
            max_content_length=self.config.max_content_length
        )
        self.tagger = ContentTagger(model_name=self.config.openai_model)
        self.splitter = ContentSplitter(
            chunk_size=self.config.max_chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        self.embedding_generator = EmbeddingGenerator(model_name=self.config.embedding_model)
        self.vector_store = VectorStore(
            collection_name=self.config.collection_name,
            embedding_dimension=self.config.embedding_dimension
        )
        self.experiment_tracker = ExperimentTracker(project_name=neptune_project)
        
        logger.info("Content processing pipeline initialized")
    
    def process_url(self, url: str) -> ProcessingResult:
        """
        Process a single URL through the complete pipeline.
        
        Args:
            url (str): URL to process
            
        Returns:
            ProcessingResult: Result of the processing pipeline
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Crawl the URL
            crawl_result = self.web_crawler.crawl(url)
            if not crawl_result.success or not crawl_result.content:
                return ProcessingResult(
                    url=url,
                    title=crawl_result.title,
                    content=crawl_result.content,
                    tags=[],
                    chunks=[],
                    embeddings=[],
                    metadata=crawl_result.metadata,
                    processing_time=0.0,
                    success=False,
                    error_message=crawl_result.error_message or "Failed to crawl content"
                )
            
            # Step 2: Generate tags
            tags = self.tagger.generate_tags(crawl_result.content, crawl_result.title)
            
            # Step 3: Split content into chunks
            chunks = self.splitter.split_content(crawl_result.content)
            
            # Step 4: Generate embeddings
            embeddings = self.embedding_generator.generate_embeddings(chunks)
            
            # Step 5: Store vectors
            metadata = {
                **crawl_result.metadata,
                "tags": tags,
                "title": crawl_result.title,
                "processing_config": asdict(self.config)
            }
            
            storage_success = self.vector_store.store_vectors(
                url, chunks, embeddings, metadata
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Log metrics
            metrics = {
                "processing_time": processing_time,
                "content_length": len(crawl_result.content),
                "num_chunks": len(chunks),
                "num_tags": len(tags),
                "storage_success": storage_success
            }
            self.experiment_tracker.log_metrics(metrics)
            
            return ProcessingResult(
                url=url,
                title=crawl_result.title,
                content=crawl_result.content,
                tags=tags,
                chunks=chunks,
                embeddings=embeddings,
                metadata=metadata,
                processing_time=processing_time,
                success=storage_success
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Pipeline processing failed for {url}: {str(e)}")
            
            return ProcessingResult(
                url=url,
                title=None,
                content="",
                tags=[],
                chunks=[],
                embeddings=[],
                metadata={},
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def process_urls(self, urls: List[str]) -> List[ProcessingResult]:
        """
        Process multiple URLs through the pipeline.
        
        Args:
            urls (List[str]): List of URLs to process
            
        Returns:
            List[ProcessingResult]: List of processing results
        """
        results = []
        for url in urls:
            result = self.process_url(url)
            results.append(result)
        return results
    
    def close(self):
        """Clean up resources."""
        self.web_crawler.close()
        self.experiment_tracker.stop()


# Factory functions for easy instantiation
def create_pipeline(config: Optional[ProcessingConfig] = None,
                   neptune_project: Optional[str] = None) -> ContentProcessingPipeline:
    """
    Factory function to create a ContentProcessingPipeline.
    
    Args:
        config (Optional[ProcessingConfig]): Pipeline configuration
        neptune_project (Optional[str]): Neptune project name
        
    Returns:
        ContentProcessingPipeline: Configured pipeline instance
    """
    return ContentProcessingPipeline(config=config, neptune_project=neptune_project)


def create_config(**kwargs) -> ProcessingConfig:
    """
    Factory function to create a ProcessingConfig with overrides.
    
    Args:
        **kwargs: Configuration overrides
        
    Returns:
        ProcessingConfig: Configuration instance
    """
    return ProcessingConfig(**kwargs)