import asyncio
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from app.services.web_crawler import AdvancedWebCrawler
from app.services.langchain_openai_service import LangChainOpenAIService
from app.services.langchain_qdrant_service import LangChainQdrantService
from app.services.langchain_neptune_service import LangChainNeptuneService
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class LangChainDocumentProcessor:
    """
    Main LangChain-based document processing pipeline that orchestrates
    web crawling, tagging, embedding, vector storage, and knowledge graph creation.
    """
    
    def __init__(self):
        """Initialize all processing services."""
        self.crawler = AdvancedWebCrawler()
        self.openai_service = LangChainOpenAIService()
        self.qdrant_service = LangChainQdrantService()
        self.neptune_service = LangChainNeptuneService()
        
        logger.info("Initialized LangChain Document Processor with all services")

    async def process_url(self, url: str) -> Dict[str, Any]:
        """
        Complete document processing pipeline for a single URL.
        
        Args:
            url: URL to process
            
        Returns:
            Dictionary containing processing results and metadata
        """
        start_time = time.time()
        
        try:
            results = {
                "url": url,
                "status": "processing",
                "timestamp": datetime.utcnow().isoformat(),
                "processing_steps": {},
                "errors": []
            }
            
            # Step 1: Web crawling and content extraction
            logger.info(f"Step 1: Crawling content from {url}")
            step_start = time.time()
            
            documents = self.crawler.extract_smart_content(url)
            if not documents:
                return {
                    **results,
                    "status": "failed",
                    "error": "Failed to extract content from URL"
                }
            
            results["processing_steps"]["crawling"] = {
                "duration": time.time() - step_start,
                "documents_extracted": len(documents),
                "total_content_length": sum(len(doc.page_content) for doc in documents)
            }
            
            # Step 2: Document tagging using OpenAI 3.5 Turbo
            logger.info(f"Step 2: Tagging {len(documents)} documents")
            step_start = time.time()
            
            tags_data = await self.openai_service.tag_documents(documents)
            
            results["processing_steps"]["tagging"] = {
                "duration": time.time() - step_start,
                "documents_tagged": len(tags_data),
                "sample_tags": self._extract_sample_tags(tags_data)
            }
            
            # Step 3: Generate embeddings using OpenAI Large Language Model
            logger.info(f"Step 3: Generating embeddings for {len(documents)} documents")
            step_start = time.time()
            
            embeddings = await self.openai_service.embed_documents(documents)
            
            results["processing_steps"]["embedding"] = {
                "duration": time.time() - step_start,
                "embeddings_generated": len(embeddings),
                "embedding_dimension": len(embeddings[0]) if embeddings else 0
            }
            
            # Step 4: Store in Qdrant vector database
            logger.info(f"Step 4: Storing {len(documents)} documents in Qdrant")
            step_start = time.time()
            
            doc_ids = await self.qdrant_service.store_documents(documents, embeddings, tags_data)
            
            results["processing_steps"]["vector_storage"] = {
                "duration": time.time() - step_start,
                "documents_stored": len(doc_ids),
                "qdrant_ids": doc_ids[:5]  # First 5 IDs as sample
            }
            
            # Step 5: Extract and store knowledge graph in Neptune
            logger.info(f"Step 5: Extracting knowledge graph and storing in Neptune")
            step_start = time.time()
            
            graph_results = await self.neptune_service.extract_and_store_graph(documents, tags_data)
            
            results["processing_steps"]["knowledge_graph"] = {
                "duration": time.time() - step_start,
                **graph_results
            }
            
            # Final results
            results.update({
                "status": "completed",
                "total_duration": time.time() - start_time,
                "summary": {
                    "url": url,
                    "documents_processed": len(documents),
                    "total_content_length": sum(len(doc.page_content) for doc in documents),
                    "unique_tags": self._count_unique_tags(tags_data),
                    "vector_storage_ids": len(doc_ids),
                    "knowledge_graph_nodes": graph_results.get("extracted_nodes", 0),
                    "knowledge_graph_relationships": graph_results.get("extracted_relationships", 0)
                }
            })
            
            logger.info(f"Successfully processed URL {url} in {results['total_duration']:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Failed to process URL {url}: {str(e)}")
            return {
                **results,
                "status": "failed",
                "error": str(e),
                "total_duration": time.time() - start_time
            }

    async def process_multiple_urls(self, urls: List[str], batch_size: int = 5) -> List[Dict[str, Any]]:
        """
        Process multiple URLs in batches.
        
        Args:
            urls: List of URLs to process
            batch_size: Number of URLs to process concurrently
            
        Returns:
            List of processing results for each URL
        """
        results = []
        
        # Process URLs in batches to avoid overwhelming services
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} URLs")
            
            # Process batch concurrently
            batch_tasks = [self.process_url(url) for url in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    results.append({
                        "url": batch[j],
                        "status": "failed",
                        "error": str(result),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                else:
                    results.append(result)
            
            # Small delay between batches
            if i + batch_size < len(urls):
                await asyncio.sleep(2)
        
        return results

    def search_similar_content(
        self, 
        query: str, 
        k: int = 5,
        include_graph_context: bool = False
    ) -> Dict[str, Any]:
        """
        Search for similar content across stored documents.
        
        Args:
            query: Search query
            k: Number of results to return
            include_graph_context: Whether to include knowledge graph context
            
        Returns:
            Dictionary with search results and metadata
        """
        try:
            # Search in vector database
            vector_results = self.qdrant_service.search_with_score(query, k=k)
            
            search_results = {
                "query": query,
                "vector_results": [
                    {
                        "document": {
                            "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                            "metadata": doc.metadata
                        },
                        "similarity_score": float(score)
                    }
                    for doc, score in vector_results
                ],
                "graph_context": []
            }
            
            # Add knowledge graph context if requested
            if include_graph_context and self.neptune_service.enabled:
                # Extract entities from query using OpenAI
                query_tags = self.openai_service.tag_single_document(query)
                
                # Search for related entities in knowledge graph
                for tag in query_tags.get("tags", [])[:3]:  # First 3 tags
                    entities = self.neptune_service.search_entities(
                        properties={"name": tag},
                        limit=5
                    )
                    if entities:
                        search_results["graph_context"].extend(entities)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search content: {str(e)}")
            return {
                "query": query,
                "error": str(e),
                "vector_results": [],
                "graph_context": []
            }

    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about processed content.
        
        Returns:
            Dictionary with statistics from all services
        """
        try:
            stats = {
                "timestamp": datetime.utcnow().isoformat(),
                "qdrant_stats": self.qdrant_service.get_collection_info(),
                "neptune_stats": self.neptune_service.get_graph_statistics(),
                "services_status": {
                    "crawler": "active",
                    "openai": "active",
                    "qdrant": "active" if self.qdrant_service else "inactive",
                    "neptune": "active" if self.neptune_service.enabled else "inactive"
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {str(e)}")
            return {"error": str(e)}

    def _extract_sample_tags(self, tags_data: List[Dict[str, Any]], max_samples: int = 10) -> List[str]:
        """Extract sample tags from tagging results."""
        all_tags = []
        for tag_data in tags_data:
            tagging_result = tag_data.get('tagging_result', {})
            tags = tagging_result.get('tags', [])
            all_tags.extend(tags)
        
        # Return unique tags up to max_samples
        unique_tags = list(set(all_tags))[:max_samples]
        return unique_tags

    def _count_unique_tags(self, tags_data: List[Dict[str, Any]]) -> int:
        """Count unique tags across all documents."""
        all_tags = set()
        for tag_data in tags_data:
            tagging_result = tag_data.get('tagging_result', {})
            tags = tagging_result.get('tags', [])
            all_tags.update(tags)
        
        return len(all_tags)

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all services.
        
        Returns:
            Dictionary with health status of all services
        """
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "services": {}
        }
        
        try:
            # Test web crawler
            try:
                test_docs = self.crawler.crawl_single_page("https://httpbin.org/html")
                health_status["services"]["crawler"] = {
                    "status": "healthy" if test_docs else "unhealthy",
                    "details": f"Extracted {len(test_docs)} documents"
                }
            except Exception as e:
                health_status["services"]["crawler"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
            
            # Test OpenAI service
            try:
                test_embedding = self.openai_service.embed_single_text("test")
                health_status["services"]["openai"] = {
                    "status": "healthy" if len(test_embedding) > 0 else "unhealthy",
                    "details": f"Embedding dimension: {len(test_embedding)}"
                }
            except Exception as e:
                health_status["services"]["openai"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
            
            # Test Qdrant service
            try:
                qdrant_info = self.qdrant_service.get_collection_info()
                health_status["services"]["qdrant"] = {
                    "status": "healthy" if "error" not in qdrant_info else "unhealthy",
                    "details": qdrant_info
                }
            except Exception as e:
                health_status["services"]["qdrant"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
            
            # Test Neptune service
            try:
                if self.neptune_service.enabled:
                    neptune_stats = self.neptune_service.get_graph_statistics()
                    health_status["services"]["neptune"] = {
                        "status": "healthy" if "error" not in neptune_stats else "unhealthy",
                        "details": neptune_stats
                    }
                else:
                    health_status["services"]["neptune"] = {
                        "status": "disabled",
                        "details": "Neptune endpoint not configured"
                    }
            except Exception as e:
                health_status["services"]["neptune"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            health_status["error"] = str(e)
        
        return health_status