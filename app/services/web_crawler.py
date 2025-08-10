import asyncio
from typing import List, Optional
from urllib.parse import urljoin, urlparse
from langchain_community.document_loaders import WebBaseLoader, RecursiveUrlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import requests
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)


class AdvancedWebCrawler:
    """
    State-of-the-art web crawler using LangChain document loaders
    with intelligent content extraction and processing capabilities.
    """
    
    def __init__(
        self,
        max_depth: int = 2,
        max_pages: int = 50,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.html2text = Html2TextTransformer()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Headers to mimic real browser requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }

    def crawl_single_page(self, url: str) -> List[Document]:
        """
        Crawl a single webpage and extract structured content.
        
        Args:
            url: URL of the webpage to crawl
            
        Returns:
            List of Document objects with extracted content
        """
        try:
            # Use WebBaseLoader with custom headers for better content extraction
            loader = WebBaseLoader(
                web_paths=[url],
                header_template={'User-Agent': self.headers['User-Agent']},
                verify_ssl=True
            )
            
            # Load documents
            documents = loader.load()
            
            # Transform HTML to clean text
            documents = self.html2text.transform_documents(documents)
            
            # Split into chunks if content is large
            if documents and len(documents[0].page_content) > self.chunk_size:
                documents = self.text_splitter.split_documents(documents)
            
            # Enhance metadata
            for doc in documents:
                doc.metadata.update({
                    'source_url': url,
                    'crawler_type': 'single_page',
                    'content_length': len(doc.page_content)
                })
                
            logger.info(f"Successfully crawled {url}, extracted {len(documents)} document chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to crawl {url}: {str(e)}")
            return []

    def crawl_website_recursive(
        self, 
        base_url: str, 
        url_filter: Optional[callable] = None
    ) -> List[Document]:
        """
        Recursively crawl a website following links up to max_depth.
        
        Args:
            base_url: Starting URL for the crawl
            url_filter: Optional function to filter URLs (return True to include)
            
        Returns:
            List of Document objects from all crawled pages
        """
        try:
            # Default filter to stay within same domain
            if url_filter is None:
                parsed_base = urlparse(base_url)
                def url_filter(url: str) -> bool:
                    parsed_url = urlparse(url)
                    return parsed_url.netloc == parsed_base.netloc

            # Use RecursiveUrlLoader for intelligent crawling
            loader = RecursiveUrlLoader(
                url=base_url,
                max_depth=self.max_depth,
                extractor=lambda html: BeautifulSoup(html, "html.parser").get_text(),
                metadata_extractor=lambda meta, soup: {
                    **meta,
                    'title': soup.find('title').get_text() if soup.find('title') else '',
                    'description': soup.find('meta', attrs={'name': 'description'})['content'] 
                              if soup.find('meta', attrs={'name': 'description'}) else ''
                },
                use_async=True,
                prevent_outside=True,
                link_regex=r"href=[\"']([^\"']+)[\"']",
            )
            
            # Load documents
            documents = loader.load()
            
            # Limit number of documents
            if len(documents) > self.max_pages:
                documents = documents[:self.max_pages]
            
            # Transform and split documents
            processed_docs = []
            for doc in documents:
                # Clean and structure content
                if len(doc.page_content.strip()) < 100:  # Skip very short content
                    continue
                    
                # Split large documents
                if len(doc.page_content) > self.chunk_size:
                    chunks = self.text_splitter.split_documents([doc])
                    processed_docs.extend(chunks)
                else:
                    processed_docs.append(doc)
                    
                # Enhanced metadata
                doc.metadata.update({
                    'crawler_type': 'recursive',
                    'base_url': base_url,
                    'content_length': len(doc.page_content)
                })
            
            logger.info(f"Successfully crawled {len(processed_docs)} pages from {base_url}")
            return processed_docs
            
        except Exception as e:
            logger.error(f"Failed to recursively crawl {base_url}: {str(e)}")
            # Fallback to single page crawl
            return self.crawl_single_page(base_url)

    def extract_smart_content(self, url: str) -> List[Document]:
        """
        Intelligently extract content from a URL using the best strategy.
        
        Args:
            url: URL to process
            
        Returns:
            List of Document objects with extracted content
        """
        try:
            # First try to determine if this is a single page or needs recursive crawling
            response = requests.head(url, headers=self.headers, timeout=10)
            content_type = response.headers.get('content-type', '').lower()
            
            if 'text/html' not in content_type:
                logger.warning(f"URL {url} doesn't appear to be HTML content")
                return []
            
            # For now, default to single page extraction for better control
            # Can be enhanced to detect if recursive crawling is beneficial
            documents = self.crawl_single_page(url)
            
            # Enhance with additional metadata extraction
            for doc in documents:
                doc.metadata.update({
                    'extraction_method': 'smart_content',
                    'timestamp': str(asyncio.get_event_loop().time() if hasattr(asyncio, '_get_running_loop') and asyncio._get_running_loop() else 'unknown')
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Smart content extraction failed for {url}: {str(e)}")
            return []

    def get_page_summary(self, documents: List[Document]) -> dict:
        """
        Generate a summary of extracted content.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary with summary information
        """
        if not documents:
            return {'total_docs': 0, 'total_content_length': 0}
            
        total_length = sum(len(doc.page_content) for doc in documents)
        sources = list(set(doc.metadata.get('source_url', 'unknown') for doc in documents))
        
        return {
            'total_docs': len(documents),
            'total_content_length': total_length,
            'unique_sources': len(sources),
            'sources': sources[:5],  # First 5 sources
            'avg_doc_length': total_length // len(documents) if documents else 0
        }