"""
Web Crawler Module for Enhanced Content Extraction

This module provides robust web crawling capabilities using trafilatura for
high-quality content extraction and BeautifulSoup as a fallback for edge cases.
It handles various URL formats and provides clean text extraction optimized
for NLP processing.
"""

import requests
import trafilatura
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import Optional, Dict, Any
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CrawlResult:
    """Result container for web crawling operations."""
    url: str
    title: Optional[str]
    content: str
    metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class WebCrawler:
    """
    Enhanced web crawler using trafilatura for content extraction.
    
    This class provides robust web crawling capabilities with intelligent
    fallback mechanisms and proper error handling.
    """
    
    def __init__(self, max_content_length: int = 10000, timeout: int = 30):
        """
        Initialize the WebCrawler.
        
        Args:
            max_content_length (int): Maximum length of content to extract
            timeout (int): Request timeout in seconds
        """
        self.max_content_length = max_content_length
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; EdTech-Crawler/1.0)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
    
    def is_valid_url(self, url: str) -> bool:
        """
        Validate if the URL is properly formatted and accessible.
        
        Args:
            url (str): URL to validate
            
        Returns:
            bool: True if URL is valid, False otherwise
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def fetch_with_trafilatura(self, url: str) -> Optional[CrawlResult]:
        """
        Extract content using trafilatura (primary method).
        
        Args:
            url (str): URL to crawl
            
        Returns:
            Optional[CrawlResult]: Extracted content or None if failed
        """
        try:
            # Download the webpage
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                return None
            
            # Extract content with trafilatura
            content = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=True,
                include_images=False,
                include_links=False
            )
            
            if not content:
                return None
            
            # Extract metadata
            metadata = trafilatura.extract_metadata(downloaded)
            title = metadata.title if metadata else None
            
            # Truncate content if too long
            if len(content) > self.max_content_length:
                content = content[:self.max_content_length] + "..."
            
            return CrawlResult(
                url=url,
                title=title,
                content=content.strip(),
                metadata={
                    'author': metadata.author if metadata else None,
                    'date': str(metadata.date) if metadata and metadata.date else None,
                    'language': metadata.language if metadata else None,
                    'extraction_method': 'trafilatura'
                },
                success=True
            )
            
        except Exception as e:
            logger.warning(f"Trafilatura extraction failed for {url}: {str(e)}")
            return None
    
    def fetch_with_beautifulsoup(self, url: str) -> Optional[CrawlResult]:
        """
        Extract content using BeautifulSoup (fallback method).
        
        Args:
            url (str): URL to crawl
            
        Returns:
            Optional[CrawlResult]: Extracted content or None if failed
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else None
            
            # Extract main content
            content_selectors = [
                'main', 'article', '.content', '.post-content', 
                '.entry-content', '.article-content', '#content'
            ]
            
            content = ""
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content = content_elem.get_text(separator=' ', strip=True)
                    break
            
            # Fallback to body if no specific content found
            if not content:
                body = soup.find('body')
                if body:
                    content = body.get_text(separator=' ', strip=True)
            
            # Clean and truncate content
            content = ' '.join(content.split())
            if len(content) > self.max_content_length:
                content = content[:self.max_content_length] + "..."
            
            return CrawlResult(
                url=url,
                title=title,
                content=content,
                metadata={'extraction_method': 'beautifulsoup'},
                success=True
            )
            
        except Exception as e:
            logger.warning(f"BeautifulSoup extraction failed for {url}: {str(e)}")
            return None
    
    def crawl(self, url: str) -> CrawlResult:
        """
        Crawl a URL and extract its content using the best available method.
        
        This method tries trafilatura first (for better content extraction),
        and falls back to BeautifulSoup if needed.
        
        Args:
            url (str): URL to crawl
            
        Returns:
            CrawlResult: Result of the crawling operation
        """
        if not self.is_valid_url(url):
            return CrawlResult(
                url=url,
                title=None,
                content="",
                metadata={},
                success=False,
                error_message="Invalid URL format"
            )
        
        logger.info(f"Crawling URL: {url}")
        
        # Try trafilatura first
        result = self.fetch_with_trafilatura(url)
        if result and result.content:
            logger.info(f"Successfully extracted content using trafilatura: {len(result.content)} chars")
            return result
        
        # Fall back to BeautifulSoup
        logger.info("Falling back to BeautifulSoup extraction")
        result = self.fetch_with_beautifulsoup(url)
        if result and result.content:
            logger.info(f"Successfully extracted content using BeautifulSoup: {len(result.content)} chars")
            return result
        
        # Both methods failed
        return CrawlResult(
            url=url,
            title=None,
            content="",
            metadata={},
            success=False,
            error_message="Failed to extract content with all available methods"
        )
    
    def crawl_multiple(self, urls: list[str]) -> list[CrawlResult]:
        """
        Crawl multiple URLs and return results.
        
        Args:
            urls (list[str]): List of URLs to crawl
            
        Returns:
            list[CrawlResult]: List of crawling results
        """
        results = []
        for url in urls:
            result = self.crawl(url)
            results.append(result)
        return results
    
    def close(self):
        """Close the requests session."""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Factory function for backward compatibility and easy instantiation
def create_web_crawler(max_content_length: int = 10000, timeout: int = 30) -> WebCrawler:
    """
    Factory function to create a WebCrawler instance.
    
    Args:
        max_content_length (int): Maximum content length to extract
        timeout (int): Request timeout in seconds
        
    Returns:
        WebCrawler: Configured WebCrawler instance
    """
    return WebCrawler(max_content_length=max_content_length, timeout=timeout)