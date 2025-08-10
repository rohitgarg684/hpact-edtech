"""
Sophisticated web crawler using trafilatura and BeautifulSoup to parse website URLs
and generate LangChain Documents.
"""

import os
import requests
from typing import Optional
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import trafilatura
from langchain_core.documents import Document


class WebCrawler:
    """
    A sophisticated web crawler that extracts content from web pages using
    trafilatura and BeautifulSoup, then creates LangChain Document objects.
    """
    
    def __init__(self, timeout: int = 10, max_content_length: int = 50000):
        """
        Initialize the web crawler.
        
        Args:
            timeout: Request timeout in seconds
            max_content_length: Maximum content length to extract
        """
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_url(self, url: str) -> str:
        """
        Fetch raw HTML content from a URL.
        
        Args:
            url: The URL to fetch
            
        Returns:
            Raw HTML content as string
            
        Raises:
            requests.RequestException: If URL fetching fails
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            raise requests.RequestException(f"Failed to fetch URL {url}: {str(e)}")
    
    def extract_with_trafilatura(self, html: str, url: str) -> Optional[str]:
        """
        Extract main text content using trafilatura.
        
        Args:
            html: Raw HTML content
            url: Source URL for context
            
        Returns:
            Extracted text content or None if extraction fails
        """
        try:
            # Use trafilatura to extract main content
            extracted = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                include_formatting=False,
                output_format='txt'
            )
            return extracted
        except Exception:
            # Fallback to None if trafilatura fails
            return None
    
    def extract_with_beautifulsoup(self, html: str) -> str:
        """
        Extract text content using BeautifulSoup as fallback.
        
        Args:
            html: Raw HTML content
            
        Returns:
            Extracted text content
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()
            
            # Extract text from main content areas
            main_content = soup.find(['main', 'article', 'div'])
            if main_content:
                text = main_content.get_text(separator=' ', strip=True)
            else:
                text = soup.get_text(separator=' ', strip=True)
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            raise ValueError(f"Failed to extract content with BeautifulSoup: {str(e)}")
    
    def extract_metadata(self, html: str, url: str) -> dict:
        """
        Extract metadata from HTML content.
        
        Args:
            html: Raw HTML content
            url: Source URL
            
        Returns:
            Dictionary containing metadata
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            metadata = {
                'url': url,
                'domain': urlparse(url).netloc,
            }
            
            # Extract title
            title_tag = soup.find('title')
            if title_tag:
                metadata['title'] = title_tag.get_text().strip()
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                metadata['description'] = meta_desc.get('content', '').strip()
            
            # Extract meta keywords
            meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
            if meta_keywords:
                metadata['keywords'] = meta_keywords.get('content', '').strip()
            
            # Extract author
            meta_author = soup.find('meta', attrs={'name': 'author'})
            if meta_author:
                metadata['author'] = meta_author.get('content', '').strip()
            
            # Extract Open Graph data
            og_title = soup.find('meta', property='og:title')
            if og_title:
                metadata['og_title'] = og_title.get('content', '').strip()
            
            og_description = soup.find('meta', property='og:description')
            if og_description:
                metadata['og_description'] = og_description.get('content', '').strip()
            
            return metadata
        except Exception:
            # Return minimal metadata if extraction fails
            return {
                'url': url,
                'domain': urlparse(url).netloc,
            }
    
    def crawl_url(self, url: str) -> Document:
        """
        Crawl a URL and return a LangChain Document.
        
        Args:
            url: The URL to crawl
            
        Returns:
            LangChain Document with extracted content and metadata
            
        Raises:
            Exception: If crawling fails completely
        """
        try:
            # Fetch HTML content
            html = self.fetch_url(url)
            
            # Try trafilatura first (more sophisticated extraction)
            content = self.extract_with_trafilatura(html, url)
            
            # Fallback to BeautifulSoup if trafilatura fails
            if not content or len(content.strip()) < 100:
                content = self.extract_with_beautifulsoup(html)
            
            # Truncate content if too long
            if len(content) > self.max_content_length:
                content = content[:self.max_content_length] + "..."
            
            # Extract metadata
            metadata = self.extract_metadata(html, url)
            
            # Add content length to metadata
            metadata['content_length'] = len(content)
            metadata['extraction_method'] = 'trafilatura' if self.extract_with_trafilatura(html, url) else 'beautifulsoup'
            
            # Create LangChain Document
            document = Document(
                page_content=content,
                metadata=metadata
            )
            
            return document
            
        except Exception as e:
            raise Exception(f"Failed to crawl URL {url}: {str(e)}")


def create_web_crawler() -> WebCrawler:
    """
    Factory function to create a WebCrawler instance with default configuration.
    
    Returns:
        Configured WebCrawler instance
    """
    timeout = int(os.getenv('WEB_CRAWLER_TIMEOUT', '10'))
    max_content_length = int(os.getenv('WEB_CRAWLER_MAX_CONTENT_LENGTH', '50000'))
    
    return WebCrawler(timeout=timeout, max_content_length=max_content_length)