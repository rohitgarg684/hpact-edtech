"""
Unit tests for the web crawler module.
Tests web crawling functionality with mocked HTTP requests and extraction.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
from langchain_core.documents import Document

from src.web_crawler import WebCrawler, create_web_crawler


class TestWebCrawler:
    """Test class for WebCrawler functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.crawler = WebCrawler(timeout=5, max_content_length=1000)
        
        # Sample HTML for testing
        self.sample_html = """
        <html>
        <head>
            <title>Test Article</title>
            <meta name="description" content="A test article for unit testing">
            <meta name="keywords" content="test, article, unittest">
            <meta name="author" content="Test Author">
            <meta property="og:title" content="OG Test Article">
            <meta property="og:description" content="OG test description">
        </head>
        <body>
            <header>Header content</header>
            <nav>Navigation</nav>
            <main>
                <article>
                    <h1>Test Article Title</h1>
                    <p>This is the main content of the test article.</p>
                    <p>It contains multiple paragraphs with useful information.</p>
                    <script>console.log("should be removed")</script>
                    <style>body { color: red; }</style>
                </article>
            </main>
            <footer>Footer content</footer>
        </body>
        </html>
        """
        
        # Expected extracted content (without header, nav, footer, script, style)
        self.expected_content = "Test Article Title This is the main content of the test article. It contains multiple paragraphs with useful information."
    
    @patch('requests.Session.get')
    def test_fetch_url_success(self, mock_get):
        """Test successful URL fetching."""
        mock_response = Mock()
        mock_response.text = self.sample_html
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = self.crawler.fetch_url("https://example.com")
        
        assert result == self.sample_html
        mock_get.assert_called_once_with("https://example.com", timeout=5)
        mock_response.raise_for_status.assert_called_once()
    
    @patch('requests.Session.get')
    def test_fetch_url_failure(self, mock_get):
        """Test URL fetching failure."""
        mock_get.side_effect = requests.RequestException("Network error")
        
        with pytest.raises(requests.RequestException, match="Failed to fetch URL"):
            self.crawler.fetch_url("https://example.com")
    
    @patch('requests.Session.get')
    def test_fetch_url_http_error(self, mock_get):
        """Test URL fetching with HTTP error."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response
        
        with pytest.raises(requests.RequestException):
            self.crawler.fetch_url("https://example.com/notfound")
    
    def test_extract_with_beautifulsoup(self):
        """Test content extraction using BeautifulSoup."""
        result = self.crawler.extract_with_beautifulsoup(self.sample_html)
        
        # Should contain main content but not header, nav, footer, script, or style
        assert "Test Article Title" in result
        assert "main content of the test article" in result
        assert "Header content" not in result
        assert "Navigation" not in result
        assert "Footer content" not in result
        assert "console.log" not in result
        assert "color: red" not in result
    
    def test_extract_with_beautifulsoup_invalid_html(self):
        """Test BeautifulSoup extraction with invalid HTML."""
        invalid_html = "<html><body><p>Unclosed paragraph"
        
        result = self.crawler.extract_with_beautifulsoup(invalid_html)
        assert "Unclosed paragraph" in result
    
    @patch('trafilatura.extract')
    def test_extract_with_trafilatura_success(self, mock_extract):
        """Test successful content extraction with trafilatura."""
        mock_extract.return_value = "Extracted content from trafilatura"
        
        result = self.crawler.extract_with_trafilatura(self.sample_html, "https://example.com")
        
        assert result == "Extracted content from trafilatura"
        mock_extract.assert_called_once()
    
    @patch('trafilatura.extract')
    def test_extract_with_trafilatura_failure(self, mock_extract):
        """Test trafilatura extraction failure."""
        mock_extract.side_effect = Exception("Trafilatura error")
        
        result = self.crawler.extract_with_trafilatura(self.sample_html, "https://example.com")
        
        assert result is None
    
    def test_extract_metadata(self):
        """Test metadata extraction from HTML."""
        url = "https://example.com/article"
        metadata = self.crawler.extract_metadata(self.sample_html, url)
        
        assert metadata['url'] == url
        assert metadata['domain'] == 'example.com'
        assert metadata['title'] == 'Test Article'
        assert metadata['description'] == 'A test article for unit testing'
        assert metadata['keywords'] == 'test, article, unittest'
        assert metadata['author'] == 'Test Author'
        assert metadata['og_title'] == 'OG Test Article'
        assert metadata['og_description'] == 'OG test description'
    
    def test_extract_metadata_minimal_html(self):
        """Test metadata extraction from minimal HTML."""
        minimal_html = "<html><head><title>Simple Title</title></head><body>Content</body></html>"
        url = "https://example.com"
        
        metadata = self.crawler.extract_metadata(minimal_html, url)
        
        assert metadata['url'] == url
        assert metadata['domain'] == 'example.com'
        assert metadata['title'] == 'Simple Title'
        assert 'description' not in metadata or metadata['description'] == ''
    
    @patch('src.web_crawler.WebCrawler.fetch_url')
    @patch('trafilatura.extract')
    def test_crawl_url_trafilatura_success(self, mock_traf_extract, mock_fetch):
        """Test successful URL crawling with trafilatura."""
        mock_fetch.return_value = self.sample_html
        mock_traf_extract.return_value = "Trafilatura extracted content"
        
        url = "https://example.com/article"
        document = self.crawler.crawl_url(url)
        
        assert isinstance(document, Document)
        assert document.page_content == "Trafilatura extracted content"
        assert document.metadata['url'] == url
        assert document.metadata['domain'] == 'example.com'
        assert document.metadata['extraction_method'] == 'trafilatura'
        
        mock_fetch.assert_called_once_with(url)
        mock_traf_extract.assert_called()
    
    @patch('src.web_crawler.WebCrawler.fetch_url')
    @patch('trafilatura.extract')
    def test_crawl_url_fallback_to_beautifulsoup(self, mock_traf_extract, mock_fetch):
        """Test URL crawling fallback to BeautifulSoup when trafilatura fails."""
        mock_fetch.return_value = self.sample_html
        mock_traf_extract.return_value = None  # Trafilatura fails
        
        url = "https://example.com/article"
        document = self.crawler.crawl_url(url)
        
        assert isinstance(document, Document)
        assert "Test Article Title" in document.page_content
        assert document.metadata['extraction_method'] == 'beautifulsoup'
        
        mock_traf_extract.assert_called()
    
    @patch('src.web_crawler.WebCrawler.fetch_url')
    @patch('trafilatura.extract')
    def test_crawl_url_short_trafilatura_content(self, mock_traf_extract, mock_fetch):
        """Test URL crawling fallback when trafilatura returns short content."""
        mock_fetch.return_value = self.sample_html
        mock_traf_extract.return_value = "Short"  # Less than 100 characters
        
        document = self.crawler.crawl_url("https://example.com")
        
        # Should fallback to BeautifulSoup and contain expected content
        assert "Test Article Title" in document.page_content
        assert document.metadata['extraction_method'] == 'beautifulsoup'
    
    @patch('src.web_crawler.WebCrawler.fetch_url')
    def test_crawl_url_content_truncation(self, mock_fetch):
        """Test content truncation for long content."""
        # Create content longer than max_content_length (1000)
        long_content = "A" * 1500
        long_html = f"<html><body><p>{long_content}</p></body></html>"
        
        mock_fetch.return_value = long_html
        
        document = self.crawler.crawl_url("https://example.com")
        
        # Content should be truncated to max_content_length + "..."
        assert len(document.page_content) == 1003  # 1000 + "..."
        assert document.page_content.endswith("...")
    
    @patch('src.web_crawler.WebCrawler.fetch_url')
    def test_crawl_url_fetch_failure(self, mock_fetch):
        """Test URL crawling when fetch fails."""
        mock_fetch.side_effect = requests.RequestException("Network error")
        
        with pytest.raises(Exception, match="Failed to crawl URL"):
            self.crawler.crawl_url("https://example.com")
    
    def test_create_web_crawler_factory(self):
        """Test the factory function for creating WebCrawler."""
        with patch.dict('os.environ', {
            'WEB_CRAWLER_TIMEOUT': '15',
            'WEB_CRAWLER_MAX_CONTENT_LENGTH': '2000'
        }):
            crawler = create_web_crawler()
            
            assert isinstance(crawler, WebCrawler)
            assert crawler.timeout == 15
            assert crawler.max_content_length == 2000
    
    def test_create_web_crawler_factory_defaults(self):
        """Test the factory function with default values."""
        with patch.dict('os.environ', {}, clear=True):
            crawler = create_web_crawler()
            
            assert isinstance(crawler, WebCrawler)
            assert crawler.timeout == 10
            assert crawler.max_content_length == 50000


class TestWebCrawlerIntegration:
    """Integration tests for WebCrawler (with real HTTP mocking)."""
    
    @patch('requests.Session.get')
    def test_full_crawl_workflow(self, mock_get):
        """Test complete crawl workflow from HTTP request to Document creation."""
        # Mock HTTP response
        html_content = """
        <html>
        <head>
            <title>Integration Test Article</title>
            <meta name="description" content="Testing full workflow">
        </head>
        <body>
            <main>
                <h1>Integration Test</h1>
                <p>This is a full integration test of the web crawler.</p>
            </main>
        </body>
        </html>
        """
        
        mock_response = Mock()
        mock_response.text = html_content
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        crawler = WebCrawler()
        url = "https://test.example.com/article"
        document = crawler.crawl_url(url)
        
        # Verify Document structure
        assert isinstance(document, Document)
        assert len(document.page_content) > 0
        assert "Integration Test" in document.page_content
        assert "full integration test" in document.page_content
        
        # Verify metadata
        assert document.metadata['url'] == url
        assert document.metadata['title'] == 'Integration Test Article'
        assert document.metadata['domain'] == 'test.example.com'
        assert 'content_length' in document.metadata
        assert 'extraction_method' in document.metadata