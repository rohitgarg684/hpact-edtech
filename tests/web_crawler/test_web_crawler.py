import pytest
from unittest.mock import patch, MagicMock
from langchain.schema import Document
import web_crawler

TEST_URL = "https://example.com/somepage"

def mock_requests_get(*args, **kwargs):
    class MockResponse:
        def __init__(self, text, status_code=200):
            self.text = text
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code != 200:
                raise Exception("HTTP Error")

    # Simple HTML for tests
    html = """
    <html>
        <head>
            <title>Example Title</title>
        </head>
        <body>
            <h1>Header</h1>
            <p>Some content here.</p>
            <script>var a = 1;</script>
        </body>
    </html>
    """
    return MockResponse(html)

@patch("web_crawler.requests.get", side_effect=mock_requests_get)
@patch("web_crawler.trafilatura.extract", return_value="Extracted Content")
def test_crawl_and_parse_trafilatura_success(mock_traf, mock_req):
    doc = web_crawler.crawl_and_parse(TEST_URL)
    assert isinstance(doc, Document)
    assert doc.page_content == "Extracted Content"
    assert doc.metadata["url"] == TEST_URL
    assert doc.metadata["source"] == "https://example.com"
    assert doc.metadata["title"] == ""

@patch("web_crawler.requests.get", side_effect=mock_requests_get)
@patch("web_crawler.trafilatura.extract", return_value=None)
def test_crawl_and_parse_beautifulsoup_fallback(mock_traf, mock_req):
    doc = web_crawler.crawl_and_parse(TEST_URL)
    assert isinstance(doc, Document)
    assert "Header" in doc.page_content
    assert "Some content here." in doc.page_content
    assert "var a = 1;" not in doc.page_content  # script removed
    assert doc.metadata["title"] == "Example Title"
    assert doc.metadata["url"] == TEST_URL
    assert doc.metadata["source"] == "https://example.com"

@patch("web_crawler.requests.get", side_effect=mock_requests_get)
@patch("web_crawler.trafilatura.extract", return_value="   ")
def test_crawl_and_parse_trafilatura_empty_string(mock_traf, mock_req):
    doc = web_crawler.crawl_and_parse(TEST_URL)
    # Should fallback to BeautifulSoup
    assert "Header" in doc.page_content
    assert doc.metadata["title"] == "Example Title"

@patch("web_crawler.requests.get")
def test_crawl_and_parse_http_error(mock_req):
    mock_req.return_value = MagicMock(status_code=404, text="")
    mock_req.return_value.raise_for_status.side_effect = Exception("HTTP Error")
    with pytest.raises(Exception):
        web_crawler.crawl_and_parse(TEST_URL)