"""
web_crawler.py

Implements web content extraction using requests, trafilatura, and BeautifulSoup.
Returns a LangChain Document object for downstream processing.

Functions:
- crawl_and_parse(url): Fetches and parses content from a URL into a Document.
"""

import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from urllib.parse import urlparse
import trafilatura

def crawl_and_parse(url: str) -> Document:
    """
    Sophisticated web crawler that fetches and parses web content.
    Uses trafilatura for extraction if possible; falls back to BeautifulSoup otherwise.

    Args:
        url (str): The target URL.

    Returns:
        Document: LangChain Document containing page content and metadata.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; LangChainCrawler/1.0; "
            "https://github.com/rohitgarg684/hpact-edtech)"
        )
    }
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    html = response.text

    # Try extracting with trafilatura
    extracted = trafilatura.extract(html, include_comments=False, include_tables=True)
    if extracted and len(extracted.strip()) > 0:
        content = extracted
        title = ""
    else:
        soup = BeautifulSoup(html, "lxml")
        for script in soup(["script", "style", "noscript"]):
            script.decompose()
        text = soup.get_text(separator="\n", strip=True)
        content = text
        title = soup.title.string if soup.title else ""

    parsed_url = urlparse(url)
    source = f"{parsed_url.scheme}://{parsed_url.netloc}"
    metadata = {
        "source": source,
        "url": url,
        "title": title,
    }

    return Document(page_content=content, metadata=metadata)
