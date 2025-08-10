import requests
from bs4 import BeautifulSoup
import trafilatura
from readability import Document

class WebCrawler:
    def __init__(self, user_agent=None, timeout=10):
        self.headers = {
            'User-Agent': user_agent or 'Mozilla/5.0 (compatible; WebCrawler/1.0; +https://example.com/bot)'
        }
        self.timeout = timeout

    def fetch_url(self, url):
        resp = requests.get(url, headers=self.headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.text

    def extract_with_trafilatura(self, html, url=None):
        downloaded = trafilatura.extract(html, url=url, include_comments=False)
        return downloaded or ""

    def extract_with_readability(self, html):
        doc = Document(html)
        content = doc.summary()
        title = doc.title()
        return title, content

    def extract_metadata(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.title.string if soup.title else ""
        meta_desc = ""
        desc_tag = soup.find('meta', attrs={'name': 'description'})
        if desc_tag and 'content' in desc_tag.attrs:
            meta_desc = desc_tag['content']
        return {
            "title": title,
            "meta_description": meta_desc
        }

    def crawl(self, url):
        html = self.fetch_url(url)
        # State-of-the-art content extraction
        main_content = self.extract_with_trafilatura(html, url)
        if not main_content:
            # fallback to readability-lxml if trafilatura fails
            _, main_content = self.extract_with_readability(html)
        meta = self.extract_metadata(html)
        return {
            "url": url,
            "title": meta["title"],
            "meta_description": meta["meta_description"],
            "content": main_content
        }

# Example usage
if __name__ == "__main__":
    crawler = WebCrawler()
    url = "https://en.wikipedia.org/wiki/LangChain"
    result = crawler.crawl(url)
    print("Title:", result["title"])
    print("Meta Description:", result["meta_description"])
    print("Content sample:", result["content"][:500])

    # For LangChain integration, you can use the result["content"] as Document input