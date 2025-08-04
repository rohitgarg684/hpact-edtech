import requests

class URLFetcher:
    def fetch(self, url: str, max_length: int = 4000) -> str:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.text[:max_length]
