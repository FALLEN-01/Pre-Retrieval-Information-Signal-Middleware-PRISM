from typing import Optional
import trafilatura


class WebSource:
    """Fetches and cleans text from web URLs using trafilatura.
    """

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def fetch(self, url: str) -> Optional[str]:
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded is None:
                return None
            return trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
            )
        except Exception as e:
            print(f"[WebSource] Failed to fetch {url}: {e}")
            return None

    def fetch_many(self, urls: list[str]) -> list[dict]:
        results = []
        for url in urls:
            text = self.fetch(url)
            if text:
                results.append({"source": url, "source_type": "web", "content": text})
        return results
