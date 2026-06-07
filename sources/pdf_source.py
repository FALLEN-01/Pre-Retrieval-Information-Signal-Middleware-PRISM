from typing import Optional
import pdfplumber


class PDFSource:
    """Extracts text from PDF files using pdfplumber.
    """

    def fetch(self, path: str) -> Optional[str]:
        try:
            with pdfplumber.open(path) as pdf:
                pages = [p.extract_text() for p in pdf.pages if p.extract_text()]
            return "\n\n".join(pages) if pages else None
        except Exception as e:
            print(f"[PDFSource] Failed to read {path}: {e}")
            return None

    def fetch_many(self, paths: list[str]) -> list[dict]:
        results = []
        for path in paths:
            text = self.fetch(path)
            if text:
                results.append({"source": path, "source_type": "pdf", "content": text})
        return results
