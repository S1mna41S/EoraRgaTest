import trafilatura
from bs4 import BeautifulSoup


def extract_text(url: str, html: str | None) -> str | None:
    if not html:
        return None

    # trafilatura извлекает текст
    extracted = trafilatura.extract(html, include_comments=False, include_tables=False, favor_recall=True)
    stripped = extracted.strip() if extracted is not None else None
    if stripped:
        return stripped

    # Если не получилось, просто возвращаем get_text() через BS4
    soup = BeautifulSoup(html, 'lxml')
    text = soup.get_text(separator='\n', stripped=True)

    return text if text is not None and len(text) > 100 else None
