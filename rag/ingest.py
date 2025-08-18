import asyncio
import json
import os

import httpx
from loguru import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

from app.config import INDEX_DIR, LINKS_PATH, RAW_DIR, gigachat_embeddings_kwargs
from app.utils import extract_text


def _get_embeddings():
    # Пытаемся GigaChat → если не взлетело (в т.ч. SSL), мгновенно уходим в локальные эмбеддинги
    try:
        from langchain_gigachat import GigaChatEmbeddings
        from app.config import gigachat_embeddings_kwargs
        emb = GigaChatEmbeddings(**gigachat_embeddings_kwargs())
        try:
            # Предполётная проверка — если тут упадёт (SSL/токен), уйдём в fallback
            _ = emb.embed_query("healthcheck")
            return emb
        except Exception as e:
            raise RuntimeError(f"GigaChat embeddings unavailable: {e}")
    except Exception as e:
        from loguru import logger
        logger.warning(f"Fallback to sentence-transformers: {e}")
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        return SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


async def _fetch(client: httpx.AsyncClient, url: str) -> dict:
    try:
        resp = await client.get(url, timeout=30)
        resp.raise_for_status()
        html = resp.text

        # Сохраним сырой html
        safe = url.replace('https://', '').replace('/', '\\')
        full_path = os.path.join(RAW_DIR, f'{safe}.html')
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(os.path.join(RAW_DIR, f'{safe}.html'), 'w', encoding='utf-8') as f:
            f.write(html)
        text = extract_text(url, html)
        return {"url": url, "text": text}
    except Exception as e:
        logger.warning(f"Не удалось извлечь {url}: {e}")
        return {"url": url, "text": None}


async def crawl(urls: list[str]) -> list[dict]:
    async with httpx.AsyncClient(follow_redirects=True, headers={"User-Agent": "eora-rag/1.0"}) as client:
        tasks = [_fetch(client, u) for u in urls]
        return await asyncio.gather(*tasks)


def _load_links(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def build_index(items: tuple):
    # Фильтруем пустые
    docs = []
    for it in items:
        if not it['text']:
            continue
        meta = {'source': it['url']}
        docs.append(Document(page_content=it["text"], metadata=meta))

    if not docs:
        return

    # Собираем чанки
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=200, separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(docs)

    # Сохраняем чанки в локальную бд
    embeddings = _get_embeddings()
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(INDEX_DIR)

    # сохраним источники для отладки
    srcs = sorted({c.metadata['source'] for c in chunks})
    with open(os.path.join(INDEX_DIR, 'sources.json'), "w", encoding='utf-8') as f:
        json.dump(srcs, f, ensure_ascii=False, indent=2)

    print(f'Indexed {len(chunks)} chunks from {len(srcs)} sources into {INDEX_DIR}')


if __name__ == '__main__':
    urls = _load_links(LINKS_PATH)
    items = asyncio.run(crawl(urls))
    build_index(items)
