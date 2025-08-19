from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import INDEX_DIR, gigachat_chat_kwargs, gigachat_embeddings_kwargs


def _get_embeddings_for_query():
    # Индекс собран на sentence-transformers — используем то же самое:
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    return SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def _load_vs():
    emb = _get_embeddings_for_query()
    return FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)


_vs = None


def _retriever(k: int = 6):
    global _vs
    if _vs is None:
        _vs = _load_vs()
    return _vs.as_retriever(search_kwargs={"k": k})


SYSTEM = (
    "Ты — помощник, который отвечает по материалам eora.ru. "
    "Используй только приведённые сниппеты контекста. "
    "Кратко и по делу. Если не хватает данных — честно скажи, чего не хватает."
)

USER_TEMPLATE = """Вопрос: {question}

Контекстные фрагменты:
{context}

Сформируй ответ на русском. В конце добавь блок:
Источники: [n] URL; перечисли только реально использованные источники.
Если конкретика опирается на один фрагмент — можешь проставлять [n] прямо в тексте.
"""


def _format_context(docs: list[Document]) -> tuple[str, list[str]]:
    sources_order: list[str] = []
    context_lines = []
    for d in docs:
        src = d.metadata.get("source", "")
        if src and src not in sources_order:
            sources_order.append(src)
    src_idx = {src: i + 1 for i, src in enumerate(sources_order)}

    for d in docs:
        src = d.metadata.get("source", "")
        if not src:
            continue
        n = src_idx[src]
        snippet = d.page_content[:600].replace("\n", " ").strip()
        context_lines.append(f"[{n}] {src}\n{snippet}\n")

    return "\n".join(context_lines), sources_order


def _get_chat_model():
    from langchain_gigachat import GigaChat
    return GigaChat(**gigachat_chat_kwargs())


def answer(question: str, k: int = 6) -> dict:
    retr = _retriever(k)
    docs = retr.get_relevant_documents(question)

    # Если ничего не нашли — вернём честный ответ без зова модели
    if not docs:
        return {
            "answer": "Не нашёл подходящих фрагментов на eora.ru для ответа. "
                      "Попробуйте переформулировать вопрос или добавить ссылки в data/links.txt и переиндексировать.",
            "sources": []
        }

    ctx_text, sources_order = _format_context(docs)

    model = _get_chat_model()
    messages = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=USER_TEMPLATE.format(question=question, context=ctx_text)),
    ]
    resp = model.invoke(messages)
    return {
        "answer": resp.content,
        "sources": sources_order
    }
