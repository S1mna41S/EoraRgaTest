import os
from dotenv import load_dotenv

load_dotenv()


def _getenv_bool(key: str, default: bool = True) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on", "y"}


GIGACHAT_CREDENTIALS = (os.getenv("GIGACHAT_CREDENTIALS") or "").strip()

# короткоживущий bearer (если уже есть). Поддерживаем оба имени: USER_TOKEN и ACCESS_TOKEN
GIGACHAT_USER_TOKEN = (
        os.getenv("GIGACHAT_USER_TOKEN")
        or os.getenv("GIGACHAT_ACCESS_TOKEN")
        or ""
).strip()

GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS").strip()
GIGACHAT_VERIFY_SSL = _getenv_bool("GIGACHAT_VERIFY_SSL", True)

# Пути проекта
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(DATA_DIR, "index")
LINKS_PATH = os.path.join(DATA_DIR, "links.txt")
RAW_DIR = os.path.join(DATA_DIR, "raw")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)


def gigachat_common_kwargs() -> dict:
    """
    Общие kwargs для langchain_gigachat.{GigaChat,GigaChatEmbeddings}.
    - Если задан user_token → используем его напрямую.
    """
    kwargs: dict = {"verify_ssl_certs": GIGACHAT_VERIFY_SSL}
    if GIGACHAT_USER_TOKEN:
        kwargs["access_token"] = GIGACHAT_USER_TOKEN
    elif GIGACHAT_CREDENTIALS:
        kwargs["credentials"] = GIGACHAT_CREDENTIALS
        kwargs["scope"] = GIGACHAT_SCOPE
    return kwargs


def gigachat_chat_kwargs() -> dict:
    return dict(gigachat_common_kwargs())


def gigachat_embeddings_kwargs() -> dict:
    return dict(gigachat_common_kwargs())


__all__ = [
    "INDEX_DIR", "LINKS_PATH", "RAW_DIR",
    "gigachat_common_kwargs", "gigachat_chat_kwargs", "gigachat_embeddings_kwargs",
]
