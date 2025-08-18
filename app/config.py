import os
from dotenv import load_dotenv

load_dotenv()


def _getenv_bool(key: str, default: bool = True) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on", "y"}


GIGACHAT_CREDENTIALS = (os.getenv("GIGACHAT_CREDENTIALS") or "").strip()  # base64(client_id:client_secret)
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS").strip()
GIGACHAT_VERIFY_SSL = _getenv_bool("GIGACHAT_VERIFY_SSL", True)

# Опционально (часто пусто)
GIGACHAT_ACCESS_TOKEN = (os.getenv("GIGACHAT_ACCESS_TOKEN") or "").strip()  # bearer-токен (~30 мин), обычно не задаём
GIGACHAT_BASE_URL = os.getenv("GIGACHAT_BASE_URL") or None
GIGACHAT_AUTH_URL = os.getenv("GIGACHAT_AUTH_URL") or None

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
    Общие kwargs для langchain_gigachat.GigaChat / GigaChatEmbeddings.
    Не задаём имя модели — используем дефолты SDK.
    """
    kwargs: dict = {"verify_ssl_certs": GIGACHAT_VERIFY_SSL}
    if GIGACHAT_BASE_URL:
        kwargs["base_url"] = GIGACHAT_BASE_URL
    if GIGACHAT_AUTH_URL:
        kwargs["auth_url"] = GIGACHAT_AUTH_URL

    if GIGACHAT_ACCESS_TOKEN:
        kwargs["access_token"] = GIGACHAT_ACCESS_TOKEN
    elif GIGACHAT_CREDENTIALS:
        kwargs["credentials"] = GIGACHAT_CREDENTIALS
        kwargs["scope"] = GIGACHAT_SCOPE
    return kwargs


def gigachat_gtchat_kwargs() -> dict:
    return dict(gigachat_common_kwargs())


def gigachat_embeddings_kwargs() -> dict:
    return dict(gigachat_common_kwargs())
