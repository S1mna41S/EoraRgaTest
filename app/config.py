import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
OPENAI_EMBEDDING_MODEL = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
LOCAL_EMBEDDER = os.getenv('LOCAL_EMBEDDER', 'sentence-transformers/all-MiniLM-L6-v2')

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
INDEX_DIR = os.path.join(DATA_DIR, 'index')
LINKS_PATH = os.path.join(DATA_DIR, 'links.txt')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)