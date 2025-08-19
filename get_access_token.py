from langchain_gigachat import GigaChat
from app.config import gigachat_chat_kwargs


def get_access_token():
    llm = GigaChat(**gigachat_chat_kwargs())
    return llm.get_token()
