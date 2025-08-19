from fastapi import FastAPI
from pydantic import BaseModel
from rag.answer import answer

app = FastAPI(title="EORA RAG API")


@app.get("/health")
def health():
    return {"ok": True}


class AskRequest(BaseModel):
    question: str
    k: int = 6


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    result = answer(req.question, k=req.k)
    return AskResponse(**result)
