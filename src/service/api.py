"""FastAPI endpoints for document retrieval and optional Ollama answers."""

import os
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data.loader import load_and_chunk_documents, model_chunk_options
from src.models.llm_generator import generate_answer, is_ollama_available
from src.models.tfidf_search import TFIDFSearch

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
STARTED_AT = time.time()
STATS = {"requests_total": 0, "requests_tfidf": 0,
         "requests_embeddings": 0, "errors_total": 0}


class AskRequest(BaseModel):
    query: str
    top_k: int = 3
    method: str = "embeddings"
    use_llm: bool = False


class SearchResult(BaseModel):
    rank: int
    score: float
    text: str
    source: str
    filename: str
    category: str
    chunk_id: int


class AskResponse(BaseModel):
    query: str
    method: str
    results: list[SearchResult]
    total_found: int
    response_time_ms: float
    llm_answer: str | None = None
    llm_available: bool = False


app = FastAPI(title="Local RAG Assistant", version="0.1.0")


@lru_cache(maxsize=1)
def get_chunks() -> list[dict]:
    """Read only documents inside this project directory."""
    chunks = load_and_chunk_documents(str(DATA_DIR))
    if not chunks:
        raise RuntimeError("No documents found")
    return chunks


@lru_cache(maxsize=1)
def get_tfidf() -> TFIDFSearch:
    model = TFIDFSearch()
    model.fit(get_chunks())
    return model


@lru_cache(maxsize=1)
def get_embeddings():
    """Load the pretrained model only after an embeddings request."""
    from src.models.embedding_search import EmbeddingSearch
    model = EmbeddingSearch(model_name=os.getenv("MODEL_NAME", "all-MiniLM-L6-v2"))
    chunk_size, overlap = model_chunk_options(model.model.tokenizer,
                                              model.model.max_seq_length)
    model.fit(load_and_chunk_documents(
        str(DATA_DIR), chunk_size=chunk_size, overlap=overlap,
        tokenizer=model.model.tokenizer,
        max_seq_length=model.model.max_seq_length,
    ))
    return model


@app.get("/health")
def health():
    try:
        chunk_count = len(get_chunks())
    except Exception:
        STATS["errors_total"] += 1
        raise HTTPException(status_code=503, detail="Document corpus unavailable") from None
    return {
        "status": "ok",
        "chunks_indexed": chunk_count,
        "methods_available": ["tfidf", "embeddings"],
        "llm_available": is_ollama_available(),
        "metrics": {**STATS, "uptime_seconds": int(time.time() - STARTED_AT)},
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    start = time.perf_counter()
    if not request.query.strip():
        STATS["errors_total"] += 1
        raise HTTPException(status_code=400, detail="Query must not be empty")
    if request.method not in ("tfidf", "embeddings"):
        STATS["errors_total"] += 1
        raise HTTPException(status_code=400, detail="Use 'tfidf' or 'embeddings'")
    if not 1 <= request.top_k <= 10:
        STATS["errors_total"] += 1
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 10")

    try:
        model = get_tfidf() if request.method == "tfidf" else get_embeddings()
        raw_results = model.search(request.query, top_k=request.top_k)
    except Exception:
        STATS["errors_total"] += 1
        detail = ("Document corpus unavailable" if request.method == "tfidf"
                  else "Embedding search unavailable")
        raise HTTPException(status_code=503, detail=detail) from None
    STATS["requests_total"] += 1
    STATS[f"requests_{request.method}"] += 1
    available = is_ollama_available()
    answer = (generate_answer(request.query, [item["text"] for item in raw_results])
              if request.use_llm else None)
    return AskResponse(
        query=request.query, method=request.method,
        results=[SearchResult(**item) for item in raw_results],
        total_found=len(raw_results),
        response_time_ms=round((time.perf_counter() - start) * 1000, 2),
        llm_answer=answer, llm_available=available,
    )
