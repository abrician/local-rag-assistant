"""Optional answer generation through a local Ollama server."""

import os

import requests


def ollama_url() -> str:
    """Return the configured Ollama server URL without a trailing slash."""
    return os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")


def is_ollama_available(url: str | None = None) -> bool:
    """Check whether the Ollama server responds."""
    try:
        response = requests.get(url or ollama_url(), timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


def generate_answer(query: str, context_chunks: list[str],
                    model: str | None = None, max_tokens: int = 300) -> str | None:
    """Ask Qwen to answer from retrieved context; return None when unavailable."""
    if not is_ollama_available():
        return None
    context = "\n\n---\n\n".join(context_chunks)
    prompt = (
        "You are an assistant for Python and machine learning documentation. "
        "Use only the supplied excerpts. Answer briefly in Russian.\n\n"
        f"Excerpts:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    try:
        response = requests.post(
            f"{ollama_url()}/api/generate",
            json={"model": model or os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"),
                  "prompt": prompt, "stream": False,
                  "options": {"num_predict": max_tokens, "temperature": 0.3}},
            timeout=60,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except (requests.RequestException, ValueError):
        return None
