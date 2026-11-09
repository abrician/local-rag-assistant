"""API checks that work without Ollama or an embedding model download."""

from fastapi.testclient import TestClient

from src.service import api


def test_tfidf_api_without_ollama(monkeypatch):
    monkeypatch.setattr(api, "is_ollama_available", lambda: False)
    client = TestClient(api.app)
    response = client.post("/ask", json={"query": "классификация accuracy",
                                         "method": "tfidf"})
    assert response.status_code == 200
    data = response.json()
    assert data["results"][0]["filename"] == "classification.txt"
    assert data["llm_answer"] is None
    assert data["llm_available"] is False
    assert client.get("/health").json()["chunks_indexed"] == 14


def test_invalid_request():
    client = TestClient(api.app)
    assert client.post("/ask", json={"query": " ", "method": "tfidf"}).status_code == 400
    assert client.post("/ask", json={"query": "text", "method": "unknown"}).status_code == 400
    assert client.post("/ask", json={"query": "text", "top_k": 0}).status_code == 400


def test_default_embeddings_request(monkeypatch):
    monkeypatch.setattr(api, "is_ollama_available", lambda: False)
    response = TestClient(api.app).post("/ask", json={"query": "классификация accuracy"})
    assert response.status_code == 200
    assert response.json()["method"] == "embeddings"
    assert response.json()["total_found"] == 3


def test_missing_documents_do_not_reveal_paths(monkeypatch):
    api.get_chunks.cache_clear()
    api.get_tfidf.cache_clear()
    monkeypatch.setattr(api, "load_and_chunk_documents", lambda *args, **kwargs: [])
    client = TestClient(api.app)
    health = client.get("/health")
    search = client.post("/ask", json={"query": "text", "method": "tfidf"})
    assert health.status_code == 503
    assert search.status_code == 503
    assert "/workspace" not in health.text + search.text
    api.get_chunks.cache_clear()
    api.get_tfidf.cache_clear()


def test_unavailable_embedding_model(monkeypatch):
    api.get_embeddings.cache_clear()
    def fail():
        raise RuntimeError("private/model/path")
    monkeypatch.setattr(api, "get_embeddings", fail)
    response = TestClient(api.app).post("/ask", json={"query": "text"})
    assert response.status_code == 503
    assert "private/model/path" not in response.text


def test_model_name_environment_variable(monkeypatch):
    from src.models import embedding_search
    names = []

    class Tokenizer:
        def num_special_tokens_to_add(self, pair=False):
            return 2

    class Model:
        def __init__(self, model_name):
            names.append(model_name)
            self.model = type("Encoder", (), {"tokenizer": Tokenizer(),
                                                "max_seq_length": 128})()

        def fit(self, chunks):
            self.chunks = chunks

    monkeypatch.setenv("MODEL_NAME", "example-model")
    monkeypatch.setattr(embedding_search, "EmbeddingSearch", Model)
    monkeypatch.setattr(api, "load_and_chunk_documents", lambda *args, **kwargs: [{}])
    api.get_embeddings.cache_clear()
    assert api.get_embeddings().chunks == [{}]
    assert names == ["example-model"]
    api.get_embeddings.cache_clear()
