"""Check Ollama request construction without starting a model server."""

from src.models import llm_generator


def test_ollama_request_uses_configured_server(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://example.invalid:11434/")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:1.5b")
    calls = {}

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "Answer from excerpts"}

    def fake_get(url, timeout):
        calls["health_url"] = url
        return Response()

    def fake_post(url, json, timeout):
        calls["generate_url"] = url
        calls["payload"] = json
        return Response()

    monkeypatch.setattr(llm_generator.requests, "get", fake_get)
    monkeypatch.setattr(llm_generator.requests, "post", fake_post)
    answer = llm_generator.generate_answer("question", ["source excerpt"])
    assert answer == "Answer from excerpts"
    assert calls["health_url"] == "http://example.invalid:11434"
    assert calls["generate_url"] == "http://example.invalid:11434/api/generate"
    assert calls["payload"]["model"] == "qwen2.5:1.5b"
    assert "source excerpt" in calls["payload"]["prompt"]


def test_unavailable_ollama_returns_none(monkeypatch):
    import requests

    def fail(*args, **kwargs):
        raise requests.ConnectionError("server unavailable")

    monkeypatch.setattr(llm_generator.requests, "get", fail)
    assert llm_generator.is_ollama_available() is False
    assert llm_generator.generate_answer("question", ["excerpt"]) is None
