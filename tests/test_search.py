"""Retrieval checks for both search methods."""

from pathlib import Path

import pytest

from src.data.loader import load_and_chunk_documents
from src.models.tfidf_search import TFIDFSearch

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def chunks():
    return load_and_chunk_documents(str(DATA_DIR))


def test_tfidf_retrieval(chunks):
    model = TFIDFSearch()
    model.fit(chunks)
    results = model.search("классификация accuracy precision recall", top_k=3)
    assert len(results) <= 3
    assert "classification.txt" in [item["filename"] for item in results]
    assert [item["score"] for item in results] == sorted(
        [item["score"] for item in results], reverse=True
    )
    assert model.search("zzzxxyy 928174") == []
    assert model.search("  ") == []
    with pytest.raises(ValueError):
        model.search("query", top_k=0)
    with pytest.raises(ValueError):
        TFIDFSearch().fit([])


def test_embeddings_retrieval():
    from src.models.embedding_search import EmbeddingSearch
    model = EmbeddingSearch()
    chunks = load_and_chunk_documents(str(DATA_DIR), chunk_size=192, overlap=32,
                                      tokenizer=model.model.tokenizer,
                                      max_seq_length=model.model.max_seq_length)
    model.fit(chunks)
    results = model.search("классификация accuracy precision recall")
    assert len(results) == 3
    assert "classification.txt" in [item["filename"] for item in results]
