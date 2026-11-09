"""Sentence-transformer retrieval over the same document chunks."""

import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class EmbeddingSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.chunks = []
        self.embeddings = None
        self.is_fitted = False

    def fit(self, chunks: list[dict]) -> None:
        """Encode the chunks with a pretrained model."""
        self.chunks = chunks
        self.embeddings = self.model.encode(
            [chunk["text"] for chunk in chunks], convert_to_numpy=True
        )
        self.is_fitted = True

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Return chunks in descending cosine similarity order."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before search()")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not query.strip():
            return []
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        indices = np.argsort(similarities)[::-1][:top_k]
        return [{**self.chunks[index], "score": round(float(similarities[index]), 4),
                 "rank": rank} for rank, index in enumerate(indices, start=1)]

    def save(self, path: str) -> None:
        """Save local vectors without copying the pretrained model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings,
                         "model_name": self.model_name}, file)

    def load(self, path: str) -> None:
        """Load trusted local vectors produced by this model."""
        with open(path, "rb") as file:
            data = pickle.load(file)
        if data["model_name"] != self.model_name:
            raise ValueError("The saved vectors use a different embedding model")
        self.chunks = data["chunks"]
        self.embeddings = data["embeddings"]
        self.is_fitted = True
