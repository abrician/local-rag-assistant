"""TF-IDF retrieval baseline for the bundled document chunks."""

import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFSearch:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000, ngram_range=(1, 2), min_df=1, sublinear_tf=True
        )
        self.chunks = []
        self.tfidf_matrix = None
        self.is_fitted = False

    def fit(self, chunks: list[dict]) -> None:
        """Build an index over the provided chunks."""
        if not chunks or not any(chunk["text"].strip() for chunk in chunks):
            raise ValueError("Cannot fit TF-IDF on an empty corpus")
        self.chunks = chunks
        self.tfidf_matrix = self.vectorizer.fit_transform(
            [chunk["text"] for chunk in chunks]
        )
        self.is_fitted = True

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Return nonzero matches in descending cosine similarity order."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before search()")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not query.strip():
            return []
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for rank, index in enumerate(indices, start=1):
            score = float(similarities[index])
            if score == 0.0:
                continue
            results.append({**self.chunks[index], "score": round(score, 4), "rank": rank})
        return results

    def save(self, path: str) -> None:
        """Save an index for trusted local use only."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump({"vectorizer": self.vectorizer, "chunks": self.chunks,
                         "tfidf_matrix": self.tfidf_matrix}, file)

    def load(self, path: str) -> None:
        """Load an index created locally; pickle is unsafe for untrusted files."""
        with open(path, "rb") as file:
            data = pickle.load(file)
        self.vectorizer = data["vectorizer"]
        self.chunks = data["chunks"]
        self.tfidf_matrix = data["tfidf_matrix"]
        self.is_fitted = True
