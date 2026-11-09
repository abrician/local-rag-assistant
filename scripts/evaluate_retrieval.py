"""Reproduce retrieval metrics on the original 13 course queries."""

import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_and_chunk_documents, model_chunk_options
from src.models.embedding_search import EmbeddingSearch
from src.models.tfidf_search import TFIDFSearch

DATA_DIR = ROOT / "data"
CASES = json.loads((DATA_DIR / "evaluation_queries.json").read_text(encoding="utf-8"))
MODELS = ["tfidf", "sentence-transformers/all-MiniLM-L6-v2",
          "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]


def unique_documents(results: list[dict]) -> list[str]:
    """Keep the first occurrence of each document in ranked chunk results."""
    return list(dict.fromkeys(item["source"] for item in results))


def evaluate(model, name: str, chunks: list[dict], build_seconds: float,
             load_seconds: float) -> dict:
    rows = []
    query_seconds = []
    for case in CASES:
        start = time.perf_counter()
        results = model.search(case["query"], top_k=len(chunks))
        query_seconds.append(time.perf_counter() - start)
        documents = unique_documents(results)
        expected = next(chunk["source"] for chunk in chunks
                        if chunk["filename"] == case["expected_filename"])
        rows.append({
            "query": case["query"], "expected_source": expected,
            "top_3_sources": documents[:3],
            "hit_at_1": expected in documents[:1],
            "hit_at_3": expected in documents[:3],
            "raw_chunk_hit_at_3": expected in [item["source"] for item in results[:3]],
        })
    return {
        "model": name, "document_count": len({chunk["source"] for chunk in chunks}),
        "chunk_count": len(chunks), "queries": len(CASES),
        "correct_at_1": sum(row["hit_at_1"] for row in rows),
        "correct_at_3": sum(row["hit_at_3"] for row in rows),
        "raw_chunk_correct_at_3": sum(row["raw_chunk_hit_at_3"] for row in rows),
        "hit_at_1": sum(row["hit_at_1"] for row in rows) / len(rows),
        "hit_at_3": sum(row["hit_at_3"] for row in rows) / len(rows),
        "index_build_seconds": round(build_seconds, 4),
        "model_load_seconds": round(load_seconds, 4),
        "query_total_seconds": round(sum(query_seconds), 4),
        "query_mean_seconds": round(sum(query_seconds) / len(query_seconds), 4),
        "cases": rows,
    }


def run() -> list[dict]:
    reports = []
    for name in MODELS:
        if name == "tfidf":
            load_seconds = 0.0
            start = time.perf_counter()
            chunks = load_and_chunk_documents(str(DATA_DIR))
            model = TFIDFSearch()
        else:
            load_start = time.perf_counter()
            model = EmbeddingSearch(name)
            load_seconds = time.perf_counter() - load_start
            start = time.perf_counter()
            chunk_size, overlap = model_chunk_options(model.model.tokenizer,
                                                      model.model.max_seq_length)
            chunks = load_and_chunk_documents(
                str(DATA_DIR), chunk_size=chunk_size, overlap=overlap,
                tokenizer=model.model.tokenizer,
                max_seq_length=model.model.max_seq_length,
            )
        model.fit(chunks)
        report = evaluate(model, name, chunks, time.perf_counter() - start,
                          load_seconds)
        if name != "tfidf":
            lengths = [len(model.model.tokenizer(chunk["text"])["input_ids"])
                       for chunk in chunks]
            report["model_max_seq_length"] = model.model.max_seq_length
            report["longest_chunk_tokens"] = max(lengths)
            report["over_limit_chunks"] = sum(
                length > model.model.max_seq_length for length in lengths
            )
            report["token_chunk_size"] = chunk_size
            report["token_overlap"] = overlap
        reports.append(report)
        print(name, report["correct_at_1"], report["correct_at_3"],
              report["index_build_seconds"], report["query_total_seconds"])
    output = ROOT / "results" / "retrieval_comparison.json"
    output.parent.mkdir(exist_ok=True)
    payload = {
        "evaluation": "Original 13 course queries, ranked by unique source document",
        "python": sys.version.split()[0],
        "versions": {package: version(package) for package in
                     ("numpy", "scikit-learn", "sentence-transformers", "torch")},
        "models": reports,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return reports


if __name__ == "__main__":
    run()
