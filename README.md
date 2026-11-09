# Local RAG Assistant

## Overview

This project is a small, local document question-answering application. Retrieval-Augmented Generation (RAG) first finds relevant passages in a document collection and can then pass those passages to a language model to draft an answer. Keeping retrieval and generation separate makes it possible to inspect the evidence before asking a model to respond.

The application offers TF-IDF and sentence-transformer retrieval, a FastAPI interface, and optional answer generation through a locally running Ollama server. Retrieval works without Ollama; TF-IDF also works without downloading an embedding model.

## Features

- **TF-IDF retrieval** ranks chunks by lexical similarity.
- **Sentence-transformer embeddings** rank chunks by vector similarity.
- **Tokenizer-aware chunking** keeps embedding input within the selected model's sequence limit, including special tokens.
- **Optional Ollama generation** sends retrieved context to a local model such as Qwen.
- **FastAPI endpoints** provide `/health` and `/ask`.
- **Reproducible retrieval evaluation** uses a fixed set of 13 questions and reports document-level Hit@1 and Hit@3.
- **Automated tests** cover data loading, chunking, retrieval, API behavior, and Ollama request handling.

## Architecture

```text
Documents → Loading → Chunking → Retrieval → Optional LLM → API Response
```

TF-IDF represents text using weighted word and phrase counts, so it is fast and easy to inspect but can miss paraphrases with different vocabulary. Embeddings represent meaning as dense vectors and can match semantically similar wording, at the cost of downloading and running a pretrained model.

The two retrieval methods use different chunking rules: TF-IDF uses word-based windows, while embeddings use the selected model's tokenizer and sequence limit. Their evaluation results are therefore comparisons of the complete retrieval setups, not only of the ranking algorithms.

## Project Structure

```text
.
├── data/
│   ├── ml_docs/                 # Demonstration machine-learning documents
│   ├── python_docs/             # Demonstration Python documents
│   ├── SOURCES.md               # Dataset provenance notes
│   └── evaluation_queries.json  # Fixed evaluation queries and expected files
├── docs/
│   └── LEARNING_GUIDE.md       # Study guide (Russian)
├── results/
│   └── retrieval_comparison.json
├── scripts/
│   └── evaluate_retrieval.py
├── src/
│   ├── data/loader.py           # Document loading and chunking
│   ├── models/
│   │   ├── tfidf_search.py
│   │   ├── embedding_search.py
│   │   └── llm_generator.py
│   └── service/api.py           # FastAPI application
├── tests/                      # pytest checks
├── .env.example                 # Environment-variable examples (not auto-loaded)
├── requirements.txt
├── requirements-dev.txt
├── LICENSE
└── README.md
```

## Requirements

- Python 3.12 was used for the recorded experiment. The project has not been validated across a wider Python version matrix.
- Runtime dependencies include NumPy, scikit-learn, Sentence Transformers, FastAPI, Pydantic, Uvicorn, and Requests. Development dependencies add pytest.
- The first embeddings request downloads the configured sentence-transformer model if it is not already cached. TF-IDF does not load that model.
- Ollama is needed only when you want local LLM answer generation; retrieval can be used without it.

## Installation

After the prepared project is merged into the public `main` branch, clone it with:

```bash
git clone --branch main https://github.com/abrician/local-rag-assistant.git
cd local-rag-assistant
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

## Quick Start

Start the API from the project root (with the virtual environment activated):

```bash
python -m uvicorn src.service.api:app --host 127.0.0.1 --port 8000
```

Open Swagger UI at <http://127.0.0.1:8000/docs>. Check service status with `curl http://127.0.0.1:8000/health` on Linux/macOS, or `Invoke-RestMethod http://127.0.0.1:8000/health` in PowerShell. The health endpoint checks the document corpus and reports Ollama availability.

TF-IDF request (does not need the embedding model or Ollama):

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"переобучение регуляризация dropout","method":"tfidf","top_k":3,"use_llm":false}'
```

Embedding request (downloads the configured model on first use if it is not cached):

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"Как распознать переобучение?","method":"embeddings","top_k":3,"use_llm":false}'
```

To request an answer from Ollama, first run Ollama and make the configured model available, then set `use_llm` to `true` in either request. The default model name is `qwen2.5:1.5b`. For example, send this JSON to `POST /ask` using Swagger UI:

```json
{"query":"What is overfitting?","method":"tfidf","top_k":3,"use_llm":true}
```

When Ollama is unavailable, retrieval still returns results and `llm_answer` is `null`.

PowerShell request example:

```powershell
$body = @{ query = 'переобучение регуляризация'; method = 'tfidf'; top_k = 3; use_llm = $false } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask -ContentType 'application/json' -Body $body
```

## Configuration

The application reads these environment variables; it does **not** load `.env` files automatically:

- `MODEL_NAME`: sentence-transformer model name; default `all-MiniLM-L6-v2`.
- `OLLAMA_URL`: Ollama server URL; default `http://localhost:11434`.
- `OLLAMA_MODEL`: model name passed to Ollama; default `qwen2.5:1.5b`.

Linux/macOS shell:

```bash
export MODEL_NAME=all-MiniLM-L6-v2
export OLLAMA_URL=http://localhost:11434
export OLLAMA_MODEL=qwen2.5:1.5b
```

Windows PowerShell:

```powershell
$env:MODEL_NAME = 'all-MiniLM-L6-v2'
$env:OLLAMA_URL = 'http://localhost:11434'
$env:OLLAMA_MODEL = 'qwen2.5:1.5b'
```

## Evaluation

The checked-in results in [`results/retrieval_comparison.json`](results/retrieval_comparison.json) use the same 13 questions and count a query as a hit when its expected source document appears among the top unique documents. Mean query time is the recorded total query time divided by 13; it excludes model loading and index construction.

| Method | Hit@1 | Hit@3 | Mean query time |
|---|---:|---:|---:|
| TF-IDF | 11/13 (84.6%) | 13/13 (100.0%) | 0.0006 s |
| `all-MiniLM-L6-v2` | 10/13 (76.9%) | 11/13 (84.6%) | 0.0179 s |
| `paraphrase-multilingual-MiniLM-L12-v2` | 10/13 (76.9%) | 13/13 (100.0%) | 0.0287 s |

Only 13 queries were evaluated, which is too small a sample for universal conclusions. Counting unique source documents avoids rewarding duplicate chunks from one document, but the underlying chunking differs between TF-IDF and embeddings. These retrieval metrics do not measure the quality or factuality of generated answers.

Reproduce the experiment with:

```bash
python scripts/evaluate_retrieval.py
```

The script loads both sentence-transformer models and may download them if they are not cached. Results can vary with hardware, dependency versions, and model revisions.

## Testing

Run the automated test suite with:

```bash
python -m pytest
```

## Limitations

- The bundled corpus is a small demonstration set, not a broad knowledge base.
- The evaluation set contains only 13 queries.
- Embedding retrieval depends on downloading the model on first use when no local cache exists.
- A real end-to-end generation test against Qwen has not been confirmed; automated Ollama tests use mocked HTTP behavior.
- The project has no production deployment infrastructure, persistent database, authentication, or monitoring system.

## Project Background

This application grew out of university coursework and was extracted into a standalone learning project. It is intended to demonstrate and support study of retrieval, embeddings, local generation, APIs, and evaluation; it is not presented as a commercial production system.
