# Demonstration corpus provenance

The 14 `.txt` documents in `data/ml_docs/` and `data/python_docs/` were newly written for this standalone educational RAG project on 2026-11-09 with AI assistance. They cover the same broad topics as the earlier course corpus and retain filenames so the retrieval application can load them. Their explanatory prose, examples, and organization were created anew rather than copied or translated line by line from the course files or third-party documentation. The course files were consulted for topic coverage only. This statement does not claim human-only authorship.

The documents use short Python and library examples written for this project. Technical names, API identifiers, standard mathematical definitions, and brief conventional code syntax are not presented as original inventions. No external prose or code excerpts were intentionally copied into the replacement corpus.

The previous course-derived versions are absent from this release branch's ancestry, which starts at the repository's initial commit. They remain in the separate `codex/initial-rag-extraction` branch. Making the existing repository public while that branch remains available would expose its older files; review their reuse terms before changing repository visibility. This provenance note does not grant a license for third-party libraries or models.

`data/evaluation_queries.json` contains the 13 earlier course queries. They were kept unchanged to provide a fixed comparison across the corpus replacement; they are a small educational test set, not an independently collected benchmark. The results file records a new run against the replacement corpus.

The root `LICENSE` is the MIT license for the original project code and newly written project materials in the current tree. It does not replace the licenses or usage terms of third-party Python packages, pretrained models, Ollama, or any older course-derived content still reachable through Git history. Check those separately before public release.
