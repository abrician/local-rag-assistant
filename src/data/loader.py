"""Load the bundled UTF-8 documents and preserve their text when chunking."""

import re
from pathlib import Path


def load_documents(data_dir: str = "data") -> list[dict]:
    """Read text files in a stable order, with paths relative to the corpus."""
    root = Path(data_dir)
    documents = []
    for path in sorted(root.rglob("*.txt")):
        parts = path.relative_to(root).parts
        category = parts[0] if parts[0] in ("ml_docs", "python_docs") else "other"
        documents.append({
            "text": path.read_text(encoding="utf-8"),
            "source": str(path.relative_to(root)),
            "filename": path.name,
            "category": category,
        })
    return documents


def clean_text(text: str) -> str:
    """Preserve indentation and line breaks; normalize only line endings."""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")


def _check_window(chunk_size: int, overlap: int) -> None:
    if not isinstance(chunk_size, int) or not isinstance(overlap, int):
        raise ValueError("chunk_size and overlap must be integers")
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("chunk_size must be positive and 0 <= overlap < chunk_size")


def _chunk_by_offsets(text: str, offsets: list[tuple[int, int]],
                      chunk_size: int, overlap: int) -> list[str]:
    """Slice the original text so whitespace and code formatting survive."""
    _check_window(chunk_size, overlap)
    if not offsets:
        return [text] if text.strip() else []
    chunks = []
    step = chunk_size - overlap
    previous_right = 0
    for start in range(0, len(offsets), step):
        stop = min(start + chunk_size, len(offsets))
        left = 0 if start == 0 else min(offsets[start][0], previous_right)
        right = len(text) if stop == len(offsets) else offsets[stop - 1][1]
        chunk = text[left:right]
        if chunk.strip():
            chunks.append(chunk)
        previous_right = right
        if stop == len(offsets):
            break
    return chunks


def split_into_chunks(text: str, chunk_size: int = 200,
                      overlap: int = 50) -> list[str]:
    """Word-based chunks for TF-IDF, without modifying document text."""
    offsets = [match.span() for match in re.finditer(r"\S+", text)]
    return _chunk_by_offsets(text, offsets, chunk_size, overlap)


def split_into_token_chunks(text: str, tokenizer, max_seq_length: int,
                            chunk_size: int = 192, overlap: int = 32) -> list[str]:
    """Use the embedding model's tokenizer and reserve room for special tokens."""
    _check_window(chunk_size, overlap)
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    if chunk_size + special_tokens > max_seq_length:
        raise ValueError("chunk_size plus special tokens exceeds the model limit")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer with character offsets is required")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False, verbose=False)
    offsets = [(left, right) for left, right in encoded["offset_mapping"] if right > left]
    if not offsets:
        return [text] if text.strip() else []
    chunks = []
    start = 0
    previous_right = 0
    while start < len(offsets):
        stop = min(start + chunk_size, len(offsets))
        left = 0 if start == 0 else min(offsets[start][0], previous_right)
        while True:
            right = len(text) if stop == len(offsets) else offsets[stop - 1][1]
            chunk = text[left:right]
            length = len(tokenizer(chunk, add_special_tokens=True,
                                   truncation=False, verbose=False)["input_ids"])
            if length <= max_seq_length:
                break
            stop -= 1
            if stop <= start:
                raise ValueError("A token cannot fit into the model context")
        if chunk.strip():
            chunks.append(chunk)
        previous_right = right
        if stop == len(offsets):
            break
        start = max(start + 1, stop - overlap)
    return chunks


def model_chunk_options(tokenizer, max_seq_length: int) -> tuple[int, int]:
    """Choose the same rule for every model, within each model's context limit."""
    content_limit = max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)
    if content_limit <= 0:
        raise ValueError("The model has no room for document tokens")
    chunk_size = min(192, content_limit)
    return chunk_size, min(32, chunk_size // 6)


def load_and_chunk_documents(data_dir: str = "data", chunk_size: int = 200,
                             overlap: int = 50, tokenizer=None,
                             max_seq_length: int | None = None) -> list[dict]:
    """Build either word chunks or model-specific token chunks."""
    all_chunks = []
    for document in load_documents(data_dir):
        text = clean_text(document["text"])
        if tokenizer is None:
            chunks = split_into_chunks(text, chunk_size, overlap)
        else:
            if max_seq_length is None:
                raise ValueError("max_seq_length is required with a tokenizer")
            chunks = split_into_token_chunks(text, tokenizer, max_seq_length,
                                             chunk_size, overlap)
        for chunk_id, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": document["source"],
                "filename": document["filename"],
                "category": document["category"],
                "chunk_id": chunk_id,
            })
    return all_chunks
