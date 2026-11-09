"""Checks for lossless, token-bounded document chunks."""

from pathlib import Path

import pytest

from src.data.loader import (clean_text, load_and_chunk_documents, load_documents,
                             split_into_chunks, split_into_token_chunks)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_bundled_corpus():
    documents = load_documents(str(DATA_DIR))
    chunks = load_and_chunk_documents(str(DATA_DIR))
    assert len(documents) == 14
    assert len(chunks) == 14
    assert all(chunk["text"] and chunk["source"] for chunk in chunks)
    assert {chunk["category"] for chunk in chunks} == {"ml_docs", "python_docs"}


def test_python_formatting_survives_cleaning_and_short_chunk():
    text = "def greet(name):\r\n    return f'Hi, {name}'\r\n\r\nprint(greet('Ada'))\r\n"
    expected = "def greet(name):\n    return f'Hi, {name}'\n\nprint(greet('Ada'))"
    assert clean_text(text) == expected
    assert split_into_chunks(expected, chunk_size=100, overlap=5) == [expected]


def test_word_chunks_preserve_all_characters():
    text = "one  two\n    three\n\nfour five"
    chunks = split_into_chunks(text, chunk_size=3, overlap=1)
    assert chunks == ["one  two\n    three", "three\n\nfour five"]
    assert not any(not chunk.strip() for chunk in chunks)


def test_empty_and_invalid_chunk_parameters():
    assert split_into_chunks("   \n", 3, 1) == []
    for size, overlap in [(0, 0), (2, 2), (2, 3), (2, -1)]:
        with pytest.raises(ValueError):
            split_into_chunks("one two", size, overlap)


@pytest.mark.parametrize("model_name", [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
])
def test_real_tokenizer_limit_and_coverage(model_name):
    from sentence_transformers import SentenceTransformer
    from src.data.loader import model_chunk_options
    model = SentenceTransformer(model_name)
    chunk_size, overlap = model_chunk_options(model.tokenizer, model.max_seq_length)
    text = "def square(x):\n    return x * x\n\n" + " ".join(
        f"word{i}" for i in range(400)
    )
    chunks = split_into_token_chunks(text, model.tokenizer, model.max_seq_length,
                                     chunk_size=chunk_size, overlap=overlap)
    assert len(chunks) > 1
    assert chunks[0].startswith("def square(x):\n    return x * x")
    assert chunks[-1].endswith("word399")
    assert all(len(model.tokenizer(chunk)["input_ids"]) <= model.max_seq_length
               for chunk in chunks)
    positions = [text.find(chunk) for chunk in chunks]
    assert all(position >= 0 for position in positions)
    assert positions[0] == 0
    assert positions[-1] + len(chunks[-1]) == len(text)
    assert all(positions[index + 1] <= positions[index] + len(chunks[index])
               for index in range(len(chunks) - 1))
    assert split_into_token_chunks("x = 1\n    print(x)", model.tokenizer,
                                   model.max_seq_length,
                                   chunk_size=chunk_size,
                                   overlap=overlap) == ["x = 1\n    print(x)"]
    with pytest.raises(ValueError):
        split_into_token_chunks(text, model.tokenizer, model.max_seq_length,
                                chunk_size=model.max_seq_length, overlap=1)
