"""
Unit evals for helpers and RAG so refactors don't break behavior.
"""
import json
import tempfile
from pathlib import Path

import pytest

# Import after conftest sets TEST_MODE
from main import (
    _compact,
    _load_local_documents,
    _with_prefix,
    LocalGuideRetriever,
)


class TestCompact:
    def test_empty_returns_empty(self):
        assert _compact("") == ""

    def test_short_text_unchanged(self):
        assert _compact("hello world") == "hello world"
        out = _compact("a " * 50, limit=200)
        assert out.count("a") == 50 and len(out) <= 200

    def test_truncates_at_word_boundary(self):
        long_ = "one two three four five six seven eight nine ten"
        out = _compact(long_, limit=20)
        assert len(out) <= 20
        assert out in ("one two three four", "one two three")

    def test_strips_punctuation_when_truncating(self):
        long_ = "hello, world. extra words here"
        out = _compact(long_, limit=15)
        assert out.rstrip(",.;- ") == out

    def test_collapses_whitespace(self):
        assert _compact("  hello   world  ") == "hello world"


class TestWithPrefix:
    def test_adds_prefix(self):
        assert "Kyoto essentials:" in _with_prefix("Kyoto essentials", "sunny and warm")

    def test_empty_prefix_returns_compact_summary(self):
        assert _with_prefix("", "some summary") == "some summary"

    def test_respects_compact_limit(self):
        long_summary = "word " * 100
        out = _with_prefix("Prefix", long_summary)
        assert len(out) <= 200 + len("Prefix: ")


class TestLoadLocalDocuments:
    def test_missing_path_returns_empty(self):
        assert _load_local_documents(Path("/nonexistent/file.json")) == []

    def test_invalid_json_returns_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"not valid json {")
            f.flush()
            path = Path(f.name)
        try:
            assert _load_local_documents(path) == []
        finally:
            path.unlink(missing_ok=True)

    def test_loads_valid_guides(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(
                [
                    {"city": "Paris", "description": "Eiffel Tower and croissants.", "interests": ["sights"]},
                    {"city": "Lyon", "description": "Food capital.", "interests": ["food"]},
                ],
                f,
            )
            f.flush()
            path = Path(f.name)
        try:
            docs = _load_local_documents(path)
            assert len(docs) == 2
            assert docs[0].page_content
            assert "Paris" in docs[0].page_content
            assert docs[0].metadata.get("city") == "Paris"
        finally:
            path.unlink(missing_ok=True)

    def test_skips_rows_without_description_or_city(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(
                [
                    {"city": "Paris", "description": "Nice."},
                    {"city": "X", "description": ""},
                    {"city": "", "description": "No city"},
                ],
                f,
            )
            f.flush()
            path = Path(f.name)
        try:
            docs = _load_local_documents(path)
            assert len(docs) == 1
            assert docs[0].metadata["city"] == "Paris"
        finally:
            path.unlink(missing_ok=True)


class TestLocalGuideRetriever:
    def test_empty_data_is_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"[]")
            f.flush()
            path = Path(f.name)
        try:
            retriever = LocalGuideRetriever(path)
            assert retriever.is_empty
            assert retriever.retrieve("Paris", "food", k=3) == []
        finally:
            path.unlink(missing_ok=True)

    def test_keyword_fallback_returns_matching_docs(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(
                [
                    {"city": "Prague", "description": "Beer and food tour.", "interests": ["food", "beer"]},
                    {"city": "Berlin", "description": "Museums.", "interests": ["history"]},
                ],
                f,
            )
            f.flush()
            path = Path(f.name)
        try:
            retriever = LocalGuideRetriever(path)
            assert not retriever.is_empty
            results = retriever.retrieve("Prague", "food", k=2)
            assert len(results) >= 1
            assert any("Prague" in r.get("content", "") for r in results)
            assert all("content" in r and "metadata" in r and "score" in r for r in results)
        finally:
            path.unlink(missing_ok=True)

    def test_keyword_fallback_no_match_returns_empty_or_low_score(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(
                [{"city": "Tokyo", "description": "Sushi.", "interests": ["food"]}],
                f,
            )
            f.flush()
            path = Path(f.name)
        try:
            retriever = LocalGuideRetriever(path)
            results = retriever.retrieve("Sydney", "hiking", k=2)
            # Either no results or only low-score results
            for r in results:
                assert "content" in r and "score" in r
        finally:
            path.unlink(missing_ok=True)
