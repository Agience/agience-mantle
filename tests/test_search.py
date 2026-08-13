# tests/test_search.py
"""
Test search endpoints and query parsing.
"""
from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from httpx import AsyncClient

from mantle.search.query_parser import QueryParser, FieldOperator


class TestQueryParser:
    """Test query parser for operators."""
    
    def test_parse_basic_terms(self):
        """Test parsing basic search terms."""
        parser = QueryParser()
        result = parser.parse("machine learning")
        
        assert len(result.terms) == 2
        assert result.terms[0].text == "machine"
        assert result.terms[1].text == "learning"
        assert not result.has_filters()
    
    def test_parse_tag_filter(self):
        """Test tag: filter parsing."""
        parser = QueryParser()
        result = parser.parse("test tag:ml tag:ai")
        
        # Should have main term and tag filters
        assert len(result.terms) >= 1
        tag_filters = [f for f in result.filters if f.field in ("tag", "tags")]
        assert len(tag_filters) == 2
        assert any(f.value == "ml" for f in tag_filters)
        assert any(f.value == "ai" for f in tag_filters)
    
    def test_parse_type_filter(self):
        """Test type: content_type filter parsing."""
        parser = QueryParser()
        result = parser.parse("documents type:pdf")

        type_filters = [f for f in result.filters if f.field in ("type", "content_type")]
        assert len(type_filters) >= 1
        assert any("pdf" in f.value.lower() for f in type_filters)
    
    def test_parse_size_filter(self):
        """Test size:> and size:< parsing."""
        parser = QueryParser()
        result = parser.parse("files size:>10MB")
        
        size_filters = [f for f in result.filters if f.field == "size"]
        assert len(size_filters) >= 1
        assert any(f.operator == FieldOperator.GT for f in size_filters)
    
    def test_parse_control_params(self):
        """Test @control: parameter parsing."""
        parser = QueryParser()
        result = parser.parse("query @hybrid:on @lang:en")
        
        assert result.controls.get("hybrid") == "on"
        assert result.controls.get("lang") == "en"
    
    def test_parse_required_term(self):
        """Test + required term prefix."""
        parser = QueryParser()
        result = parser.parse("+required optional")
        
        # Find the required term
        required_terms = [t for t in result.terms if t.text == "required"]
        assert len(required_terms) == 1
    
    def test_parse_excluded_term(self):
        """Test ! excluded term prefix."""
        parser = QueryParser()
        result = parser.parse("include !exclude")
        
        # Find the excluded term
        excluded_terms = [t for t in result.terms if t.text == "exclude"]
        assert len(excluded_terms) == 1
    
    def test_parse_semantic_term(self):
        """Test ~ semantic term prefix."""
        parser = QueryParser()
        result = parser.parse("~semantic normal")
        
        # Find the semantic term
        semantic_terms = [t for t in result.terms if t.text == "semantic"]
        assert len(semantic_terms) == 1
    
    def test_parse_phrase_quoted(self):
        """Test quoted phrase parsing."""
        parser = QueryParser()
        result = parser.parse('"machine learning"')
        
        # Should recognize as phrase
        phrase_terms = [t for t in result.terms if t.is_phrase]
        assert len(phrase_terms) >= 1
    
    def test_there_is_no_query_side_hybrid_predicate(self):
        """Gone, not returning False. A predicate that exists is one a caller can believe.

        Nothing in a query string decides whether the semantic arm runs: the arm needs a query
        vector on the request and a provisioned AnchorSet on the node. A `should_use_hybrid`
        that answered from `@hybrid:` or `~` would be reporting an opinion no retrieval step
        consults, and `@hybrid:on` read as a working switch for exactly that reason.

        `@hybrid:` still PARSES, like any `@name:value` — `test_parse_control_params` covers
        that, and lifting it out of the term stream is why it does not become a search term.
        Parsing it is not the same as acting on it, and only the second was ever claimed here.
        """
        from mantle.search.query_parser import ParsedQuery

        assert not hasattr(ParsedQuery, "should_use_hybrid")
        parsed = QueryParser().parse("query @hybrid:on")
        assert parsed.controls.get("hybrid") == "on"      # parsed, and inert
        assert [t.text for t in parsed.terms] == ["query"]


    def test_empty_query(self):
        """Test empty query parsing."""
        parser = QueryParser()
        result = parser.parse("")
        
        assert result.is_empty()
        assert not result.has_topics()
        assert not result.has_filters()
    
    def test_filters_only_query(self):
        """Test query with only filters, no terms."""
        parser = QueryParser()
        result = parser.parse("type:pdf tag:report")
        
        assert result.has_filters()


class TestSearchEndpoints:
    """Test search endpoint integration behaviour with patched accessor."""

    def _fake_result(self, *, collection_id: str | None = None):
        hit = SimpleNamespace(
            doc_id="doc-1",
            score=1.0,
            root_id="root-1",
            version_id="ver-1",
            collection_id=collection_id,
            title="Test",
            description="Test description",
            content="Test content",
            tags=[],
            highlights=None,
        )
        return SimpleNamespace(
            hits=[hit],
            total=1,
            parsed_query="parsed",
            applied_filters=[],
            corrections=[],
            ordering="semantic",
        )

    @pytest.mark.asyncio
    @patch("mantle.search.mantle.wiring.build_sse_search_accessor")
    async def test_search_with_scope(self, MockAccessor, client: AsyncClient):
        """Search with scope narrows to specified containers."""
        MockAccessor.return_value.search.return_value = self._fake_result(collection_id="col-9")

        resp = await client.post(
            "/artifacts/recall",
            headers={"Authorization": "Bearer fake-token"},
            json={"query_text": "docs", "scope": ["col-9"]},
        )

        assert resp.status_code == 200

    @pytest.mark.asyncio
    @patch("mantle.search.mantle.wiring.build_sse_search_accessor")
    async def test_search_default_returns_results(self, MockAccessor, client: AsyncClient):
        """Default search returns results."""
        MockAccessor.return_value.search.return_value = self._fake_result(collection_id="col-1")

        resp = await client.post(
            "/artifacts/recall",
            headers={"Authorization": "Bearer fake-token"},
            json={"query_text": "docs"},
        )

        assert resp.status_code == 200
        assert len(resp.json()["hits"]) == 1

    @pytest.mark.asyncio
    async def test_search_rejects_empty_query(self, client: AsyncClient):
        """Empty query_text is rejected."""
        resp = await client.post(
            "/artifacts/recall",
            headers={"Authorization": "Bearer fake-token"},
            json={"query_text": ""},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    @patch("mantle.search.mantle.wiring.build_sse_search_accessor")
    async def test_search_response_shape(self, MockAccessor, client: AsyncClient):
        """Response includes expected metadata fields."""
        MockAccessor.return_value.search.return_value = self._fake_result(collection_id="col-1")

        resp = await client.post(
            "/artifacts/recall",
            headers={"Authorization": "Bearer fake-token"},
            json={"query_text": "docs"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert "hits" in body
        assert "total" in body
        assert "query_text" in body
