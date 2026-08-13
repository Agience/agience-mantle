"""Tests for ingest runner service utilities."""

import json
from unittest.mock import MagicMock, patch


class FakeArtifact:
    _next_id = 0

    def __init__(self, id=None, context_dict=None, content=""):
        if id is None:
            FakeArtifact._next_id += 1
            id = f"art-{FakeArtifact._next_id}"
        self.id = id
        self.context = json.dumps(context_dict or {})
        self.content = content
        self.collection_id = "ws-1"
        self.state = "draft"

    def to_dict(self):
        return {"id": self.id, "context": self.context, "content": self.content}


class TestIngestRunnerService:

    def test_is_text_extractable_text_types(self):
        from mantle.services.ingest_runner_service import is_text_extractable
        assert is_text_extractable("text/plain")
        assert is_text_extractable("text/html")
        assert is_text_extractable("text/csv")
        assert is_text_extractable("application/json")
        assert is_text_extractable("application/xml")

    def test_is_text_extractable_binary_types(self):
        from mantle.services.ingest_runner_service import is_text_extractable
        assert not is_text_extractable("application/pdf")
        assert not is_text_extractable("image/png")
        assert not is_text_extractable("audio/mpeg")
        assert not is_text_extractable("")

    def test_describe_content_processing_for_deterministic_text(self):
        from mantle.services.ingest_runner_service import describe_content_processing

        result = describe_content_processing("text/plain", upload_complete=True)

        assert result["strategy"] == "deterministic"
        assert result["handler"] is None
        assert result["content_status"] == "available"
        assert result["index_status"] == "ready"

    def test_describe_content_processing_for_audio_handler(self):
        from mantle.services.ingest_runner_service import describe_content_processing

        result = describe_content_processing("audio/mpeg", upload_complete=True)

        assert result["strategy"] == "handler"
        # Absent a type-local capability contract, every non-text MIME resolves to
        # the single "extract-content" handler.
        assert result["handler"] == "extract-content"
        assert result["content_status"] == "pending_handler"
        assert result["index_status"] == "pending_handler"

    def test_infer_extraction_handler_falls_back_to_unified_audio(self):
        from mantle.services.ingest_runner_service import infer_extraction_handler

        assert infer_extraction_handler("audio/mpeg") == "extract-content"

    def test_infer_extraction_handler_falls_back_to_unified_image(self):
        from mantle.services.ingest_runner_service import infer_extraction_handler

        assert infer_extraction_handler("image/png") == "extract-content"

    def test_infer_extraction_handler_falls_back_to_unified_pdf(self):
        from mantle.services.ingest_runner_service import infer_extraction_handler

        assert infer_extraction_handler("application/pdf") == "extract-content"

    @patch("mantle.services.ingest_runner_service.resolve_capability_target", return_value="custom_extract_text")
    def test_infer_extraction_handler_prefers_contract(self, _mock_resolve):
        from mantle.services.ingest_runner_service import infer_extraction_handler

        assert infer_extraction_handler("audio/mpeg") == "custom_extract_text"

    def test_extract_text_from_artifact_inline(self):
        from mantle.services.ingest_runner_service import extract_text_from_artifact
        artifact = FakeArtifact(content="Hello world")
        assert extract_text_from_artifact(artifact) == "Hello world"

    def test_extract_text_from_artifact_empty_inline_with_s3(self):
        from mantle.services.ingest_runner_service import extract_text_from_artifact
        artifact = FakeArtifact(
            content="",
            context_dict={"content_key": "tenant/abc.content", "content_type": "text/plain"},
        )
        with patch("mantle.services.ingest_runner_service.extract_text_from_s3", return_value="S3 content") as mock:
            result = extract_text_from_artifact(artifact)
            assert result == "S3 content"
            mock.assert_called_once_with("tenant/abc.content", "text/plain", filename=None)

    def test_extract_text_from_artifact_no_content_key(self):
        from mantle.services.ingest_runner_service import extract_text_from_artifact
        artifact = FakeArtifact(content="", context_dict={"content_type": "text/plain"})
        assert extract_text_from_artifact(artifact) is None

    @patch("mantle.services.ingest_runner_service.generate_signed_url", return_value="https://s3.example.com/obj")
    @patch("mantle.services.ingest_runner_service.urlopen")
    def test_extract_text_from_s3_success(self, mock_urlopen, mock_sign):
        from mantle.services.ingest_runner_service import extract_text_from_s3

        mock_resp = MagicMock()
        mock_resp.read.return_value = b"Downloaded text content"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = extract_text_from_s3("tenant/abc.content", "text/plain")
        assert result == "Downloaded text content"

    def test_extract_text_from_s3_binary_content_type_skipped(self):
        from mantle.services.ingest_runner_service import extract_text_from_s3
        result = extract_text_from_s3("tenant/abc.content", "application/pdf")
        assert result is None

    def test_extract_text_from_artifact_uses_content_type_field(self):
        """context uses 'content_type' for the content type field."""
        from mantle.services.ingest_runner_service import extract_text_from_artifact
        artifact = FakeArtifact(
            content="",
            context_dict={"content_key": "tenant/xyz.content", "content_type": "text/markdown"},
        )
        with patch("mantle.services.ingest_runner_service.extract_text_from_s3", return_value="# Hello") as mock:
            result = extract_text_from_artifact(artifact)
            assert result == "# Hello"
            mock.assert_called_once_with("tenant/xyz.content", "text/markdown", filename=None)

    def test_extract_text_from_artifact_ignores_legacy_mime_field(self):
        """Only content_type is read; the legacy mime field is not."""
        from mantle.services.ingest_runner_service import extract_text_from_artifact
        artifact = FakeArtifact(
            content="",
            context_dict={
                "content_key": "tenant/both.content",
                "content_type": "text/html",
            },
        )
        with patch("mantle.services.ingest_runner_service.extract_text_from_s3", return_value="<p>hi</p>") as mock:
            result = extract_text_from_artifact(artifact)
            assert result == "<p>hi</p>"
            mock.assert_called_once_with("tenant/both.content", "text/html", filename=None)

    def test_extract_text_from_artifact_no_content_type_returns_none(self):
        """Artifacts without content_type field are not text-extractable."""
        from mantle.services.ingest_runner_service import extract_text_from_artifact
        artifact = FakeArtifact(
            content="",
            context_dict={"content_key": "tenant/legacy.content"},
        )
        result = extract_text_from_artifact(artifact)
        assert result is None
