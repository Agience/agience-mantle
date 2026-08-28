"""Content extraction utilities for the ingest runner agent.

Downloads S3 content and extracts text for indexing and chunking.
"""

from __future__ import annotations

import logging
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

from mantle.services.content_service import generate_signed_url
from mantle.services.types_service import resolve_capability_target

logger = logging.getLogger(__name__)

# Maximum bytes to download for text extraction
MAX_DOWNLOAD_BYTES = 10_000_000  # 10 MB

# MIME prefixes/types where we can extract raw text
_TEXT_EXTRACTABLE = {
    "text/",
    "application/json",
    "application/xml",
    "application/javascript",
    "application/x-yaml",
    "application/yaml",
    "application/ld+json",
    "application/xhtml+xml",
}


def infer_extraction_handler(content_type: str) -> Optional[str]:
    """Return the handler key required to derive searchable content for *content_type*."""
    if not content_type:
        return None

    # Prefer type-local capability contracts when present.
    declared = resolve_capability_target(content_type, "extract_text")
    if declared:
        return declared

    # Keep a generic fallback when no contract is declared for a MIME.
    return "extract-content"


def describe_content_processing(
    content_type: str,
    *,
    has_inline_content: bool = False,
    upload_complete: bool = True,
) -> dict:
    """Describe whether content can be indexed immediately or needs a handler."""
    deterministic = has_inline_content or is_text_extractable(content_type)
    handler = None if deterministic else infer_extraction_handler(content_type)

    if not upload_complete and not has_inline_content:
        return {
            "strategy": "deterministic" if deterministic else "handler",
            "handler": handler,
            "asset_status": "uploading",
            "content_status": "pending_upload",
            "index_status": "pending_upload",
            "status": "pending_upload",
        }

    if deterministic:
        return {
            "strategy": "deterministic",
            "handler": None,
            "asset_status": "available",
            "content_status": "available",
            "index_status": "ready",
            "status": "ready",
        }

    return {
        "strategy": "handler",
        "handler": handler,
        "asset_status": "available",
        "content_status": "pending_handler",
        "index_status": "pending_handler",
        "status": "pending_handler",
    }


def is_text_extractable(content_type: str) -> bool:
    """Return True if we can extract text directly from this content type."""
    if not content_type:
        return False
    content_type = content_type.lower().split(";")[0].strip()
    if content_type.startswith("text/"):
        return True
    return content_type in _TEXT_EXTRACTABLE


def extract_text_from_s3(
    content_key: str,
    content_type: str,
    filename: Optional[str] = None,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
) -> Optional[str]:
    """Download an S3 object via signed URL and extract text content.

    Returns the extracted text, or None if extraction isn't possible.
    """
    if not is_text_extractable(content_type):
        logger.info("Content type '%s' not text-extractable, skipping download", content_type)
        return None

    url = generate_signed_url(content_key, filename=filename, content_type=content_type, as_attachment=False)
    if not url:
        logger.warning("Could not generate signed URL for key=%s", content_key)
        return None

    try:
        req = Request(url)
        with urlopen(req, timeout=30) as resp:
            raw = resp.read(max_bytes)
    except (URLError, OSError) as exc:
        logger.warning("Failed to download content for key=%s: %s", content_key, exc)
        return None

    # Detect encoding from Content-Type header or default to utf-8
    encoding = "utf-8"
    content_type_lower = content_type.lower()
    if "charset=" in content_type_lower:
        charset_part = content_type_lower.split("charset=")[-1].split(";")[0].strip()
        encoding = charset_part or "utf-8"

    try:
        text = raw.decode(encoding, errors="replace")
    except (LookupError, UnicodeDecodeError):
        text = raw.decode("utf-8", errors="replace")

    return text if text.strip() else None


def extract_text_from_artifact(artifact) -> Optional[str]:
    """Extract text from an artifact's content or S3 storage.

    Prefers inline content; falls back to S3 download for binary uploads.
    """
    import json

    # Check inline content first
    content = getattr(artifact, "content", None) or ""
    if content.strip():
        return content

    # Parse context for S3 metadata
    raw_ctx = getattr(artifact, "context", None) or ""
    try:
        ctx = json.loads(raw_ctx) if isinstance(raw_ctx, str) else (raw_ctx or {})
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(ctx, dict):
        return None

    # ── the LOCAL tier first: the object store is where a ref is promoted TO ────────────────
    # `workspace_service._store_content_in_s3` states the contract this had backwards: "One
    # destination. Content lives at `cas/<sha256 of the plaintext>` ... the object store is where
    # the ref gets promoted to rather than an alternative place to put the bytes." An artifact
    # therefore carries BOTH a `content_ref` (the local address) and a `content_key` (the
    # promotion address), and this function read only the second.
    #
    # Measured 2026-08-25 on 71/home, which has no object-store credentials — a supported
    # configuration, per `content_service.local_content_tier`: "a node whose keys volume is not
    # mounted still has an object store to fall back to". The reverse is equally allowed and is
    # what this node is. Every one of the 197 memory-lane captures had its blob present on local
    # disk, and 116 of them failed to index with
    #
    #     ContentUrlSigningError: could not generate a signed content URL for artifacts/<id>.content
    #     ... botocore NoCredentialsError
    #
    # So the body was two directories away and the indexer went to the internet for it.
    cas_ref = getattr(artifact, "content_ref", None)
    if cas_ref:
        try:
            from mantle.db.doc_boundary import CONTENT_KEY_PRINCIPAL, content_key_scope
            from mantle.services.content_service import get_bytes_decrypted

            doc = artifact.to_dict() if hasattr(artifact, "to_dict") else {}
            principal = (doc.get(CONTENT_KEY_PRINCIPAL)
                         or getattr(artifact, "created_by", None))
            if principal:
                return get_bytes_decrypted(
                    "", principal, cas_ref=cas_ref,
                    collection_id=content_key_scope(doc) if doc else None,
                ).decode("utf-8")
        except Exception:  # noqa: BLE001 — fall through to the promotion address below
            logger.debug("local CAS read failed for %s; trying the object store",
                         getattr(artifact, "id", None), exc_info=True)

    content_key = ctx.get("content_key")
    if not content_key:
        return None

    content_type = ctx.get("content_type") or ""
    filename = ctx.get("filename")
    return extract_text_from_s3(content_key, content_type, filename=filename)
