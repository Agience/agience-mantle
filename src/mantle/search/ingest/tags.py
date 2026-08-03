# search/ingest/tags.py
import logging
import re
from typing import List

logger = logging.getLogger(__name__)


def normalize_tag(tag: str) -> str:
    """
    Normalize a single tag to canonical form.
    
    - Lowercase
    - Strip whitespace
    - Remove special characters (keep alphanumeric, hyphens, underscores)
    """
    if not tag:
        return ""
    
    # Lowercase and strip
    normalized = tag.lower().strip()
    
    # Keep only alphanumeric, hyphens, underscores, spaces
    normalized = re.sub(r"[^a-z0-9\-_\s]", "", normalized)
    
    # Collapse multiple spaces
    normalized = re.sub(r"\s+", " ", normalized)
    
    return normalized.strip()


def normalize_tags(tags: List[str]) -> List[str]:
    """
    Normalize a list of tags to canonical form.
    
    Returns unique, sorted list of normalized tags.
    """
    if not tags:
        return []
    
    normalized = set()
    for tag in tags:
        if isinstance(tag, str) and tag.strip():
            norm = normalize_tag(tag)
            if norm:
                normalized.add(norm)
    
    return sorted(normalized)


def parse_tags_from_context(context_str: str) -> List[str]:
    """
    Extract tags from artifact context JSON.
    
    Returns list of tag strings.
    """
    import json
    
    # Same defect as `chunking.extract_text_from_context` -- see the note there.
    if context_str is None or (isinstance(context_str, str) and not context_str.strip()):
        return []
    if not context_str:
        return []
    
    # ⛔ NON-JSON CONTEXT IS PROSE, NOT AN ERROR — see the note in `chunking.extract_text_from_context`.
    # Measured: 48 of 500 live vertices carry a bare string here. Unlike the text extractor there is
    # genuinely NOTHING to recover for TAGS from free prose (inventing tags by splitting a sentence
    # would be adding words the artifact never carried, the other half of the same law), so this
    # returns no tags — but it does so SILENTLY, because "this artifact has no tags" is an ordinary
    # reading, not a fault worth a warning per artifact per ingest.
    if isinstance(context_str, str):
        try:
            context = json.loads(context_str)
        except json.JSONDecodeError:
            return []
    else:
        context = context_str
    if not isinstance(context, dict):
        return []

    try:
        tags = context.get("tags", [])
        
        if isinstance(tags, list):
            return [str(t) for t in tags if t]
        elif isinstance(tags, str):
            # Split comma-separated tags
            return [t.strip() for t in tags.split(",") if t.strip()]
        
        return []
    except (json.JSONDecodeError, AttributeError, TypeError) as e:
        logger.warning(f"Failed to parse tags from context: {e}")
        return []


def extract_metadata_from_context(context_str: str) -> dict:
    """
    Extract searchable metadata from artifact context JSON.
    
    Returns dict with fields like:
    - doc_type
    - content_type
    - filename
    - amount
    - score
    - pii
    """
    import json
    
    metadata = {}
    
    # Third instance of the same type-assuming guard -- see chunking.extract_text_from_context.
    if context_str is None or (isinstance(context_str, str) and not context_str.strip()):
        return metadata
    if not context_str:
        return metadata
    
    try:
        context = json.loads(context_str) if isinstance(context_str, str) else context_str
        
        # Extract known metadata fields
        if "content_type" in context:
            metadata["content_type"] = context["content_type"]
        
        if "filename" in context:
            metadata["filename"] = context["filename"]
        
        if "doc_type" in context:
            metadata["doc_type"] = context["doc_type"]
        
        # Numeric fields
        if "amount" in context:
            try:
                metadata["amount"] = float(context["amount"])
            except (ValueError, TypeError):
                pass
        
        if "score" in context:
            try:
                metadata["score"] = float(context["score"])
            except (ValueError, TypeError):
                pass
        
        # Boolean fields
        if "pii" in context:
            metadata["pii"] = bool(context["pii"])
        
    except (json.JSONDecodeError, AttributeError, TypeError) as e:
        logger.warning(f"Failed to extract metadata from context: {e}")
    
    return metadata
