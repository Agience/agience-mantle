import json
from pathlib import Path

import pytest

from mantle.main import app
from mantle.services import types_service
from mantle.services.dependencies import AuthContext, get_auth


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_resolve_exact_with_inheritance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Arrange: create a minimal types tree with wildcard parent and exact child.
    root = tmp_path / "types"

    _write_json(
        root / "text" / "_wildcard" / "type.json",
        {"content_type": "text/*", "version": 1},
    )
    _write_json(
        root / "text" / "_wildcard" / "preview.json",
        {"version": 1, "icon": "file-text", "preview": {"kind": "text_excerpt"}},
    )

    _write_json(
        root / "text" / "plain" / "type.json",
        {"content_type": "text/plain", "version": 1, "inherits": ["text/*"]},
    )
    _write_json(
        root / "text" / "plain" / "preview.json",
        {"version": 1, "icon": "note", "preview": {"max_chars": 12}},
    )

    monkeypatch.setenv("AGIENCE_TYPES_PATHS", str(root))
    monkeypatch.setenv("AGIENCE_TYPES_DISABLE_BUILTIN", "1")
    monkeypatch.setattr(types_service, "_default_server_ui_roots", lambda: [])

    # Act
    res = types_service.resolve_type_definition("text/plain; charset=utf-8")

    # Assert
    assert res is not None
    assert res.content_type == "text/plain"

    preview = res.definition.get("preview")
    assert isinstance(preview, dict)

    # icon overridden by child
    assert preview.get("icon") == "note"

    # inherited kind from parent
    assert preview.get("preview", {}).get("kind") == "text_excerpt"

    # child-only field present
    assert preview.get("preview", {}).get("max_chars") == 12


def test_resolve_falls_back_to_wildcard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "types"

    _write_json(
        root / "text" / "_wildcard" / "type.json",
        {"content_type": "text/*", "version": 1},
    )
    _write_json(
        root / "text" / "_wildcard" / "preview.json",
        {"version": 1, "icon": "file-text"},
    )

    monkeypatch.setenv("AGIENCE_TYPES_PATHS", str(root))
    monkeypatch.setenv("AGIENCE_TYPES_DISABLE_BUILTIN", "1")
    monkeypatch.setattr(types_service, "_default_server_ui_roots", lambda: [])

    res = types_service.resolve_type_definition("text/csv")
    assert res is not None
    assert res.content_type == "text/csv"
    assert res.definition.get("preview", {}).get("icon") == "file-text"


# `routers/types_router.py` does not exist — the `/types/*` HTTP surface is unmounted, and
# type resolution/registration behaviour is covered in-process by the tests in this file instead.
# `types_service._builtin_types_root()` and the `AGIENCE_TYPES_ROOT` / `AGIENCE_TYPES_DISABLE_BUILTIN`
# env hooks do not exist either: `get_types_roots()` reads exactly one env var
# (`AGIENCE_TYPES_PATHS`) and has no code path that reaches a persona `ui/` tree — Mantle does not
# scan a server's filesystem for server-owned types.


def test_resolve_capability_target_from_handler_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "types"

    _write_json(
        root / "application" / "vnd.test+json" / "type.json",
        {"content_type": "application/vnd.test+json", "version": 1},
    )
    _write_json(
        root / "application" / "vnd.test+json" / "handlers" / "extract_text.json",
        {"capability": "extract_text", "tool": "extract_text"},
    )

    monkeypatch.setenv("AGIENCE_TYPES_PATHS", str(root))
    monkeypatch.setenv("AGIENCE_TYPES_DISABLE_BUILTIN", "1")

    target = types_service.resolve_capability_target("application/vnd.test+json", "extract_text")
    assert target == "extract_text"


def test_resolve_event_target_from_behaviors_handler_ref(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "types"

    _write_json(
        root / "application" / "vnd.test+json" / "type.json",
        {"content_type": "application/vnd.test+json", "version": 1},
    )
    _write_json(
        root / "application" / "vnd.test+json" / "behaviors.json",
        {"version": 1, "events": {"on_commit": {"handler": "handlers/on_commit.json"}}},
    )
    _write_json(
        root / "application" / "vnd.test+json" / "handlers" / "on_commit.json",
        {
            "capability": "on_commit",
            "implementation": {"kind": "mcp-tool", "tool": "on_commit"},
        },
    )

    monkeypatch.setenv("AGIENCE_TYPES_PATHS", str(root))
    monkeypatch.setenv("AGIENCE_TYPES_DISABLE_BUILTIN", "1")

    target = types_service.resolve_event_target("application/vnd.test+json", "on_commit")
    assert target == "on_commit"


def test_resolve_event_binding_prefers_event_server_over_handler_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "types"

    _write_json(
        root / "application" / "vnd.test+json" / "type.json",
        {"content_type": "application/vnd.test+json", "version": 1},
    )
    _write_json(
        root / "application" / "vnd.test+json" / "behaviors.json",
        {
            "version": 1,
            "events": {
                "on_commit": {
                    "handler": "handlers/on_commit.json",
                    "server": "event-server",
                }
            },
        },
    )
    _write_json(
        root / "application" / "vnd.test+json" / "handlers" / "on_commit.json",
        {
            "capability": "on_commit",
            "implementation": {
                "kind": "mcp-tool",
                "tool": "on_commit",
                "server": "handler-server",
            },
        },
    )

    monkeypatch.setenv("AGIENCE_TYPES_PATHS", str(root))
    monkeypatch.setenv("AGIENCE_TYPES_DISABLE_BUILTIN", "1")

    binding = types_service.resolve_event_binding("application/vnd.test+json", "on_commit")
    assert binding == {"tool": "on_commit", "server_artifact_id": "event-server"}


def test_resolve_event_binding_supports_direct_tool_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "types"

    _write_json(
        root / "application" / "vnd.test+json" / "type.json",
        {"content_type": "application/vnd.test+json", "version": 1},
    )
    _write_json(
        root / "application" / "vnd.test+json" / "behaviors.json",
        {
            "version": 1,
            "events": {
                "on_commit": {
                    "tool": "direct_commit_tool",
                    "server_artifact_id": "direct-server",
                }
            },
        },
    )

    monkeypatch.setenv("AGIENCE_TYPES_PATHS", str(root))
    monkeypatch.setenv("AGIENCE_TYPES_DISABLE_BUILTIN", "1")

    binding = types_service.resolve_event_binding("application/vnd.test+json", "on_commit")
    assert binding == {"tool": "direct_commit_tool", "server_artifact_id": "direct-server"}


def test_resolve_type_definition_reports_event_validation_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "types"

    _write_json(
        root / "application" / "vnd.invalid+json" / "type.json",
        {"content_type": "application/vnd.invalid+json", "version": 1},
    )
    _write_json(
        root / "application" / "vnd.invalid+json" / "behaviors.json",
        {
            "version": 1,
            "events": {
                "on_commit": {
                    "tool": "commit_tool",
                    "handler": "handlers/on_commit.json",
                },
                "on_publish": {
                    "handler": "handlers/missing.json",
                },
            },
        },
    )
    _write_json(
        root / "application" / "vnd.invalid+json" / "handlers" / "on_commit.json",
        {
            "capability": "on_commit",
            "implementation": {"kind": "mcp-tool", "tool": "commit_tool"},
        },
    )

    monkeypatch.setenv("AGIENCE_TYPES_PATHS", str(root))
    monkeypatch.setenv("AGIENCE_TYPES_DISABLE_BUILTIN", "1")

    res = types_service.resolve_type_definition("application/vnd.invalid+json")
    assert res is not None
    assert any("only one of 'tool' or 'handler'" in msg for msg in res.validation_errors)
    assert any("references missing handler 'missing'" in msg for msg in res.validation_errors)


def test_get_field_index_hints_extracts_per_field_hints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "types"

    _write_json(
        root / "application" / "vnd.example+json" / "type.json",
        {
            "content_type": "application/vnd.example+json",
            "version": 1,
            "context_schema": {
                "title": {"index": ["lexical"]},
                "description": {"index": ["lexical", "semantic"]},
                "offers": {"index": ["semantic"]},
                "location": {"index": ["geo"]},
                "price": {"index": ["numeric"]},
                "no_hints": {"type": "string"},
                "bad_hint": {"index": ["bogus"]},
                "free_form_string": "string — some prose",
            },
        },
    )

    monkeypatch.setenv("AGIENCE_TYPES_PATHS", str(root))
    monkeypatch.setenv("AGIENCE_TYPES_DISABLE_BUILTIN", "1")
    monkeypatch.setattr(types_service, "_default_server_ui_roots", lambda: [])
    types_service.invalidate_type_cache()

    hints = types_service.get_field_index_hints("application/vnd.example+json")

    assert hints == {
        "title": ["lexical"],
        "description": ["lexical", "semantic"],
        "offers": ["semantic"],
        "location": ["geo"],
        "price": ["numeric"],
    }
    assert "no_hints" not in hints
    assert "free_form_string" not in hints
    assert "bad_hint" not in hints  # unknown hint kinds dropped silently


def test_get_field_index_hints_returns_empty_when_no_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "types"

    _write_json(
        root / "application" / "vnd.bare+json" / "type.json",
        {"content_type": "application/vnd.bare+json", "version": 1},
    )

    monkeypatch.setenv("AGIENCE_TYPES_PATHS", str(root))
    monkeypatch.setenv("AGIENCE_TYPES_DISABLE_BUILTIN", "1")
    monkeypatch.setattr(types_service, "_default_server_ui_roots", lambda: [])
    types_service.invalidate_type_cache()

    assert types_service.get_field_index_hints("application/vnd.bare+json") == {}
    assert types_service.get_field_index_hints("application/vnd.unknown+json") == {}


# ---------------------------------------------------------------------------
# Canonical-source invariants — no duplicate / shadow definitions in the repo.
#
# Type resolution is an override chain: a core (`package/types`) base, overlaid
# with the server-owned definition (self-registered at runtime; the owning server
# still OWNS and ships its `type.json` files, it just pushes them rather than
# having Mantle scan them). For overlays to be unambiguous, a content type's CANONICAL
# definition (a `type.json` that DECLARES a `content_type`) must live in exactly
# ONE place, and that file must sit at the folder its `content_type` names.
# Higher layers may add override-ONLY overlays (a `type.json` with no declared
# `content_type`). The guards below hold that property over the canonical tree —
# every declared `content_type` sits where its folder says, and every file parses
# without a BOM — which is the half Mantle's own resolution depends on. An overlay
# shipped by a persona is checked in the repository that ships it; a scan reaching
# across repositories asserts against files this one cannot fix.
# ---------------------------------------------------------------------------

from ._package_root import types_root  # noqa: E402 — single source
_CANONICAL_ROOT = types_root()


def _content_type_for_folder(type_json_path: Path, root: Path) -> str:
    """Inverse of `_content_type_to_rel_folder`: derive the content type a
    `type.json` is resolved under from its `<top>/<sub>` folder beneath a root
    (`_wildcard` ↦ `*`)."""
    top, sub = type_json_path.parent.relative_to(root).parts
    return f"{top}/{'*' if sub == '_wildcard' else sub}"


def _all_type_files() -> list[tuple[Path, Path]]:
    """Every checked-in `type.json` under the canonical type root, paired with that root.

    ONE root, and it is resolved through `_package_root.types_root()` — the module that owns
    where the split seeds/types tree lives, and the same answer `conftest.py`'s skip gate acts
    on. A root spelled out here by hand would be a second answer to that question, and a
    hand-spelled path that stops resolving scans an empty directory and asserts nothing, which
    is why the callers below check the count instead of trusting it.

    Persona `ui/` overlays live in their own repositories and are guarded there; this scan
    speaks for the canonical definitions only, which is the invariant Mantle resolves against.
    """
    out: list[tuple[Path, Path]] = []
    for tj in sorted(_CANONICAL_ROOT.rglob("type.json")):
        out.append((tj, _CANONICAL_ROOT))
    return out


def _measured_type_files() -> list[tuple[Path, Path]]:
    """`_all_type_files()`, refusing to answer with nothing.

    Every guard below is a scan, and a scan over an empty set passes while measuring nothing.
    The tree is a precondition of these tests (`conftest.py` skips them when it is absent), so
    reaching one of them with zero files is a broken root rather than a repository with no types.
    """
    found = _all_type_files()
    assert found, (
        "no type.json found under the canonical type root %s — this scan measures nothing and "
        "every assertion built on it passes vacuously. Fix the root (`_package_root.types_root`), "
        "do not relax the guard." % _CANONICAL_ROOT
    )
    return found


# There is one optional filesystem root (`AGIENCE_TYPES_PATHS`) plus the runtime overlay, which is
# keyed by content type in a dict — a second registration replaces the first (documented upsert,
# `register_runtime_type`), so two canonical definitions for the same content type cannot coexist.
# `test_declared_content_type_matches_folder_location` below covers the remaining invariant: a
# `type.json` that declares a `content_type` must sit at the folder that content type names.


def test_declared_content_type_matches_folder_location():
    """When a ``type.json`` declares a ``content_type`` it must equal the type its
    folder resolves under. Override-only overlays (no declared ``content_type``)
    are exempt — they inherit identity from the folder + the core base."""
    mismatches = []
    for tj, root in _measured_type_files():
        declared = json.loads(tj.read_text(encoding="utf-8-sig")).get("content_type")
        if not declared:
            continue  # override-only overlay
        expected = _content_type_for_folder(tj, root)
        if declared != expected:
            mismatches.append(f"  {tj.relative_to(root).as_posix()}: declares {declared!r}, folder says {expected!r}")
    assert not mismatches, "content_type does not match folder location:\n" + "\n".join(mismatches)


# Core platform types (`authority`/`resource`/`prompt`/`collection`/`workspace`) do not resolve
# with no configuration — there is no default filesystem root. Their owning server pushes them via
# `register_runtime_type`, and the tests below exercise that path: the overlay merging onto an
# explicitly-supplied filesystem base, and the lazy loader fetching an unregistered type from its
# artifact.


# ---------------------------------------------------------------------------
# Self-registration overlay — servers push the types they own; resolution overlays them on top of
# whatever filesystem base was supplied (and applies inherits).
#
# The three tests below supply the base explicitly (`AGIENCE_TYPES_PATHS` → the package types tree):
# naming the root is the only way to get a filesystem base, and the mechanism under test is the
# runtime overlay, so it is tested over a base that is actually there.
# ---------------------------------------------------------------------------


@pytest.fixture
def _clean_runtime_types():
    types_service.clear_runtime_types()
    types_service.invalidate_type_cache()
    yield
    types_service.clear_runtime_types()
    types_service.invalidate_type_cache()


def test_runtime_full_type_overlays_and_applies_inherits(_clean_runtime_types, monkeypatch):
    """A server-owned type (no filesystem base of its own) resolves from its pushed
    definition and still inherits its declared parent (a supplied base type)."""
    monkeypatch.setenv("AGIENCE_TYPES_PATHS", str(types_root()))
    types_service.invalidate_type_cache()

    types_service.register_runtime_type(
        "application/vnd.agience.demo+json",
        {
            "content_type": "application/vnd.agience.demo+json",
            "inherits": ["application/json"],
            "ui": {"label": "Demo", "viewer": "json", "resource_uri": "ui://demo/x.html"},
        },
        "chorus/test",
    )

    res = types_service.resolve_type_definition_cached("application/vnd.agience.demo+json")
    assert res is not None
    assert res.definition["ui"]["resource_uri"] == "ui://demo/x.html"
    # inherits applied: the core application/json base contributed to sources.
    assert any("application" in str(s) and "json" in str(s) for s in res.sources)

    # Upsert: a re-push (server restart / edit) replaces the prior overlay.
    types_service.register_runtime_type(
        "application/vnd.agience.demo+json",
        {"content_type": "application/vnd.agience.demo+json", "ui": {"label": "Demo v2", "viewer": "json"}},
        "chorus/test",
    )
    res2 = types_service.resolve_type_definition_cached("application/vnd.agience.demo+json")
    assert res2 is not None
    assert res2.definition["ui"]["label"] == "Demo v2"


def test_runtime_overlay_merges_onto_core_base(_clean_runtime_types, monkeypatch):
    """A partial overlay (just a viewer pointer) merges onto a filesystem base type —
    the base is preserved and the overlay wins on the keys it sets."""
    monkeypatch.setenv("AGIENCE_TYPES_PATHS", str(types_root()))
    types_service.invalidate_type_cache()

    types_service.register_runtime_type(
        "application/vnd.agience.collection+json",
        {"ui": {"resource_uri": "ui://test/collection.html", "resource_server": "test"}},
        "chorus/test",
    )

    res = types_service.resolve_type_definition_cached("application/vnd.agience.collection+json")
    assert res is not None
    # overlay applied
    assert res.definition["ui"]["resource_uri"] == "ui://test/collection.html"
    # core base preserved (collection declares operations in package/types)
    assert isinstance(res.definition.get("operations"), dict)
    assert "commit" in res.definition["operations"]
    canonical = types_root().resolve()
    assert any(canonical in Path(s).resolve().parents or Path(s).resolve() == canonical for s in res.sources)


def test_lazy_loader_resolves_third_party_type_on_miss(_clean_runtime_types, monkeypatch):
    """A third-party type is not pre-registered: on a resolution miss the lazy
    loader fetches it on demand (and the result is cached so it fires once). A type
    present on the supplied filesystem base never triggers the loader (no DB hit)."""
    monkeypatch.setenv("AGIENCE_TYPES_PATHS", str(types_root()))
    types_service.invalidate_type_cache()
    calls: list[str] = []

    def loader(content_type: str) -> bool:
        calls.append(content_type)
        if content_type == "application/vnd.thirdparty.crm+json":
            types_service.register_runtime_type(
                content_type,
                {"content_type": content_type, "ui": {"label": "CRM", "viewer": "record"}},
                "app:crm",
            )
            return True
        return False

    types_service.set_lazy_type_loader(loader)

    res = types_service.resolve_type_definition_cached("application/vnd.thirdparty.crm+json")
    assert res is not None
    assert res.definition["ui"]["label"] == "CRM"

    # Cached → the loader fires at most once for the resolved type.
    types_service.resolve_type_definition_cached("application/vnd.thirdparty.crm+json")
    assert calls.count("application/vnd.thirdparty.crm+json") == 1

    # A type on the supplied filesystem base resolves without ever touching the lazy loader.
    res_core = types_service.resolve_type_definition_cached("application/vnd.agience.collection+json")
    assert res_core is not None
    assert "application/vnd.agience.collection+json" not in calls






def test_no_type_json_carries_a_utf8_bom():
    """Keep every committed type.json BOM-free. Mantle reads `utf-8-sig` and FACET's
    build strips a BOM too, but a stray BOM is a latent trap — it parses server-side
    while silently dropping the type on the frontend. Guard the files so neither
    resolver has to compensate."""
    bom = b"\xef\xbb\xbf"
    offenders = [
        tj.relative_to(root).as_posix()
        for tj, root in _measured_type_files()
        if tj.read_bytes().startswith(bom)
    ]
    assert not offenders, "type.json file(s) start with a UTF-8 BOM:\n" + "\n".join(
        f"  {p}" for p in offenders
    )
