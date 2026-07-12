import json
from pathlib import Path

import pytest

from main import app
from services import types_service
from services.dependencies import AuthContext, get_auth


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


@pytest.mark.asyncio
async def test_router_resolve_uses_env_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client):
    root = tmp_path / "types"

    _write_json(
        root / "application" / "json" / "type.json",
        {"content_type": "application/json", "version": 1},
    )

    monkeypatch.setenv("AGIENCE_TYPES_PATHS", str(root))
    monkeypatch.setenv("AGIENCE_TYPES_DISABLE_BUILTIN", "1")
    monkeypatch.setattr(types_service, "_default_server_ui_roots", lambda: [])

    resp = await client.get("/types/resolve", params={"content_type": "application/json"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["content_type"] == "application/json"
    assert data["definition"]["type"]["content_type"] == "application/json"
    assert data["validation_errors"] == []


def test_get_types_roots_is_builtin_and_local_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """get_types_roots NO LONGER scans Chorus — it is core (``package/types``) plus
    any ``AGIENCE_TYPES_PATHS`` local override. Server-owned types are
    self-registered at runtime (overlay), never discovered on the filesystem, so a
    persona ``ui/`` tree must NOT appear among the roots."""
    builtin_root = tmp_path / "package" / "types"
    server_root = tmp_path / "src" / "chorus" / "astra" / "ui"

    _write_json(
        builtin_root / "application" / "json" / "type.json",
        {"content_type": "application/json", "version": 1},
    )
    _write_json(
        server_root / "application" / "vnd.agience.stream+json" / "type.json",
        {"content_type": "application/vnd.agience.stream+json", "version": 1},
    )

    monkeypatch.delenv("AGIENCE_TYPES_PATHS", raising=False)
    monkeypatch.delenv("AGIENCE_TYPES_DISABLE_BUILTIN", raising=False)
    monkeypatch.setattr(types_service, "_repo_root", lambda: tmp_path)

    roots = types_service.get_types_roots()

    assert roots == [builtin_root.resolve()]
    assert server_root.resolve() not in roots


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
# with the server-owned definition (self-registered at runtime; chorus still
# OWNS and ships its `type.json` files, it just pushes them rather than having
# Mantle scan them). For overlays to be unambiguous, a content type's CANONICAL
# definition (a `type.json` that DECLARES a `content_type`) must live in exactly
# ONE place, and that file must sit at the folder its `content_type` names.
# Higher layers may add override-ONLY overlays (a `type.json` with no declared
# `content_type`). These guards fail if a shadow canonical `type.json` (a persona
# overlay duplicating a `package/types/` canonical) is reintroduced.
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[3]
_CANONICAL_ROOT = _REPO / "package" / "types"


def _content_type_for_folder(type_json_path: Path, root: Path) -> str:
    """Inverse of `_content_type_to_rel_folder`: derive the content type a
    `type.json` is resolved under from its `<top>/<sub>` folder beneath a root
    (`_wildcard` ↦ `*`)."""
    top, sub = type_json_path.parent.relative_to(root).parts
    return f"{top}/{'*' if sub == '_wildcard' else sub}"


def _all_type_files() -> list[tuple[Path, Path]]:
    """Every checked-in `type.json` paired with the type-root it lives under
    (`package/types` + each persona's `src/chorus/<persona>/ui`)."""
    out: list[tuple[Path, Path]] = []
    roots = [_CANONICAL_ROOT, *sorted((_REPO / "src" / "chorus").glob("*/ui"))]
    for root in roots:
        for tj in sorted(root.rglob("type.json")):
            out.append((tj, root))
    return out


def test_no_canonical_type_definition_is_duplicated_across_roots():
    """A content type's CANONICAL definition — the ``type.json`` that DECLARES a
    ``content_type`` — lives in exactly one root. Higher roots MAY add override-
    only overlays (a ``type.json`` with no ``content_type``, e.g. a chorus viewer
    override onto a core base; "Open-with"-style). Those are overrides, not
    duplicates. What's forbidden is two roots both claiming to be canonical."""
    canonical: dict[str, list[str]] = {}
    for tj, root in _all_type_files():
        declared = json.loads(tj.read_text(encoding="utf-8-sig")).get("content_type")
        if not declared:
            continue  # override-only overlay — identity comes from the folder/base
        canonical.setdefault(declared, []).append(tj.relative_to(_REPO).as_posix())

    dups = {ct: locs for ct, locs in canonical.items() if len(locs) > 1}
    assert not dups, "content type(s) with more than one CANONICAL definition:\n" + "\n".join(
        f"  {ct}:\n" + "\n".join(f"      {p}" for p in locs) for ct, locs in sorted(dups.items())
    )


def test_declared_content_type_matches_folder_location():
    """When a ``type.json`` DECLARES a ``content_type`` it must equal the type its
    folder resolves under. Override-only overlays (no declared ``content_type``)
    are exempt — they inherit identity from the folder + the core base."""
    mismatches = []
    for tj, root in _all_type_files():
        declared = json.loads(tj.read_text(encoding="utf-8-sig")).get("content_type")
        if not declared:
            continue  # override-only overlay
        expected = _content_type_for_folder(tj, root)
        if declared != expected:
            mismatches.append(f"  {tj.relative_to(_REPO).as_posix()}: declares {declared!r}, folder says {expected!r}")
    assert not mismatches, "content_type does not match folder location:\n" + "\n".join(mismatches)


@pytest.mark.parametrize(
    "content_type",
    [
        "application/vnd.agience.authority+json",
        "application/vnd.agience.resource+json",
        "application/vnd.agience.prompt+json",
        "application/vnd.agience.collection+json",
        "application/vnd.agience.workspace+json",
    ],
)
def test_known_types_resolve_with_canonical_base(content_type, monkeypatch: pytest.MonkeyPatch):
    """Core types resolve with their ``package/types`` canonical definition as the
    base. Under the override model a chorus overlay may also contribute (so the
    canonical root is AMONG the sources, not necessarily the last) — but it must
    always be present as the base."""
    monkeypatch.delenv("AGIENCE_TYPES_PATHS", raising=False)
    monkeypatch.delenv("AGIENCE_TYPES_DISABLE_BUILTIN", raising=False)
    types_service.invalidate_type_cache()

    res = types_service.resolve_type_definition(content_type)
    assert res is not None, f"{content_type} did not resolve"
    canonical = _CANONICAL_ROOT.resolve()
    assert any(
        Path(s).resolve() == canonical or canonical in Path(s).resolve().parents
        for s in res.sources
    ), f"{content_type} sources {res.sources} — expected the canonical package/types base among them"


# ---------------------------------------------------------------------------
# Self-registration overlay — servers push the types they own; resolution
# overlays them on top of the core base (and applies inherits).
# ---------------------------------------------------------------------------


@pytest.fixture
def _clean_runtime_types():
    types_service.clear_runtime_types()
    types_service.invalidate_type_cache()
    yield
    types_service.clear_runtime_types()
    types_service.invalidate_type_cache()


def test_runtime_full_type_overlays_and_applies_inherits(_clean_runtime_types, monkeypatch):
    """A chorus-OWNED type (no core base of its own) resolves from its pushed
    definition and still inherits its declared parent (a core type)."""
    monkeypatch.delenv("AGIENCE_TYPES_PATHS", raising=False)
    monkeypatch.delenv("AGIENCE_TYPES_DISABLE_BUILTIN", raising=False)

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
    """A partial overlay (just a viewer pointer) merges onto a CORE type — the
    core base is preserved and the overlay wins on the keys it sets."""
    monkeypatch.delenv("AGIENCE_TYPES_PATHS", raising=False)
    monkeypatch.delenv("AGIENCE_TYPES_DISABLE_BUILTIN", raising=False)

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
    canonical = (_REPO / "package" / "types").resolve()
    assert any(canonical in Path(s).resolve().parents or Path(s).resolve() == canonical for s in res.sources)


def test_lazy_loader_resolves_third_party_type_on_miss(_clean_runtime_types):
    """A third-party type is NOT pre-registered: on a resolution MISS the lazy
    loader fetches it on demand (and the result is cached so it fires once). Core
    types never trigger the loader (no DB hit)."""
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

    # A core (filesystem) type resolves without ever touching the lazy loader.
    res_core = types_service.resolve_type_definition_cached("application/vnd.agience.collection+json")
    assert res_core is not None
    assert "application/vnd.agience.collection+json" not in calls


@pytest.mark.asyncio
async def test_register_endpoint_requires_service_identity(client):
    """Default auth in tests is a user principal → registration is refused."""
    resp = await client.post(
        "/types/register",
        json={"types": {"application/vnd.x+json": {"content_type": "application/vnd.x+json"}}},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_register_endpoint_registers_and_resolves(client, monkeypatch, _clean_runtime_types):
    """A first-party service principal can push types; they persist (mocked) and
    become resolvable through the runtime overlay."""
    app.dependency_overrides[get_auth] = lambda: AuthContext(
        principal_id="chorus", principal_type="service",
    )
    monkeypatch.setattr("services.type_registry_store.persist_type_definition", lambda *a, **k: None)

    payload = {
        "source": "chorus/test",
        "types": {
            "application/vnd.agience.demo2+json": {
                "content_type": "application/vnd.agience.demo2+json",
                "inherits": ["application/json"],
                "ui": {"label": "Demo2", "viewer": "json", "resource_uri": "ui://test/demo2.html"},
            }
        },
    }
    resp = await client.post("/types/register", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 1
    assert "application/vnd.agience.demo2+json" in body["registered"]

    res = types_service.resolve_type_definition_cached("application/vnd.agience.demo2+json")
    assert res is not None
    assert res.definition["ui"]["resource_uri"] == "ui://test/demo2.html"


def test_no_type_json_carries_a_utf8_bom():
    """Keep every committed type.json BOM-free. MANTLE reads `utf-8-sig` and
    FACET's build now strips a BOM too, but a stray BOM is a latent trap — it
    parses server-side yet (historically) silently dropped the type on the
    frontend. Guard the files so neither resolver has to compensate."""
    bom = b"\xef\xbb\xbf"
    offenders = [
        tj.relative_to(_REPO).as_posix()
        for tj, _root in _all_type_files()
        if tj.read_bytes().startswith(bom)
    ]
    assert not offenders, "type.json file(s) start with a UTF-8 BOM:\n" + "\n".join(
        f"  {p}" for p in offenders
    )
