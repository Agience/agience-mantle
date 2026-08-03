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


# ⛔ THREE `/types/*` HTTP TESTS DELETED 2026-07-29 (Contract Builder): the router is GONE.
#   · test_router_resolve_uses_env_root                 (GET  /types/resolve)
#   · test_register_endpoint_requires_service_identity  (POST /types/register)
#   · test_register_endpoint_registers_and_resolves     (POST /types/register)
# `routers/types_router.py` does not exist, and `main.py:650` states it plainly: *"types_router (/types
# type-definition API) unmounted in 2b cleanup"*. Those routes 404 BY DESIGN, so the tests asserted 200
# against a surface deliberately removed — they could only ever fail. Same class as `test_router_types.py`,
# already quarantined in this package's `collect_ignore`.
#
# ⓘ WHY THESE SURVIVED WHILE THE SIBLING FILE WAS QUARANTINED — the general failure mode of file-granular
# quarantine: `collect_ignore` works on whole FILES, and these three lived inside `test_types_service.py`,
# whose other ~30 tests are healthy and must keep running. A per-file mechanism cannot say "three functions
# in this file are dead", so they stayed invisible — surfacing only once the seeds/types roots were
# supplyable (§13.21/§13.25) and the whole file finally executed.
#
# ⚠ WHAT THEY PROVED, RECORDED NOT DROPPED: that a type definition could be RESOLVED over HTTP from an
# env-supplied root, and that REGISTERING one REQUIRED A SERVICE IDENTITY. Resolution/registration
# behaviour is still covered in-process elsewhere in this file; **only the HTTP surface's existence and its
# AUTH GATE are now unasserted.** If `/types/*` ever returns, re-establish the auth requirement first — it
# was the only test covering who may register a type.


# ⛔ `test_get_types_roots_is_builtin_and_local_only` DELETED 2026-07-30, with the mechanism it
# described. John's ruling: *"package/types and SEEDS_ROOT seem old and stale. better to get rid of
# it."* `types_service._builtin_types_root()` and the `AGIENCE_TYPES_ROOT` / `AGIENCE_TYPES_DISABLE_
# BUILTIN` env hooks are gone (see the removal note at the top of `types_service.py`).
#
# ⚠ WHAT IT PROVED, RECORDED NOT DROPPED — two claims, and they did not die together:
#   1. **`get_types_roots()` returns the builtin `package/types` root.** GONE, deliberately: there is
#      no builtin root any more, so the assertion has no referent. It was in any case asserting over
#      a `tmp_path` the test itself created — it could never have detected that the REAL default
#      pointed at a directory absent from every deployment, which is precisely the defect that
#      outlived it.
#   2. **A persona `ui/` tree must NOT appear among the roots** — i.e. Mantle does not scan Chorus's
#      filesystem for server-owned types. STILL TRUE AND STILL LOAD-BEARING. It is now structural
#      rather than asserted: `get_types_roots()` reads exactly one env var and has no code path that
#      could reach a persona tree, and `_default_server_ui_roots()` is deprecated-and-always-empty.
#      If a server-tree scan is ever reintroduced, this is the guard that used to stop it.


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
from ._package_root import types_root  # noqa: E402 — single source
_CANONICAL_ROOT = types_root()


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


# ⛔ `test_no_canonical_type_definition_is_duplicated_across_roots` DELETED 2026-07-30, alongside
# the builtin `package/types` root it existed to police.
#
# ⚠ WHAT IT PROVED, RECORDED NOT DROPPED: that no content type had TWO canonical `type.json` files —
# one declaring a `content_type` under `package/types` and another declaring the same one under a
# persona's `src/chorus/<persona>/ui`. A duplicate made the override chain ambiguous: which root won
# depended on root ORDER, so the same type could resolve differently in two processes.
#
# WHY IT NO LONGER HAS A SUBJECT: the ambiguity needed TWO roots claiming canonicity. With the
# builtin root removed there is one optional filesystem root (`AGIENCE_TYPES_PATHS`) plus the runtime
# overlay, and the overlay is keyed by content type in a dict — a second registration REPLACES the
# first (documented upsert, `register_runtime_type`), so "two canonical definitions" is no longer a
# representable state rather than a forbidden one.
#
# ⚠ WHAT IS STILL COVERED, and must stay: `test_declared_content_type_matches_folder_location` below
# keeps the OTHER half — a `type.json` that declares a `content_type` must sit at the folder that
# content type names. That is the check that catches a file in the wrong place, and it is unaffected.


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


# ⛔ `test_known_types_resolve_with_canonical_base` (5 parametrizations) DELETED 2026-07-30, for the
# same reason as the two above: it asserted that resolution with DEFAULT roots put the canonical
# `package/types` base among a type's sources, and there is no longer a default filesystem root to
# find it in.
#
# ⚠ WHAT IT PROVED, RECORDED NOT DROPPED: that five core platform types
# (`authority`/`resource`/`prompt`/`collection`/`workspace`) resolved out of the box, from the
# builtin root, with no env supplied. **That claim was already FALSE IN PRODUCTION and this test
# could not see it** — it passed only because the test process happened to sit in a checkout where
# `_repo_root()/package/types` could be made to resolve, while no deployment ever mounted that tree
# or set a path to it. So it was asserting a property of the developer's filesystem, not of the
# platform. That is exactly the class of check this cleanup removes.
#
# HOW THESE TYPES ACTUALLY RESOLVE NOW, and where it is covered: their owning server pushes them via
# `register_runtime_type`, and the tests immediately below exercise that path — including the overlay
# merging onto an explicitly-supplied filesystem base and the lazy loader fetching an unregistered
# type from its artifact. If a "core types resolve with no configuration" guarantee is ever wanted
# again, it needs a real base first (a mounted tree + `AGIENCE_TYPES_PATHS`, or a seeded registry) —
# re-adding the assertion without one would only re-encode the developer's checkout.


# ---------------------------------------------------------------------------
# Self-registration overlay — servers push the types they own; resolution
# overlays them on top of whatever filesystem base was SUPPLIED (and applies
# inherits).
#
# ⚠ THESE THREE SUPPLY THE BASE EXPLICITLY (`AGIENCE_TYPES_PATHS` → the package
# types tree). They used to `delenv` it and lean on the builtin `package/types`
# default, which was removed 2026-07-30 — naming the root is now the only way to
# get a filesystem base, and being explicit is the point: the mechanism under test
# is the RUNTIME OVERLAY, and it must be tested over a base that is actually there
# rather than one the default may or may not find.
# ---------------------------------------------------------------------------


@pytest.fixture
def _clean_runtime_types():
    types_service.clear_runtime_types()
    types_service.invalidate_type_cache()
    yield
    types_service.clear_runtime_types()
    types_service.invalidate_type_cache()


def test_runtime_full_type_overlays_and_applies_inherits(_clean_runtime_types, monkeypatch):
    """A chorus-OWNED type (no filesystem base of its own) resolves from its pushed
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
    """A third-party type is NOT pre-registered: on a resolution MISS the lazy
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
