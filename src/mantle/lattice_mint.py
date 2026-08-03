"""LATTICE MINTING — the small writes that put a principal, a grant and a citation into the store.

`_mint` stamps the invariant fields every artifact carries (state, times, provenance, citation) and
writes it. `_ensure_edge` adds a labelled edge idempotently. `_author_ref` resolves an author CLAIM to
the vertex id of its person artifact, minting that artifact if absent. `_ensure_private` mints a
principal's private collection plus the owner READ GRANT that is the only thing making it private.

⚠ MOVED HERE FROM `ember/genesis.py` — 2026-08-02, the chorus→ember DAG work. Every one of these is a
STORE WRITE, and they sat in the runner's 3,000-line op-table because that is where the first caller
happened to be. The personas need them — `lumen/conversation` mints a private collection for every
remembered turn and resolves the author of every triple — and reaching the RUNNER to write a row is the
edge the target DAG forbids.

⛔ `_ensure_private` IS LOAD-BEARING AND FAILS CLOSED, and that is the reason it is worth reading before
touching. PRIVATE IS A GRANT, NOT A FLAG: the collection is non-public only because
`access.mint_owner_read_grant` put the owner's Read grant on it, so `access.is_public` is False for
every member filed there and only the owner's light-cone reaches them. There is no flag fallback, so a
private collection that cannot be gated must RAISE rather than silently leak — the mint is deliberately
not wrapped.

`_author_ref` resolves through `origin.person` (identity is origin's, storage is mantle's — the
concern map, and `mantle → origin` is an already-declared edge). A dangling `created_by` does not
error: it SILENTLY BREAKS GRANT PROPAGATION, because authorization stops flowing through a reference to
a vertex that is not there. That is why the resolution exists at all.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from mantle.db.lattice.constants import COLLECTION_CONTENT_TYPE
from prism.grounding import CITE_GENESIS, P_HUMAN, _now

#: The root collection every other collection hangs under.
UNIVERSE = "universe"
CITATION_CONTENT_TYPE = "application/x-citation"


def _mint(store, doc: Dict[str, Any]) -> None:
    doc.setdefault("state", "committed")
    doc.setdefault("created_time", _now())
    doc.setdefault("modified_time", doc["created_time"])
    doc.setdefault("provenance", P_HUMAN)
    # INVARIANT (§12): every artifact carries a citation. System artifacts anchor on cite.genesis.
    doc.setdefault("cited_from", CITE_GENESIS)
    store.artifacts.put_artifact(doc)


def _ensure_edge(store, from_id: str, to_id: str, label: str,
                 props: Optional[Dict[str, Any]] = None) -> bool:
    """Idempotent labeled edge: add only if `to_id` is not already an out-neighbor under
    `label`. The leaf graph store's add_edge always CREATEs, so we dedupe here."""
    if store.graph is None:
        return False
    if to_id in store.graph.neighbors(from_id, label, direction="out"):
        return False
    store.graph.add_edge(from_id, to_id, label, props or {})
    return True


def _author_ref(store, author: str) -> str:
    """Resolve an author CLAIM (an email / subject) to the vertex id of its person artifact,
    minting that artifact if it does not exist. Idempotent; safe to call per-row.

    ⚠ THIS IS WHY IT EXISTS. Contract §2.1: `created_by` is a VERTEX REFERENCE, not an opaque
    string, and a dangling one **silently breaks grant propagation rather than erroring** —
    authorization stops flowing through a reference to a vertex that is not there, and nothing
    raises. Writing the raw claim (`"john@ikailo.com"`) into `created_by` left every seed row
    citing a vertex that had never been minted; `node-repair.py --deep` reports it as
    `created_by resolves FAIL`, and it did, on every real store.

    ⭐ PROCESS AUTHORS GET A FOUNDATION ENTITY — ANSWERED 2026-07-30. This paragraph used to read
    "PROCESS AUTHORS PASS THROUGH UNCHANGED … contract §5.7 is OPEN, so those rows keep their bare
    string and keep failing the check, which is the honest state." It WAS honest, and it was also
    permanent: `ember-source` authors ~98% of the corpus, so `created_by resolves` could never go
    green on any real store, and no data repair or fresh lattice would have cleared it.

    John's ruling: *"ember-source was added by you. It doesn't matter, it shouldn't raise. If it
    needs a person owner, make a foundation entity."* So a process author now resolves to a
    FOUNDATION entity — the same `uuid5(ns, issuer\nsub)` derivation under a distinct foundation
    issuer, so every observer computes the identical id (what makes `created_by` a legal column)
    and a foundation id can never collide with a person id. It is ONE ENTITY PER PROCESS, not one
    for all: `ember-source` and `ember-local` wrote different rows, and collapsing them would
    destroy the only WHO those 200,000+ rows carry.

    `principal_artifact` is the single dispatch point — it mints a person for a person and a
    foundation for a process, so this function no longer needs to know which it has."""
    from origin.person import person_id, principal_artifact   # identity is origin's
    if not author:
        return author
    pid = person_id(author)          # dispatches: a process author resolves to its foundation id
    if store.artifacts.get_artifact(pid) is None:
        # NOT via `_mint`: an identity carries no citation and no provenance rung. Provenance is a
        # claim about where CONTENT came from; an identity is not content, and stamping
        # `cited_from: cite.genesis` on it would assert that GENESIS is where this principal came
        # from. It also breaks the ordering — `cite.genesis` itself needs an author.
        store.artifacts.put_artifact(principal_artifact(author))
    return pid


def _ensure_private(store, principal: str) -> Tuple[str, str]:
    pid = f"private.{principal}"
    if store.artifacts.get_artifact(pid) is None:
        _mint(store, {
            "id": pid, "content_type": COLLECTION_CONTENT_TYPE, "name": f"Private ({principal})",
            "context": {"name": f"Private ({principal})", "kind": "private",
                        "provenance": P_HUMAN, "origin": "genesis"},   # owner = the grant grantee
            "lemmas": ["private", principal.lower()], "content": ""})
        _ensure_edge(store, pid, UNIVERSE, "sub_collection_of")
        # PRIVATE IS A GRANT, NOT A FLAG. The owner's Read grant is the ONLY thing that makes this
        # collection non-public: `access.gated_collections` contains `pid`, so `access.is_public` is
        # False for every member filed here and only the owner's light-cone reaches them. LOAD-BEARING
        # — there is no flag fallback any more, so a private collection that cannot be gated must FAIL
        # rather than silently leak. Not wrapped: a mint failure propagates and aborts the write.
        from mantle.db.lattice import access
        access.mint_owner_read_grant(store, pid, principal)
    cite = f"cite.owner.{principal}"
    if store.artifacts.get_artifact(cite) is None:
        _mint(store, {
            "id": cite, "content_type": CITATION_CONTENT_TYPE,
            "context": {"dataset": f"owner-provided ({principal})", "kind": "citation",
                        "provenance": P_HUMAN,
                        "role": "the owner deliberately provided this in conversation"},
            "lemmas": ["owner", "private", principal.lower()],
            "content": f"provided by {principal} in conversation"})
    return pid, cite
