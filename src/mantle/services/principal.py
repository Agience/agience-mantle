"""A person is an artifact — the WHO that `created_by` points at.

LATTICE contract §2.1: *"A person record is an artifact. There is no separate identity table, no
special-cased principal type."* `vertex.created_by` is a **vertex reference**, treated exactly like
`ct` and `offer`.

`person_id` reproduces Mantle's multi-issuer derivation byte for byte —
`uuid5(_USER_NS, f"{tenant}\\n{sub}")`, `mantle/services/oidc.py:179`. That is the property making
`created_by` a legal COLUMN at all, under `db/schema.py`'s rule *deterministic → column,
observer-dependent → content*: every observer computes the same `uuid5(tenant, sub)` from the same
issuer assertion, while two observers reading their own clocks disagree with no function to
reconcile them.

A non-person author resolves to a **foundation entity**: an artifact, minted once, recording a
principal that is not a human. Three properties keep it from pre-empting §5.7:

1. **It is not a person.** It wears `FOUNDATION_CONTENT_TYPE`, never `PERSON_CONTENT_TYPE`, and is
   asserted by `FOUNDATION_ISSUER` — the system itself, never a person's IdP. Nothing reading the
   store can mistake it for a human or a fabricated user.
2. **The id comes from the SAME derivation**, `uuid5(_USER_NS, f"{issuer}\\n{sub}")`, with the
   foundation issuer substituted. Every observer computes the same id, and no foundation id can
   collide with a person id — the issuer is in the hash.
3. **One entity per process.**

"""
from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

_USER_NS = uuid.UUID("a9c1e0de-1d9f-4e7a-8b2c-9f0e1d2c3b4a")

# Mantle's discriminator (`seed_provisioning/user_provisioning.py:65`). A person artifact is an
# ordinary artifact wearing this content type — there is no identity type (§2.1).
PERSON_CONTENT_TYPE = "application/vnd.agience.person+json"

# The tenant key for an identity asserted LOCALLY rather than by an external IdP — the
# bootstrap author's case, where there is no token and no `iss` claim. It is a fixed string
# (never a node id): every observer must derive the same vertex id for the same person, and a
# per-node tenant would fork one human into one identity per box, which is the exact failure
# `uuid5(tenant, sub)` exists to prevent. When a real issuer IS known, pass it — an identity
# later re-asserted by an external IdP derives a DIFFERENT id, which is correct: a different
# issuer is a different assertion about who this is, and §7.4 keeps that judgement local.
LOCAL_ISSUER = "urn:agience:issuer:local"

# The issuer of a FOUNDATION ENTITY — a principal the SYSTEM asserts about itself, with no token,
# no `iss` claim and no human behind it. Distinct from `LOCAL_ISSUER` on purpose: `LOCAL_ISSUER`
# still says "a person, asserted here rather than by an IdP", and a process is not that. Because
# the issuer is inside the hash, a foundation id can never equal a person id for the same subject.
FOUNDATION_ISSUER = "urn:agience:issuer:foundation"

# A foundation entity's discriminator. Deliberately NOT `PERSON_CONTENT_TYPE`: the whole point is
# that a reader can see this is not a human without knowing the subject string.
FOUNDATION_CONTENT_TYPE = "application/vnd.agience.foundation+json"

# Not people — they resolve to foundation entities. See the module docstring for the ruling.
#
# This set has one home, `prism.principals`, which both this repo and `crystal` depend on: a
# vocabulary the store, the runner and the personas all stamp into artifacts cannot live in any one
# of them, or the two could drift into different answers about the same author. Re-exported here so
# `from mantle.services.principal import PROCESS_AUTHORS` reads the same set.
from prism.principals import PROCESS_AUTHORS, is_process_author       # noqa: F401  (re-export)


def foundation_id(sub: str) -> str:
    """The deterministic vertex id of the FOUNDATION ENTITY named ``sub``.

    The person derivation with `FOUNDATION_ISSUER` substituted, so an observer computes it the
    same way it computes any other principal id and the two can never collide.
    """
    if not sub:
        raise ValueError("a foundation entity needs a subject to be identified by")
    return str(uuid.uuid5(_USER_NS, "%s\n%s" % (FOUNDATION_ISSUER, sub)))


def person_id(sub: str, *, issuer: str = LOCAL_ISSUER) -> str:
    """The deterministic vertex id of the principal identified by ``sub`` under ``issuer``.

    ``uuid5(_USER_NS, f"{issuer}\\n{sub}")`` — Mantle's derivation exactly. The IdP ``sub``
    is unique only WITHIN its issuer, so the issuer must be in the hash or two IdPs minting
    the same ``sub`` would collapse into one person.

    A PROCESS AUTHOR resolves to its foundation entity rather than raising — see the module
    docstring. `is_process_author` still answers *which kind* of principal this is, so a caller
    that needs to distinguish a human from a process asks that question directly instead of
    catching an exception.
    """
    if not sub:
        raise ValueError("a person needs a subject claim to be identified by")
    if is_process_author(sub):
        return foundation_id(sub)
    return str(uuid.uuid5(_USER_NS, "%s\n%s" % (issuer, sub)))


def person_artifact(sub: str, *, issuer: str = LOCAL_ISSUER,
                    public_key: Optional[str] = None,
                    name: Optional[str] = None) -> Dict[str, Any]:
    """The person artifact itself — the UNIVERSAL part of an identity and nothing else.

    ``created_by`` is the person's own id. That is not a placeholder: an identity is
    self-attesting at the root of its own chain, the same way `cite.genesis` is the
    self-anchored root of the citation chain. It also means the vertex resolves under
    node-repair's `created_by resolves` check without a second, prior person to point at —
    which is what stops the fix from needing a bootstrap of its own.
    """
    pid = person_id(sub, issuer=issuer)
    ctx: Dict[str, Any] = {
        "kind": "person",
        # The issuer binding — the claim this id was DERIVED from. Keeping (issuer, sub) on
        # the record makes the derivation auditable: any observer can recompute the id and
        # confirm it, which is what "every observer computes the same id" means in practice.
        "issuer": issuer,
        "sub": sub,
    }
    if name:
        ctx["name"] = name
    if public_key:
        # Universal: a public key is the same fact for everyone who holds it.
        ctx["public_key"] = public_key
    return {
        "id": pid,
        "content_type": PERSON_CONTENT_TYPE,
        "name": name or sub,
        "context": ctx,
        "lemmas": [w for w in str(sub).lower().replace("@", " ").replace(".", " ").split() if w],
        "content": "",
        "created_by": pid,
    }


def foundation_artifact(sub: str) -> Dict[str, Any]:
    """The FOUNDATION ENTITY artifact — the WHO for a principal that is not a human.

    Same shape and same rules as `person_artifact`: universal fields only, self-attesting at the
    root of its own chain, no trust weight and no per-observer judgement (§2.1 consequence 4 /
    §7.4 apply to any shared vertex, not only to people).

    It carries no `cited_from` and no provenance rung, for the same reason `_author_ref` does not
    mint a person through `_mint`: provenance is a claim about where CONTENT came from, and an
    identity is not content. Stamping `cite.genesis` on it would also invert the ordering — the
    citation anchor itself needs an author.
    """
    fid = foundation_id(sub)
    return {
        "id": fid,
        "content_type": FOUNDATION_CONTENT_TYPE,
        "name": sub,
        "context": {
            # `kind` is the field a reader checks; it names what this principal IS. The issuer
            # binding is kept beside it so the derivation stays auditable — any observer can
            # recompute `uuid5(_USER_NS, issuer\nsub)` and confirm this id.
            "kind": "foundation",
            "issuer": FOUNDATION_ISSUER,
            "sub": sub,
        },
        "lemmas": [w for w in str(sub).lower().replace("-", " ").replace(".", " ").split() if w],
        "content": "",
        "created_by": fid,
    }


def principal_artifact(sub: str, *, issuer: str = LOCAL_ISSUER, **kw) -> Dict[str, Any]:
    """The artifact for whatever kind of principal ``sub`` names — the one call a writer needs.

    A process author gets its foundation entity; anything else gets a person. This is the single
    place the two kinds are told apart, so a writer cannot mint the wrong shape by forgetting to
    ask; `person_id(sub)` is the id of whichever artifact this returns.
    """
    if is_process_author(sub):
        return foundation_artifact(sub)
    return person_artifact(sub, issuer=issuer, **kw)


__all__ = [
    "PERSON_CONTENT_TYPE", "FOUNDATION_CONTENT_TYPE", "PROCESS_AUTHORS",
    "LOCAL_ISSUER", "FOUNDATION_ISSUER",
    "is_process_author", "person_id", "foundation_id",
    "person_artifact", "foundation_artifact", "principal_artifact",
]
