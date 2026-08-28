"""A subscription that saves itself must not drop the artifact fields it does not model.

A subscription IS an artifact — no private table, no bespoke endpoint, which is what makes it
light-cone authorized like anything else. The cost of that choice is that saving one goes through
`update_artifact`, and this store has no field-level update: `put_artifact` writes the whole
document, deliberately, because an artifact is one versioned object. So a writer holding a PARTIAL
view of an artifact has to read the whole thing before it writes.

`save_subscription` did not. It built a fresh 8-field artifact from `to_artifact` and
hand-copied `created_time` and `created_by` back onto it — the right idea applied to two fields and
forgotten for four. `root_id`, `origin_root`, `content_ref` and `description` were absent from the
new entity, `to_lattice_doc` strips Nones on the way in, and the row came out without them.

`origin_root` is the one that matters, and `entities/artifact.py` states the invariant verbatim
where it declares the field: it must SURVIVE the storage round trip, because it is the key
principal the content was encrypted under — "a key root that a `from_dict` silently dropped would
be recomputed, or defaulted, on the next save, re-keying content whose ciphertext was written
under the old value."

And this ran on **every durable ack** (`ack` → `advance_cursor` → `save_subscription`), so the
rate of destruction was the consumer's throughput.

The fix inverts the shape: an update starts from the stored artifact and changes only what a
subscription owns. That makes the preserved set "everything not named", which is the only version
that stays correct when `Artifact` grows a field — so the last test here is about the shape rather
than about any particular field.
"""
from __future__ import annotations

import pytest

from mantle.db import lattice_api
from mantle.events import event_bus
from mantle.entities.subscription import (
    Subscription, advance_cursor, load_subscription, save_subscription,
)


@pytest.fixture
def store(tmp_path):
    return lattice_api.LatticeDatabase(str(tmp_path / "sub.db"), origin="node-a")


def _saved(store, **kw):
    """A subscription persisted once, with whatever the store stamps on creation."""
    sub = Subscription(
        name="watcher",
        filter=event_bus.EventFilter(container_id="ws-1", event_names=["artifact.*"]),
        owner_id="u-1",
        **kw,
    )
    return save_subscription(store, sub)


def _raw(store, artifact_id):
    from mantle.db.backend import get_raw_artifact

    return get_raw_artifact(store, artifact_id)


# ── the key root ─────────────────────────────────────────────────────────────────────────────


def test_an_ack_preserves_the_origin_root(store):
    """The whole finding, at the smallest scale that reproduces it: save, ack, look again."""
    sub = _saved(store, container_id="ws-1")
    before = _raw(store, sub.id).get("origin_root")
    assert before, "the create path should have stamped an origin_root to preserve"

    advance_cursor(store, sub.id, "node-a:7")

    after = _raw(store, sub.id).get("origin_root")
    assert after == before, (
        f"acking the subscription changed its key root from {before!r} to {after!r} — the content "
        "key principal moved under ciphertext that was written against the old one"
    )


def test_repeated_acks_do_not_erode_it(store):
    """A durable subscription is acked continuously, so once is not the interesting number.

    A per-save loss shows up on the first ack; a per-save DRIFT (re-derived each time from
    something that itself moves) would only show up after several. Both are excluded here.
    """
    sub = _saved(store, container_id="ws-1")
    first = _raw(store, sub.id).get("origin_root")
    for seq in range(1, 6):
        advance_cursor(store, sub.id, "node-a:%d" % seq)
    assert _raw(store, sub.id).get("origin_root") == first


def test_a_top_level_subscription_keeps_its_own_id_as_its_root(store):
    """A subscription in no container is its own origin root (`_stamp_origin_root`: "top-level
    artifact IS the root"), so the field is non-empty for that case too and has just as much to
    lose."""
    sub = _saved(store)
    assert _raw(store, sub.id).get("origin_root") == sub.id
    advance_cursor(store, sub.id, "node-a:3")
    assert _raw(store, sub.id).get("origin_root") == sub.id


# ── and the other three fields ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("field,value", [
    ("root_id", "root-abc"),
    ("description", "what this subscription is for"),
])
def test_an_ack_preserves_the_other_dropped_fields(store, field, value):
    """The audit named four fields; `origin_root` is the dangerous one and these are two of the
    other three. Set on the stored row directly, because a `Subscription` has no way to express
    them — which is the entire reason they were being lost.

    `content_ref` is not tested here. Planting one makes the very next read fail:
    `doc_boundary.decrypt_artifact_content`
    resolves the ref against the content tier and raises `ContentDecryptionError` when it is not
    there — refusing rather than serving an artifact that claims content it cannot produce, which
    is the correct behaviour and not something to work around in a test. So a subscription carrying
    a `content_ref` cannot be loaded at all, ack or no ack, and the field being preserved is
    unobservable from this path. It is preserved by construction anyway (see
    `test_the_preserved_set_is_everything_not_named`): nothing enumerates it, so nothing drops it.
    """
    sub = _saved(store, container_id="ws-1")
    entity = lattice_api.get_artifact(store, sub.id)
    setattr(entity, field, value)
    lattice_api.update_artifact(store, entity)
    assert _raw(store, sub.id).get(field) == value

    advance_cursor(store, sub.id, "node-a:9")

    assert _raw(store, sub.id).get(field) == value, (
        f"acking dropped {field}, which a subscription never models and therefore never owned"
    )


def test_the_fields_a_subscription_does_own_still_change(store):
    """The fix must not overshoot into "preserve everything". A cursor that stopped advancing
    would be a worse bug than the one being fixed — the consumer would redeliver forever."""
    sub = _saved(store, container_id="ws-1")
    advance_cursor(store, sub.id, "node-a:42")

    back = load_subscription(store, sub.id)
    assert back is not None
    assert back.cursor.to_dict() == {"node-a": 42}, "the ack did not land"

    back.name = "renamed"
    save_subscription(store, back)
    assert load_subscription(store, sub.id).name == "renamed"


def test_creation_still_stamps_rather_than_preserving(store):
    """`to_artifact()` building from eight fields is correct on a CREATE — there is nothing to
    preserve, and `create_artifact` stamps `origin_root` on the way through. Only the update path
    needed inverting, so this pins that the create path was left alone."""
    sub = _saved(store, container_id="ws-1")
    doc = _raw(store, sub.id)
    assert doc.get("origin_root")
    assert doc.get("created_by") == "u-1"
    assert doc.get("collection_id") == "ws-1"


# ── the shape, not the field list ────────────────────────────────────────────────────────────


def test_the_preserved_set_is_everything_not_named(store):
    """The property that keeps this fixed.

    Enumerating survivors is what fails: two fields get listed, four do not, and nothing about
    the code said the list was supposed to be complete. `OWNED_FIELDS` enumerates the other
    direction — what a subscription may write — so a field added to `Artifact` tomorrow is
    preserved without anyone remembering this file exists.

    Asserted by planting a field name that no `Subscription` will ever know about.
    """
    sub = _saved(store, container_id="ws-1")
    entity = lattice_api.get_artifact(store, sub.id)
    entity.description = "a field the subscription entity cannot express"
    lattice_api.update_artifact(store, entity)

    advance_cursor(store, sub.id, "node-a:1")

    assert _raw(store, sub.id).get("description") == \
        "a field the subscription entity cannot express"
    assert "description" not in Subscription.OWNED_FIELDS, (
        "a field a subscription cannot express must not be in the set it claims to own"
    )
