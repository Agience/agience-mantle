"""A shared KEYS_DIR must not let one install destroy another's secrets.

Four of the six files `mantle-init-keys` writes carry no component in their name —
`encryption.key`, `inbound_nonce.secret`, `instance.uuid`, `authority.manifest.json` are what every
Agience component's init writes. Pointed at one directory, two installs overwrite each other, and
the damage is asymmetric: a replaced signing key costs a restart, a replaced `encryption.key`
makes every secret sealed under it unreadable, with no recovery short of re-keying the store.

So the directory carries an owner, and the tests below pin the three things that follow from it:
refuse a directory someone else owns, refuse one holding unattributable key files, and detect
after the fact that a shared file was replaced. The second and third matter most — the first is
the case an operator can see coming.
"""
from __future__ import annotations

import json
from pathlib import Path

from mantle.scripts import dev_init_keys as dik


def _init(tmp_path: Path, *extra: str) -> int:
    return dik.main(["--keys-dir", str(tmp_path), *extra])


# ── the marker is written, and describes what actually landed ───────────────────────────────────

def test_a_fresh_init_claims_the_directory(tmp_path):
    assert _init(tmp_path / "keys") == 0
    marker = json.loads((tmp_path / "keys" / dik._OWNER_FILE).read_text(encoding="utf-8"))
    assert marker["component"] == "mantle"
    # Fingerprints, never the material: the marker sits beside the keys it describes, so it has to
    # be useless to anyone who can read it.
    for name, digest in marker["shared_fingerprints"].items():
        assert name in dik._SHARED_NAMES
        assert digest not in (tmp_path / "keys" / name).read_text(encoding="utf-8")


def test_every_unnamespaced_file_is_fingerprinted(tmp_path):
    """The fingerprint set IS the collision detector — a shared file left out of it is a file
    another install can replace silently."""
    keys = tmp_path / "keys"
    assert _init(keys) == 0
    marker = json.loads((keys / dik._OWNER_FILE).read_text(encoding="utf-8"))
    assert set(marker["shared_fingerprints"]) == set(dik._SHARED_NAMES)


def test_the_shared_names_are_the_ones_without_a_component_in_them(tmp_path):
    """Guards the list itself: a new keyset member that carries no component name must join
    `_SHARED_NAMES`, or it becomes a silent collision the marker cannot see."""
    unnamespaced = [n for n in dik._KEYSET if not n.startswith("mantle.")]
    assert set(unnamespaced) == set(dik._SHARED_NAMES)


# ── refusing a directory that is not ours ───────────────────────────────────────────────────────

def test_a_directory_owned_by_another_component_is_refused(tmp_path):
    keys = tmp_path / "shared"
    keys.mkdir()
    (keys / dik._OWNER_FILE).write_text(json.dumps({"component": "origin"}), encoding="utf-8")
    assert _init(keys) == 1
    assert not (keys / "encryption.key").exists()


def test_force_does_not_override_another_components_ownership(tmp_path):
    """`--force` is consent to replace THIS install's keys. It cannot be consent to destroy the
    secrets of a component that is not even running."""
    keys = tmp_path / "shared"
    keys.mkdir()
    (keys / dik._OWNER_FILE).write_text(json.dumps({"component": "origin"}), encoding="utf-8")
    assert _init(keys, "--force") == 1
    assert not (keys / "encryption.key").exists()


def test_unattributable_key_files_are_refused_by_default(tmp_path):
    """Key files with no marker: something wrote here that does not participate in this scheme.
    Guessing which component it was would be worse than refusing."""
    keys = tmp_path / "shared"
    keys.mkdir()
    (keys / "encryption.key").write_text("someone-elses-key\n", encoding="utf-8")
    assert _init(keys) == 1
    assert (keys / "encryption.key").read_text(encoding="utf-8") == "someone-elses-key\n"


def test_an_unknown_writer_can_be_overridden_but_a_named_one_cannot(tmp_path):
    """The asymmetry that makes `--force` meaningful. An unmarked directory is ambiguous and the
    operator can see what this script cannot, so `--force` proceeds. A marker NAMING another
    component is not ambiguous, and no flag here is consent to destroy its secrets."""
    unknown = tmp_path / "unknown"
    unknown.mkdir()
    (unknown / "encryption.key").write_text("stray\n", encoding="utf-8")
    assert _init(unknown, "--force") == 0

    named = tmp_path / "named"
    named.mkdir()
    (named / dik._OWNER_FILE).write_text(json.dumps({"component": "chorus"}), encoding="utf-8")
    (named / "encryption.key").write_text("theirs\n", encoding="utf-8")
    assert _init(named, "--force") == 1
    assert (named / "encryption.key").read_text(encoding="utf-8") == "theirs\n"


def test_force_still_replaces_our_own_keyset(tmp_path):
    """The refusals above must not have made the ordinary case impossible."""
    keys = tmp_path / "keys"
    assert _init(keys) == 0
    first = (keys / "encryption.key").read_text(encoding="utf-8")
    assert _init(keys, "--force") == 0
    assert (keys / "encryption.key").read_text(encoding="utf-8") != first


def test_an_occupied_directory_is_left_alone_without_force(tmp_path):
    keys = tmp_path / "keys"
    assert _init(keys) == 0
    original = (keys / "encryption.key").read_text(encoding="utf-8")
    assert _init(keys) == 1
    assert (keys / "encryption.key").read_text(encoding="utf-8") == original


# ── detecting the collision after it has already happened ───────────────────────────────────────

def test_verify_passes_on_an_intact_keyset(tmp_path):
    keys = tmp_path / "keys"
    assert _init(keys) == 0
    assert dik.verify_keyset(keys) == []
    assert _init(keys, "--verify") == 0


def test_verify_names_a_shared_file_replaced_by_another_install(tmp_path):
    """The case the marker exists for. Nothing fails at the moment of the clobber — the node keeps
    running on a key it can no longer decrypt anything with, and the damage only surfaces the next
    time a secret is read. This is the check that can be run before that happens."""
    keys = tmp_path / "keys"
    assert _init(keys) == 0
    (keys / "encryption.key").write_text("another-installs-key\n", encoding="utf-8")

    problems = dik.verify_keyset(keys)
    assert any("encryption.key" in p and "REPLACED" in p for p in problems), problems
    assert _init(keys, "--verify") == 1


def test_verify_reports_a_missing_member(tmp_path):
    keys = tmp_path / "keys"
    assert _init(keys) == 0
    (keys / "inbound_nonce.secret").unlink()
    problems = dik.verify_keyset(keys)
    assert any("inbound_nonce.secret" in p for p in problems), problems


def test_verify_refuses_to_vouch_for_an_unmarked_directory(tmp_path):
    keys = tmp_path / "keys"
    assert _init(keys) == 0
    (keys / dik._OWNER_FILE).unlink()
    assert any(dik._OWNER_FILE in p for p in dik.verify_keyset(keys))


def test_a_corrupt_marker_does_not_lock_an_operator_out(tmp_path):
    """A marker that cannot be parsed must not be able to block someone from re-initialising their
    own directory — that would turn a corrupt byte into an unrecoverable state."""
    keys = tmp_path / "keys"
    assert _init(keys) == 0
    (keys / dik._OWNER_FILE).write_text("{not json", encoding="utf-8")
    assert _init(keys, "--force") == 0
    assert dik.verify_keyset(keys) == []


# ── the dry run still writes nothing, marker included ───────────────────────────────────────────

def test_dry_run_leaves_no_marker(tmp_path):
    keys = tmp_path / "keys"
    assert _init(keys, "--dry-run") == 0
    assert not keys.exists()
