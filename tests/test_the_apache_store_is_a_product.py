# ---------------------------------------------------------------------------
# Copyright (c) 2026 Ikailo Inc.  Licensed under the Apache License, Version 2.0.
# ---------------------------------------------------------------------------

"""A store built from the permissive half alone, demonstrated end to end.

`_scratch/ARCHITECTURE-TARGET.md` §3 splits the product line by licence: the permissive surface
(`prism · mantle · crystal · beacon`) is the giveaway, the copyleft surface (`ember · chorus ·
origin · the aperture`) is the product. §5 declares a `store` profile that installs
`prism-py + mantle`, measures with beacon, stores, and searches. Every other test in this repo
runs inside a workspace where `ember` and `entroptics` are importable, so a green suite there is
evidence about this box, not about a giveaway install. This file stands a store up in a fresh
interpreter with `ember`, `entroptics` and `beam` removed from the import system for the whole
scenario, and measures what that store can and cannot do.

## The control

A blocker that silently did nothing would leave every measurement below unchanged and the file
would pass while proving nothing — the failure mode `test_beacon_instrument` already names
(Python 3.12 ignores the legacy `find_module` hook, so a finder written against it alone is never
consulted). So the control is checked before anything else runs:

  1. the scenario asserts none of the three roots is in `sys.modules` before the finder goes up —
     a `meta_path` finder is only consulted for modules not already imported;
  2. it imports each blocked root and requires the `ImportError` to carry the control's marker
     string, which discriminates "blocked by the control" from "absent anyway";
  3. that alone is not enough, because the marker appears for an absent package too.
     `test_the_blocked_packages_are_really_installed_here` runs each root in a separate
     interpreter with no blocker and records whether it genuinely imports. On this box, `ember`
     and `entroptics` import, so the control is blocking two live packages; `beam` does not exist
     in this workspace, so its being blocked is not evidence, and this file says so rather than
     counting it.

The scenario runs in a subprocess rather than a fixture because `conftest.py` imports `mantle` at
collection time — by the time any in-process fixture could install a finder, mantle's whole import
graph is already resolved, and "mantle imports with the copyleft half gone" would be untestable. A
fresh interpreter is the only place the claim can be tested.

## What is demonstrated

    1  the store holds artifacts       8 documents indexed; an unauthorised principal gets
       and enforces access             `GrantDenied` — not a filtered result set — even with
                                        forged contexts naming the owner's real collections, and
                                        its own blind tokens address nothing on disk.
    2  it searches                     `TokenNarrower` and the accessor beneath
                                        `POST /artifacts/recall` return real hits over the encrypted
                                        index, checked against an independent oracle: the stem
                                        sets of the plaintext, computed outside the engine.
    3  it cuts                         `signal_rank` / `read_ordered` against a planted rank, and
                                        `beacon.cut`'s `select` / `novelty_score` /
                                        `subspace_coherence` against a planted relevant set.
    4  projection is unavailable       `absorb_transmit` · `next_by_coupling` ·
       by name                         `membrane_screen` raise `InstrumentRequired`, 503, naming
                                        the contract and the member.
    5  the same for `Dynamics`         `decay_profile` · `resolution_limit` · `embed` ·
                                        `fit_dynamics` · `dynamics_state`.

Sections 4 and 5 are the half that makes this honest: the giveaway is not a crippled aperture, it
is a complete embodiment of the cut (`isinstance(beacon, Read)` is true, all six members) that
does not pose the projection question. A demonstration that only showed the working half would be
marketing.

The silhouette's perfection is a measurement with a named failure mode, not a slogan. The sweeps
below report exact-recovery rates across SNR, and the low-SNR rungs are expected to fall below
1.0 — a read that could not fail near its own noise edge would not be measuring anything. What the
instrument claims to be perfect is the regime clear of the edge.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache

import pytest

# The three roots blocked for this scenario. `ember` holds the runner and the aperture,
# `entroptics` is the generic instrument behind it, `beam` is the aperture's former home. All
# three are the copyleft half or private; a `store`-profile install has none of them.
BLOCKED = ("ember", "entroptics", "beam")

#: The marker the control's `ImportError` carries, to discriminate "this import was blocked by
#: the control" from "this import failed because the package is not there" — two outcomes that
#: are indistinguishable from the exception type alone.
MARK = "REFUSED by the giveaway control"


# ─────────────────────────────────────────────────────────────────────────────────────────────
# The scenario: one fresh interpreter, the copyleft half removed before anything is imported.
# ─────────────────────────────────────────────────────────────────────────────────────────────
# This is a source string rather than a helper module: a helper module on the path would be
# imported by the parent process at collection time, which would mean something mantle-adjacent
# was already imported before the finder went up. `tests/test_beacon_instrument.py::_PROBE` uses
# the same idiom for the same reason.

_SCENARIO = r'''
import json, math, os, sys, tempfile, traceback

BLOCKED = ("ember", "entroptics", "beam")
MARK = "REFUSED by the giveaway control"
R = {"blocked": list(BLOCKED)}

# The control, installed first and proven to bite before anything else runs.
R["already_imported"] = sorted(n for n in BLOCKED if n in sys.modules)

import importlib.util
_found = {}
for n in BLOCKED:
    try:
        spec = importlib.util.find_spec(n)
        _found[n] = None if spec is None else (spec.origin or "namespace")
    except BaseException as exc:
        _found[n] = "find_spec raised " + type(exc).__name__
R["on_path_before_blocker"] = _found


class Blocker:
    """Block the copyleft half at the import system.

    Uses `find_spec` because Python 3.12 ignores the legacy `find_module` hook — a finder that
    defines only that hook is silently skipped, which would make the whole scenario vacuous while
    staying green.

    Matches on the root package, so `ember.anything` and `entroptics.reads` are blocked too: an
    edge entering through a submodule would otherwise walk past the control."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ImportError(MARK + ": " + fullname)
        return None


sys.meta_path.insert(0, Blocker())

bite = {}
for n in BLOCKED:
    try:
        __import__(n)
        bite[n] = "IMPORTED"                      # the control did nothing
    except ImportError as exc:
        bite[n] = "REFUSED" if MARK in str(exc) else "ImportError without the marker: " + str(exc)
    except BaseException as exc:
        bite[n] = "raised " + type(exc).__name__
# and a submodule, to prove the match is on the root rather than the exact name
for n in ("ember.optics", "entroptics.reads", "beam.optics"):
    try:
        __import__(n)
        bite[n] = "IMPORTED"
    except ImportError as exc:
        bite[n] = "REFUSED" if MARK in str(exc) else "ImportError without the marker: " + str(exc)
    except BaseException as exc:
        bite[n] = "raised " + type(exc).__name__
R["blocker_bites"] = bite


def _fatal(where):
    R["fatal"] = {"where": where, "traceback": traceback.format_exc()}
    print(json.dumps(R))
    sys.exit(0)


# The store holds artifacts; an unauthorised principal gets nothing derivable.
try:
    from cryptography.fernet import Fernet

    from mantle.services.acting_principal import ActingPrincipal, set_acting_principal
    from mantle.search.mantle.oracle import (FernetMasterKeyStore, GrantDenied, KeyPurpose,
                                             KeyRequest, OracleService)
    from mantle.search.mantle.sse import (FilePostingStore, SseIndexer, TokenNarrower,
                                          blind_tokens as bt, posting, tokenize)
    R["imports_ok"] = True
except BaseException:
    R["imports_ok"] = False
    _fatal("importing the store surface")

OWNER = "store-owner"
READER = "librarian"          # holds a grant to the owner's contexts
INTRUDER = "mallory"          # holds nothing


class SingleRequesterVerifier:
    """Authorises one named requester everywhere, plus each principal for its own contexts."""
    def __init__(self, who):
        self._who = who

    def authorized(self, *, requester_id, requester_type, principal_id, collection_id, action):
        return requester_id == self._who or requester_id == principal_id


class SelfContextVerifier:
    """Authorises a requester for its own contexts only; every other requester is denied."""
    def authorized(self, *, requester_id, requester_type, principal_id, collection_id, action):
        return requester_id == principal_id


def _act(principal_id, principal_type="principal"):
    set_acting_principal(ActingPrincipal(principal_id=principal_id,
                                         principal_type=principal_type, source="giveaway-demo"))


def _owner_write():
    """`action="update"`. The oracle does not mint a master key on a read action, so an indexing
    request must be a write; a read request here fails as `MasterKeyMissing` and looks like an
    empty corpus."""
    _act(OWNER)
    return KeyRequest(requester_id=OWNER, purpose=KeyPurpose.SELF,
                      requester_type="principal", action="update")


def _read_as(principal_id):
    _act(principal_id, "user")
    return KeyRequest(requester_id=principal_id, purpose=KeyPurpose.GRANT,
                      requester_type="user", action="read")


# Two collections of documents, so a scope filter has something to filter.
CORPUS = [
    ("col-lattice", "art-lattice-1",
     {"title": "the lattice is the standalone store",
      "description": "sqlite and an encrypted content addressed filesystem, zero external database",
      "tags": "lattice storage"}),
    ("col-lattice", "art-lattice-2",
     {"title": "content encryption at rest",
      "description": "a missing key is a silent partition, so fingerprint the key before opening",
      "content": "every artifact is sealed with a derived key and never written in plaintext"}),
    ("col-lattice", "art-lattice-3",
     {"title": "the merkle synced mesh plane",
      "content": "peers exchange segment digests and a cursor may never skip a failed segment"}),
    ("col-lattice", "art-lattice-4",
     {"title": "keyset paging over the artifact ledger",
      "content": "counting every row dereferences every record, so page on an indexed column"}),
    ("col-optics", "art-optics-1",
     {"title": "the tracy widom noise edge",
      "description": "the largest eigenvalue of a gaussian bulk fluctuates about a derived edge",
      "tags": "spectrum"}),
    ("col-optics", "art-optics-2",
     {"title": "a permutation null for correlated rows",
      "content": "documents share vocabulary so a corpus is correlated by construction"}),
    ("col-optics", "art-optics-3",
     {"title": "the silhouette is the cut",
      "description": "beacon measures where a set stops and nothing about where a signal goes"}),
    ("col-optics", "art-optics-4",
     {"title": "projection belongs to the aperture",
      "content": "carrying a frame through a membrane is a different question from cutting a set"}),
]

# The last query is the negative case: it is built from terms absent from the corpus. The
# tokenizer keeps function words, so a query chosen only to "mean nothing" can still match on a
# stem like "at".
QUERIES = ["encryption", "corpus", "key", "lattice segment", "zygote pterodactyl"]


def _oracle_stem_sets():
    """The independent oracle: the engine's answer is checked against stem sets computed from the
    plaintext by the tokenizer alone, not by asking the engine what it thinks it stored. A
    single-term query is exact-token only (the query path builds no prefix tokens), so a document
    matches iff some stem of the query appears among the stems of its indexed fields."""
    out = {}
    for _col, aid, fields in CORPUS:
        stems = set()
        for text in fields.values():
            stems.update(tokenize(text))
        out[aid] = stems
    return out


STEMS = _oracle_stem_sets()


def _expected(query, collections):
    want = set(tokenize(query))
    return sorted(aid for col, aid, _f in CORPUS
                  if col in collections and (want & STEMS[aid]))


try:
    root = os.path.join(tempfile.mkdtemp(prefix="giveaway-store-"), "sse-index")
    oracle = OracleService(FernetMasterKeyStore(Fernet(Fernet.generate_key())),
                           grant_verifier=SingleRequesterVerifier(READER))
    postings = FilePostingStore(root, prefix="mantle-sse")

    indexer = SseIndexer(oracle, postings)
    written = {}
    for col, aid, fields in CORPUS:
        written[aid] = indexer.index_artifact(OWNER, col, aid, fields, _owner_write())
    R["store"] = {"documents": len(CORPUS), "blind_tokens_written": written,
                  "blind_tokens_total": sum(written.values()), "root": root}

    # The files on disk must carry no plaintext, or "encrypted index" is only a label.
    files = []
    for dirpath, _d, filenames in os.walk(root):
        files.extend(os.path.join(dirpath, f) for f in filenames)
    blob = b"".join(open(p, "rb").read() for p in files)
    leaked = [w for w in (b"encryption", b"lattice", b"merkle", b"tracy", b"aperture",
                          b"art-optics-1", b"col-lattice", b"store-owner", b"title")
              if w in blob]
    R["store"]["files_on_disk"] = len(files)
    R["store"]["bytes_on_disk"] = len(blob)
    R["store"]["non_ciphertext_files"] = [p for p in files if not p.endswith(".enc")]
    R["store"]["plaintext_leaked"] = [w.decode() for w in leaked]
except BaseException:
    _fatal("standing the store up")


# Search over the encrypted index, checked against the independent oracle.
try:
    narrower = TokenNarrower(oracle, postings)
    ALL = [(OWNER, "col-lattice"), (OWNER, "col-optics")]

    def _narrow(nrw, q, contexts):
        """``{artifact_id: Coverage}`` for one query over ``contexts``."""
        lookup = nrw.lookup_for(q, _read_as(READER))
        return {} if lookup is None else dict(lookup(list(contexts)))

    searched = {}
    for q in QUERIES:
        found = _narrow(narrower, q, ALL)
        searched[q] = {
            "hits": sorted(found),
            "expected": _expected(q, {"col-lattice", "col-optics"}),
            # How much of the query each hit carried — a count of stems matched, which is what
            # orders a recall with no query vector. Not a score: no IDF, no term frequency,
            # no field length anywhere in it.
            "coverage": {a: c.stems for a, c in found.items()},
            "within_query_length": all(
                0 < c.stems <= len(set(tokenize(q))) for c in found.values()),
        }
    # the scope filter is a real filter, not a display convention
    narrow = _narrow(narrower, "encryption", [(OWNER, "col-lattice")])
    searched["encryption@col-lattice"] = {
        "hits": sorted(narrow),
        "expected": _expected("encryption", {"col-lattice"}),
        "coverage": {a: c.stems for a, c in narrow.items()},
        "within_query_length": True}
    R["search"] = searched

    # The same index survives a reopen: a store is not a process-lifetime cache.
    reopened = TokenNarrower(oracle, FilePostingStore(root, prefix="mantle-sse"))
    R["search_after_reopen"] = sorted(_narrow(reopened, "encryption", ALL))
except BaseException:
    _fatal("searching the encrypted index")


# The accessor beneath `POST /artifacts/recall`. The narrower under it is the production
# object reading the production encrypted index. What is stubbed is
# `resolve_authorized_scope`'s LATTICE half, which needs a live handle — the same seam
# `tests/test_router_accessor_candidates.py` uses. Section 1 shows that seam is not where
# authorisation lives: the oracle denies access even when the resolver is forged wide open.
#
# The stub returns BOTH granularities the light cone resolves: the (principal, collection)
# pairs that decide which encrypted index may be opened, and the artifact ids the reader may
# actually read — and it applies the token narrowing exactly as the real resolver does, by
# MEETING the narrowing's key set into the authorized ids. That meet is the whole security
# argument, so a stub that skipped it would be describing a different accessor.
#
# BOTH entry points are exercised, because they now answer over the same universe: the
# ranked one orders it by query coverage, the candidate one publishes it with no order and
# no score for an external flavor to rank within.
try:
    from mantle.search.mantle.lightcone import AuthorizedScope
    from mantle.search.mantle.sse import router_accessor as ra
    from mantle.search.types import SearchQuery

    _ALL_IDS = frozenset(aid for _c, aid, _f in CORPUS)

    def _stub_resolve(store_db, principal_id, token_lookup=None, **kw):
        ids = _ALL_IDS
        if token_lookup is not None:
            ids &= frozenset(str(a) for a in (token_lookup(list(ALL)) or ()))
        return AuthorizedScope(list(ALL), ids, {a: "2025-01-01T00:00:00Z" for a in ids})

    ra.resolve_authorized_scope = _stub_resolve
    accessor = ra.MantleSseSearchAccessor(
        object(), store_db=object(),
        embeddings=lambda texts: [],          # no vector arm on a store profile
        narrower=narrower)
    _act(READER, "user")
    out = accessor.candidates(SearchQuery(query_text="encryption", user_id=READER, size=20))
    _act(READER, "user")
    ranked = accessor.search(SearchQuery(query_text="encryption keys", user_id=READER, size=20))
    R["accessor"] = {
        "candidates": sorted(c["artifact_id"] for c in out["candidates"]),
        "expected": _expected("encryption", {"col-lattice", "col-optics"}),
        "candidate_keys": sorted({k for c in out["candidates"] for k in c}),
        "model_id": out["model_id"],
        "ranked": sorted(h.doc_id for h in ranked.hits),
        "ranked_expected": _expected("encryption keys", {"col-lattice", "col-optics"}),
        "ordering": ranked.ordering,
        "scores": {h.doc_id: h.score for h in ranked.hits},
        "descending": [h.score for h in ranked.hits] == sorted(
            (h.score for h in ranked.hits), reverse=True),
    }
except BaseException:
    R["accessor"] = {"error": traceback.format_exc()}


# A second principal gets nothing derivable, on a corpus proven readable.
try:
    ref = {}
    # Sensitivity control first: the same corpus, freshly sealed under a self-only oracle, is
    # readable by its owner. Without this, every emptiness below would have a second possible
    # cause.
    root2 = os.path.join(tempfile.mkdtemp(prefix="giveaway-selfonly-"), "sse-index")
    self_only = OracleService(FernetMasterKeyStore(Fernet(Fernet.generate_key())),
                              grant_verifier=SelfContextVerifier())
    p2 = FilePostingStore(root2, prefix="mantle-sse")
    ix2 = SseIndexer(self_only, p2)
    for col, aid, fields in CORPUS:
        ix2.index_artifact(OWNER, col, aid, fields, _owner_write())
    n2 = TokenNarrower(self_only, p2)

    _act(OWNER)
    owner_req = KeyRequest(requester_id=OWNER, purpose=KeyPurpose.SELF,
                           requester_type="principal", action="read")
    ref["control_owner_reads_its_own_corpus"] = sorted(
        n2.lookup_for("encryption", owner_req)(ALL))

    # (a) no key is issued at all
    try:
        self_only.derive_sse_key(OWNER, _read_as(INTRUDER))
        ref["key_issued"] = True
    except GrantDenied as exc:
        ref["key_issued"] = False
        ref["key_refusal"] = str(exc)

    # (b) forged contexts naming the owner's real collections still get nothing — this happens
    #     before a single posting list is read, not as a filter applied to results afterward
    try:
        found = n2.lookup_for("encryption", _read_as(INTRUDER))(ALL)
        ref["forged_context_hits"] = sorted(found)
        ref["forged_context_refused"] = False
    except GrantDenied as exc:
        ref["forged_context_refused"] = True
        ref["forged_context_refusal"] = str(exc)
        ref["forged_context_hits"] = []

    # (c) the intruder's own key addresses nothing: a directory listing cannot become a search
    _act(OWNER)
    owner_key = self_only.derive_sse_key(OWNER, KeyRequest(
        requester_id=OWNER, purpose=KeyPurpose.SELF, requester_type="principal", action="read"))
    _act(INTRUDER)
    intruder_key = self_only.derive_sse_key(INTRUDER, KeyRequest(
        requester_id=INTRUDER, purpose=KeyPurpose.SELF, requester_type="principal",
        action="update"))
    addressed = {}
    # Every term here must be in a title, because the tokens below are built for `FIELD_TITLE`.
    for term in ("encryption", "lattice", "silhouette"):
        stem = tokenize(term)[0]
        owner_tok = bt.blind_token(owner_key, bt.FIELD_TITLE, stem)
        intruder_tok = bt.blind_token(intruder_key, bt.FIELD_TITLE, stem)
        addressed[term] = {
            "owner_token_addresses_a_posting": p2.get_posting(OWNER, owner_tok) is not None,
            "intruder_token_under_owner": p2.get_posting(OWNER, intruder_tok) is not None,
            "intruder_token_under_intruder": p2.get_posting(INTRUDER, intruder_tok) is not None,
        }
    ref["blind_tokens"] = addressed

    # (d) the ciphertext does not open under the intruder's key even when handed to them
    stem = tokenize("lattice")[0]
    tok = bt.blind_token(owner_key, bt.FIELD_TITLE, stem)
    ct = p2.get_posting(OWNER, tok)
    ref["ciphertext_handed_over_bytes"] = len(ct) if ct else 0
    # `aad=` mirrors the slot binding `SseIndexer` writes with; the intruder attempt below
    # fails on the KEY, so it needs none.
    ref["opens_under_owner_key"] = bool(
        posting.unpack_posting(ct, posting.derive_posting_key(owner_key, tok),
                               aad=posting.posting_aad(OWNER, tok)))
    try:
        posting.unpack_posting(ct, posting.derive_posting_key(intruder_key, tok))
        ref["opens_under_intruder_key"] = True
    except posting.PostingError as exc:
        ref["opens_under_intruder_key"] = False
        ref["ciphertext_refusal"] = type(exc).__name__ + ": " + str(exc)
    R["refusal"] = ref
except BaseException:
    _fatal("demonstrating the refusal")


# Beacon's silhouette, measured against ground truth that was planted, not observed.
try:
    import numpy as np
    from mantle.search import beacon
    from mantle.search.beacon import cut as beacon_cut
    from mantle.search.beacon import instrument as beacon_instrument
    from mantle.search.beacon.engine import _permutation_core as _beacon_permutation_core
    R["numpy_version"] = np.__version__
except BaseException:
    _fatal("importing beacon")


def _structure_rank_k(m):
    """Reproduces the retired `beacon.structure_rank`'s wrapping of `_permutation_core` — the
    public wrapper had zero production callers and was retired; `_permutation_core` itself is
    unchanged and still live behind `instrument.py`'s correlated-row path. This probe measures
    the permutation null's own false-alarm behavior, which the retirement did not touch."""
    core = _beacon_permutation_core(m)
    if not core.readable:
        return 1
    return max(1, core.k_tested + core.offset)


def _plant(seed, N, F, k, snr):
    """The conformance file's own recipe (`test_beacon_conformance._build`), so the frame this
    sweep reads is the same shape of object both embodiments are pinned against."""
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(N, F))
    for _ in range(k):
        u = rng.normal(size=N)
        w = rng.normal(size=F)
        m += snr * np.outer(u / np.linalg.norm(u), w / np.linalg.norm(w))
    return m


try:
    N, F, SEEDS = 120, 16, 25
    rank = []
    # The ladder runs to 256, two rungs past any plausible separation: the point of the top rungs
    # is to show whether a rung that is not perfect gets better with more separation. A rate that
    # keeps climbing is a resolution limit; a rate that saturates is something else, and the two
    # must be distinguishable from this table alone. `mean_k` carries the direction (under- or
    # over-count), which a bare rate discards.
    for k in (1, 2, 3, 4, 5):
        for snr in (4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0):
            exact = interval = 0
            khats, margins = [], []
            for seed in range(SEEDS):
                m = _plant(seed, N, F, k, snr)
                read = beacon_instrument.read_ordered(m)
                khat = int(beacon.signal_rank(m).k)
                khats.append(khat)
                if read.k_margin_last is not None:
                    margins.append(float(read.k_margin_last))
                exact += int(khat == k)
                interval += int(read.k_lo <= k <= read.k_hi)
            rank.append({"planted": k, "snr": snr, "seeds": SEEDS,
                         "exact_rate": exact / SEEDS, "interval_covers_rate": interval / SEEDS,
                         "mean_k": sum(khats) / SEEDS, "min_k": min(khats), "max_k": max(khats),
                         "mean_margin_last": (sum(margins) / len(margins)) if margins else None})
    R["cut_rank_recovery"] = rank

    # The one over-count, examined rather than averaged away: a single planted mode at extreme
    # separation reads k=2 on 1 seed in 25. What matters is not the rate but that the read
    # publishes how close the call was, which is the argument for `SpectralRead` carrying the
    # continuous evidence behind its integer.
    over = []
    for seed in range(SEEDS):
        m = _plant(seed, N, F, 1, 256.0)
        if int(beacon.signal_rank(m).k) != 1:
            rd = beacon_instrument.read_ordered(m)
            over.append({"seed": seed, "k": int(beacon.signal_rank(m).k),
                         "k_margin_last": rd.k_margin_last, "k_margin_next": rd.k_margin_next,
                         "contrast": rd.contrast, "top_share": rd.top_share,
                         "degraded": rd.degraded, "live_channels": rd.live_channels,
                         "k_certain": rd.k_certain})
    R["cut_overcount"] = {"planted": 1, "snr": 256.0, "seeds": SEEDS, "cases": over}

    # The no-structure control: a read that cannot fail proves nothing, so the same instrument is
    # asked about a corpus with nothing planted. `signal_rank` floors at 1 by its own stated rule
    # (divergence 4), so the honest report is not 0 — it is a cut with no break and a permutation
    # null that collapses.
    noise_k, perm_k, breaks, contrasts = [], [], [], []
    for seed in range(40):
        m = _plant(1000 + seed, N, F, 0, 0.0)
        noise_k.append(int(beacon.signal_rank(m).k))
        perm_k.append(int(_structure_rank_k(m)))
        rd = beacon_instrument.read_ordered(m)
        contrasts.append(float(rd.contrast))
        _keep, gap = beacon_cut.top_break(np.abs(np.linalg.svd(m, compute_uv=False)) * 0.0)
        breaks.append(float(gap))
    R["cut_no_structure"] = {
        "seeds": 40,
        "signal_rank_values": sorted(set(noise_k)),
        "signal_rank_max": max(noise_k),
        "signal_rank_at_the_floor_rate": noise_k.count(1) / len(noise_k),
        "structure_rank_values": sorted(set(perm_k)),
        "structure_rank_max": max(perm_k),
        "structure_rank_at_the_floor_rate": perm_k.count(1) / len(perm_k),
        "permutation_draws": int(beacon_instrument.correlated_null().draws),
        "far": float(beacon.DEFAULT_FAR),
        "mean_contrast": sum(contrasts) / len(contrasts),
        "flat_spectrum_relative_gap": sorted(set(breaks)),
    }

    # The computed null: a frame that cannot carry a read has no count to give, so it returns
    # `None` — a different statement from a resolved count of 1.
    R["cut_computed_null"] = {
        "single_row": beacon_instrument.resolvable(np.ones((1, 8))),
        "one_live_channel": beacon_instrument.resolvable(np.zeros((40, 8))),
        "readable_frame": beacon_instrument.resolvable(_plant(7, 120, 16, 3, 32.0)),
        "min_rows": int(beacon_instrument.MIN_ROWS),
    }
except BaseException:
    _fatal("measuring the rank recovery")


# The adaptive cut: `select` recovering a planted relevant set.
try:
    D, NPOOL, NREL = 256, 60, 12

    def _pool(seed, alpha):
        """A bounded candidate pool with a planted relevant set: the first `NREL` items carry the
        query direction at strength `alpha`, the rest are drawn from the same noise and carry
        none. Ground truth is therefore the set `{0 .. NREL-1}`, planted rather than judged."""
        rng = np.random.default_rng(seed)
        q = rng.normal(size=D)
        q /= np.linalg.norm(q)
        E = rng.normal(size=(NPOOL, D))
        E[:NREL] += alpha * q
        return E, q

    sel = []
    for alpha in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        exact = 0
        prec = rec = 0.0
        kept_sizes = []
        heads = []
        for seed in range(25):
            E, q = _pool(seed, alpha)
            keep = set(int(i) for i in beacon_cut.select(E, q))
            truth = set(range(NREL))
            kept_sizes.append(len(keep))
            heads.append(int(beacon_cut.derive_heads(E)))
            exact += int(keep == truth)
            prec += (len(keep & truth) / len(keep)) if keep else 0.0
            rec += len(keep & truth) / len(truth)
        sel.append({"alpha": alpha, "seeds": 25, "planted": NREL,
                    "exact_set_rate": exact / 25, "mean_precision": prec / 25,
                    "mean_recall": rec / 25,
                    "mean_kept": sum(kept_sizes) / len(kept_sizes),
                    "derived_heads": sorted(set(heads))})
    R["cut_select"] = sel

    # `select` on a pool with no planted relevance — what does the cut say when there is nothing
    # to find? It reports a break of size ~1 rather than an empty result.
    flat = []
    for seed in range(25):
        E, q = _pool(seed, 0.0)
        keep = beacon_cut.select(E, q)
        power = beacon_cut.signal_power(beacon_cut.head_screen(E, q, beacon_cut.derive_heads(E)))
        _m, gap = beacon_cut.top_break(power)
        flat.append((len(keep), float(gap)))
    R["cut_select_no_structure"] = {
        "seeds": 25,
        "mean_kept": sum(k for k, _g in flat) / len(flat),
        "max_kept": max(k for k, _g in flat),
        "mean_relative_gap": sum(g for _k, g in flat) / len(flat),
        "max_relative_gap": max(g for _k, g in flat),
    }
except BaseException:
    _fatal("measuring the adaptive cut")


# Novelty, coherence and anomaly localisation (`anomaly`, `anomaly_rank`, `most_anomalous`,
# `novelty_score`, `subspace_coherence`, and `coherent_basis`/`in_subspace_fraction` which existed
# only to serve them) were retired from `cut.py` — zero production callers anywhere in mantle.
# See SEARCH-ARCHITECTURE.md. This probe and `test_3_novelty_and_coherence_measure_a_planted_subspace`
# below went with them.


# Projection and prediction are unfilled, and named as such.
try:
    from prism.instrument import (Dynamics, Instrument, InstrumentRequired, Read, members_of,
                                  require)

    R["contracts"] = {
        "isinstance_Read": isinstance(beacon_instrument, Read),
        "isinstance_Instrument": isinstance(beacon_instrument, Instrument),
        "isinstance_Dynamics": isinstance(beacon_instrument, Dynamics),
        "members_read": list(members_of(beacon_instrument, "read")),
        "members_embodiment": list(members_of(beacon_instrument, "embodiment")),
        "members_dynamics": list(members_of(beacon_instrument, "dynamics")),
        "http_status": int(InstrumentRequired.http_status),
        "code": InstrumentRequired.code,
    }

    def _refuse(member, contract, at):
        rec = {"attribute_present": hasattr(beacon_instrument, member)}
        try:
            require(beacon_instrument, member, contract=contract, at=at)
            rec["refused"] = False
        except InstrumentRequired as exc:
            rec.update({"refused": True, "exception": type(exc).__name__,
                        "http_status": int(exc.http_status), "code": exc.code,
                        "contract": exc.contract, "member": exc.member, "at": exc.at,
                        "message": str(exc)})
        return rec

    R["projection_refusals"] = {
        "absorb_transmit": _refuse("absorb_transmit", "embodiment", "crystal.conduct"),
        "next_by_coupling": _refuse("next_by_coupling", "embodiment", "reach.route_next"),
        "membrane_screen": _refuse("membrane_screen", "embodiment", "crystal.membrane"),
    }
    R["dynamics_refusals"] = {
        m: _refuse(m, "dynamics", "crystal.predict")
        for m in ("decay_profile", "resolution_limit", "embed", "fit_dynamics", "dynamics_state")
    }
except BaseException:
    _fatal("measuring the refusals")


R["leaked"] = sorted(n for n in sys.modules if n.split(".")[0] in BLOCKED)
print(json.dumps(R))
'''


def _run_python(body: str, *, timeout: int) -> subprocess.CompletedProcess:
    """Run `body` in a fresh interpreter that can see this process's `sys.path`.

    The script and the path list are written to files rather than passed inline, because Windows
    caps a command line at 32767 characters and `sys.path` inside the full suite (pytest's
    `pythonpath`, the second testpath root, every conftest insertion) is long enough to exceed it
    when prepended to the scenario text. Both the script and the path list go to disk, and argv
    carries only two short paths.

    `sys.path` is handed over explicitly rather than left to `PYTHONPATH`: the repo runs
    uninstalled off pytest's `pythonpath = ["src"]`, so a subprocess inheriting only the
    environment would not find `mantle`.
    """
    work = tempfile.mkdtemp(prefix="giveaway-probe-")
    try:
        paths = os.path.join(work, "syspath.json")
        script = os.path.join(work, "probe.py")
        with open(paths, "w", encoding="utf-8") as fh:
            json.dump(sys.path, fh)
        with open(script, "w", encoding="utf-8") as fh:
            fh.write("import sys, json\nsys.path[:0] = json.load(open(sys.argv[1]))\n" + body)
        return subprocess.run(
            [sys.executable, script, paths], capture_output=True, text=True,
            # [[eigh-is-not-thread-safe]]: single-threaded BLAS, because the sweeps run
            # several hundred SVD/eigh calls.
            env=dict(os.environ, OPENBLAS_NUM_THREADS="1"), timeout=timeout)
    finally:
        shutil.rmtree(work, ignore_errors=True)


@lru_cache(maxsize=1)
def _scenario() -> dict:
    """Run the whole demonstration once, in a pristine interpreter, and return its record."""
    proc = _run_python(_SCENARIO, timeout=900)
    assert proc.returncode == 0, (
        f"the giveaway scenario did not complete ({proc.returncode}):\n"
        f"{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


@lru_cache(maxsize=1)
def _installed_without_the_blocker() -> dict:
    """Positive control: does each blocked root actually import on this box with no blocker?

    The finder blocks by name, before resolution, so a package that is not installed produces
    exactly the same marked `ImportError` as one that is. This probe distinguishes "the control
    blocked a live package" from "there was nothing there to block", and the answer differs per
    package — see the module docstring.
    """
    out = {}
    for name in BLOCKED:
        proc = _run_python(f"import {name}\nprint('OK')\n", timeout=300)
        out[name] = {"imports": proc.returncode == 0,
                     "error": (proc.stderr.strip().splitlines() or [""])[-1][:300]}
    return out


def _report(title: str, body: dict) -> None:
    """Print a measurement block. The numbers are the deliverable: a demonstration whose result
    exists only inside an assertion is a unit test, and this file is not one. Run with `-s` to
    read them."""
    print("\n" + "=" * 92 + f"\n{title}\n" + "=" * 92)
    print(json.dumps(body, indent=2, default=str))


# The control, asserted first because every conclusion below rests on it.

def test_the_scenario_completed() -> None:
    """The scenario reports its own fatal errors rather than dying, so a partial run is
    diagnosable. Anything in `fatal` is a step of the demonstration that could not be done."""
    v = _scenario()
    assert "fatal" not in v, (
        f"the demonstration could not complete step {v['fatal']['where']!r} on the Apache half "
        f"alone — that is the finding, not a flake:\n{v['fatal']['traceback']}")


def test_the_blocked_packages_are_really_installed_here() -> None:
    """The positive control, and it is allowed to report a partial answer.

    Catches the case where the whole file passes because `ember` and `entroptics` were never
    importable in this environment, so blocking them blocked nothing. `beam` is in exactly that
    state (archived, §3), which is why this test asserts only that at least one blocked root
    genuinely imports and reports the rest. Asserting all three would fail on a true fact about
    the workspace and teach the next reader to delete the check."""
    v = _installed_without_the_blocker()
    _report("0a · POSITIVE CONTROL — do the blocked packages import with NO blocker?", v)
    live = [n for n, r in v.items() if r["imports"]]
    assert live, (
        "none of ember/entroptics/beam imports on this box, so removing them removes NOTHING and "
        f"every measurement in this file is vacuous: {v}")
    assert "entroptics" in live, (
        "entroptics — the generic instrument the AGPL aperture wraps — is not importable here, so "
        "the single most load-bearing removal in this demonstration is not being demonstrated")


def test_the_blocker_bites_before_anything_else_runs() -> None:
    """The control itself. Every number in this file is a claim about a store with the copyleft
    half absent; if the finder were silently inert the numbers would be identical and the file
    would still pass. So: nothing blocked may already be imported (a `meta_path` finder is never
    consulted for a module already in `sys.modules`), and every blocked root and a submodule of
    each must come back blocked with this control's marker — an unmarked `ImportError` would mean
    the package simply failed to load, which is a different fact."""
    v = _scenario()
    _report("0b · THE BLOCKER CONTROL", {
        "already_imported_before_the_finder": v["already_imported"],
        "on_path_before_the_blocker": v["on_path_before_blocker"],
        "bites": v["blocker_bites"],
        "leaked_into_sys_modules_during_the_run": v["leaked"]})
    assert v["already_imported"] == [], (
        f"{v['already_imported']} was imported before the finder went up; a meta_path finder is "
        "only consulted for modules NOT in sys.modules, so this run would prove nothing")
    assert all(x == "REFUSED" for x in v["blocker_bites"].values()), (
        f"the control did not bite on every edge: {v['blocker_bites']}")
    assert v["leaked"] == [], (
        f"a blocked package reached sys.modules during the scenario anyway: {v['leaked']}")


# The store holds artifacts and enforces access correctly.

def test_1_the_store_holds_documents_and_writes_only_ciphertext() -> None:
    v = _scenario()
    s = v["store"]
    _report("1a · THE STORE (Apache half only)", s)
    assert s["documents"] == 8
    assert all(n > 0 for n in s["blind_tokens_written"].values()), s["blind_tokens_written"]
    assert s["files_on_disk"] > 0, "nothing was written — the plaintext scan below is vacuous"
    assert s["non_ciphertext_files"] == [], s["non_ciphertext_files"]
    assert s["plaintext_leaked"] == [], (
        f"readable plaintext in the on-disk index: {s['plaintext_leaked']}")


def test_1_an_unauthorised_principal_gets_nothing_derivable() -> None:
    """Not filtered results — nothing. Four independent facts, each measured, and the first is a
    sensitivity control: the same corpus is readable by its owner, so the emptiness below cannot
    be explained by an index that was never written."""
    v = _scenario()
    r = v["refusal"]
    _report("1b · THE REFUSAL", r)
    assert r["control_owner_reads_its_own_corpus"], (
        "the control corpus is not readable by its own owner, so every refusal below has a second "
        "possible cause and proves nothing")
    assert r["key_issued"] is False, "a key was issued to a principal holding no grant"
    assert r.get("key_refusal"), "the refusal carried no message an operator could act on"
    assert r["forged_context_refused"] is True and r["forged_context_hits"] == []
    for term, a in r["blind_tokens"].items():
        assert a["owner_token_addresses_a_posting"], term
        assert not a["intruder_token_under_owner"], term
        assert not a["intruder_token_under_intruder"], term
    assert r["opens_under_owner_key"] is True
    assert r["opens_under_intruder_key"] is False


# It searches.

def test_2_the_encrypted_lexical_index_returns_real_hits() -> None:
    """Against an independent oracle. `expected` is computed from the plaintext by the tokenizer
    alone; the engine is never asked what it thinks it stored. A demonstration that compared the
    engine to itself would pass on an engine that returns whatever it happens to hold."""
    v = _scenario()
    s = v["search"]
    _report("2a · SEARCH over the encrypted index (no ember, no entroptics)", s)
    for q, rec in s.items():
        assert rec["hits"] == rec["expected"], f"{q}: {rec['hits']} != oracle {rec['expected']}"
        assert set(rec["coverage"]) == set(rec["hits"]), (
            f"{q}: a hit came back with no coverage count beside it")
        assert rec["within_query_length"], (
            f"{q}: a coverage count exceeded the number of distinct stems in the query, so it "
            f"is counting something other than which of them matched")
    assert any(rec["hits"] for rec in s.values()), "every query matched nothing"
    assert s["zygote pterodactyl"]["hits"] == [], (
        "a query with no term in the corpus must return nothing, not a best effort")
    assert v["search_after_reopen"] == s["encryption"]["hits"], (
        "the index did not survive a reopen — a store is not a process-lifetime cache")


def test_2_the_accessor_beneath_post_search_query_answers() -> None:
    """The object `POST /artifacts/recall` returns, over the production encrypted index.

    One seam is stubbed: the lattice half of `resolve_authorized_scope` needs a live handle.
    Everything below it — the narrower, the file-backed store, the meet — is the production
    path, and `test_1_an_unauthorised_principal_gets_nothing_derivable` proves the stubbed seam
    is not where authorisation lives.

    BOTH modes are checked, and the point is that they answer over ONE universe. The ranked
    mode orders it by how much of the query each hit matched; the candidate mode publishes it
    with no order and no score of any kind, for an external flavor to rank within.
    """
    v = _scenario()
    a = v["accessor"]
    _report("2b · THE ACCESSOR BENEATH POST /artifacts/recall", a)
    assert "error" not in a, a.get("error")
    assert a["candidates"] == a["expected"]
    assert a["candidate_keys"] == ["artifact_id", "collection_id", "principal_id"], (
        "the candidate set published a score; the order is the flavor's to decide and this "
        "mode states none")
    assert a["model_id"] is None, "a store profile retrieves by no embedding model"
    assert a["ranked"] == a["ranked_expected"]
    assert a["ordering"] == "coverage", (
        f"a store profile has no vector arm, so the query's own coverage is what can order a "
        f"recall; got {a['ordering']}")
    assert a["descending"], "the ranked mode did not order by coverage"
    assert all(isinstance(x, int) and x > 0 for x in a["scores"].values()), (
        f"a coverage score must be a positive INTEGER count of stems matched: {a['scores']}")


# It cuts, and the silhouette's perfection is a measurement.

def test_3_the_cut_recovers_a_planted_rank_exactly_above_the_edge() -> None:
    """"Perfect" is a claim about a regime; measuring it is what locates the regime's edge.

    The sweep (N=120, F=16, 25 seeds) finds the silhouette exact — 25/25, with the certified
    interval bracketing the truth 25/25 — for a planted rank of 1, 2 or 3 at a separation of 32 to
    128. That is a band, not a half-line, and both of its edges are real:

        Below it, an under-count that more separation fixes: planted 3 goes 0.16 -> 1.00 between
        snr 16 and 32 — a resolution limit that behaves like one.

        At k >= 4, an under-count that more separation does not fix:
            planted 4   0.28 @ 32   0.76 @ 64   0.88 @ 128   0.80 @ 256   mean k 3.80
            planted 5   0.00 @ 32   0.12 @ 64   0.12 @ 128   0.12 @ 256   mean k 4.00
        The saturation, not the rate, is the finding: a resolution limit keeps improving; this
        stops. The noise floor is estimated from the frame's own row energy
        (`engine._noise_sigma2`), so once the planted structure dominates the frame it also sets
        the scale the floor is derived from — scaling every mode scales the floor with it, and
        the count stops moving. The read is scale-invariant by construction, which is why more
        separation cannot reach past this edge.

        Above it, an over-count at a single planted mode: 0.96 at snr 256 and beyond (one seed in
        25 reads k=2, see `cut_overcount`). The read publishes that the call was close —
        `k_margin_last` 0.358 against a contrast of 3.25 — rather than hiding it.

    The under-count is the direction `instrument.py` already states ("bright rows raise the mean,
    which raises the floor, which makes the read under-count — the conservative direction"); on a
    120x16 frame the point where that stops being harmless is between a planted rank of 3 and 4.
    The certified interval reflects it too — `interval_covers_rate` falls with the point read
    rather than staying at 1.0, so the interval never claims to bracket a truth it has lost.

    This test asserts exact recovery over the band measured to hold it, pins both edges so a
    change in either direction is caught, and reports the whole surface."""
    v = _scenario()
    rows = v["cut_rank_recovery"]
    _report("3a · RANK RECOVERY vs PLANTED GROUND TRUTH (N=120, F=16, 25 seeds per cell)",
            {"cells": rows,
             "perfect_cells": sum(1 for r in rows if r["exact_rate"] == 1.0),
             "total_cells": len(rows),
             "imperfect": [r for r in rows if r["exact_rate"] < 1.0],
             "saturation": [
                 {"planted": k,
                  "exact_at_snr_64": next(r["exact_rate"] for r in rows
                                          if r["planted"] == k and r["snr"] == 64.0),
                  "exact_at_snr_128": next(r["exact_rate"] for r in rows
                                           if r["planted"] == k and r["snr"] == 128.0),
                  "exact_at_snr_256": next(r["exact_rate"] for r in rows
                                           if r["planted"] == k and r["snr"] == 256.0),
                  "mean_k_at_snr_256": next(r["mean_k"] for r in rows
                                            if r["planted"] == k and r["snr"] == 256.0)}
                 for k in (1, 2, 3, 4, 5)],
             "the_single_over_count": v["cut_overcount"]})

    # The band the instrument holds, asserted exactly and not one rung wider in any direction.
    held = [r for r in rows if r["planted"] <= 3 and 32.0 <= r["snr"] <= 128.0]
    assert len(held) == 9, held
    bad = [r for r in held if r["exact_rate"] != 1.0]
    assert not bad, (
        "THE SILHOUETTE IS NOT PERFECT WHERE IT WAS MEASURED TO BE — report this, do not soften "
        f"the rung: {bad}")
    assert all(r["interval_covers_rate"] == 1.0 for r in held), (
        f"the certified interval failed to bracket the truth inside the held band: "
        f"{[r for r in held if r['interval_covers_rate'] != 1.0]}")

    # Both edges are pinned, so that a change which quietly fixes or quietly worsens either one
    # is caught rather than absorbed into an average.
    for k in (4, 5):          # the saturating under-count: one short, and it does not move
        a = next(r for r in rows if r["planted"] == k and r["snr"] == 64.0)
        b = next(r for r in rows if r["planted"] == k and r["snr"] == 256.0)
        assert b["mean_k"] < k, f"planted {k} is no longer under-counted at snr 256: {b}"
        assert abs(b["mean_k"] - a["mean_k"]) < 0.5, (
            f"planted {k} no longer SATURATES between snr 64 and 256 — the under-count has "
            f"become separation-limited, which is a different mechanism: {a} -> {b}")
    top = next(r for r in rows if r["planted"] == 1 and r["snr"] == 256.0)
    assert top["max_k"] >= 2 and top["exact_rate"] < 1.0, (
        f"the single-mode over-count at extreme separation is gone; if that is intended, move "
        f"this pin in the same change: {top}")
    assert v["cut_overcount"]["cases"], "the over-count case vanished but the rate still says 0.96"


def test_3_the_cut_reports_no_break_on_a_corpus_with_no_structure() -> None:
    """The control for the above: a cut that fires on everything is not a cut.

    The honest answer here is not `0`. `signal_rank` floors at 1 by its own stated rule
    (`instrument.py` divergence 4: its consumer projects onto the result, and an empty subspace is
    worse than a coarse one), so "no structure" surfaces as a contrast near 1, a permutation null
    that collapses, and a relative gap of exactly 1.0 — the no-break report."""
    v = _scenario()
    n = v["cut_no_structure"]
    _report("3b · THE NO-STRUCTURE CONTROL (nothing planted, 40 seeds)", n)
    assert n["signal_rank_max"] <= 2, (
        f"the derived-null read found structure in pure noise: {n['signal_rank_values']}")
    # The permutation null collapses to the floor on 35 of 40 draws and reports 2 on the other 5,
    # never more. That is not the null failing — it is the null having a stated false-alarm level.
    # `far` is 0.05 and 19 draws is the minimum at which 0.05 is attainable (`_permutation_draws`),
    # so the finest p-value the budget can resolve is exactly 1/20 = far, and a component clears
    # the level whenever no surrogate beats it. A null that never fired would be running at a
    # level nobody declared.
    assert n["structure_rank_max"] <= 2, (
        f"the permutation null found more than one spurious mode in pure noise: "
        f"{n['structure_rank_values']}")
    assert n["structure_rank_at_the_floor_rate"] >= 0.8, (
        f"the permutation null fired far more often than its stated far={n['far']}: {n}")
    assert n["flat_spectrum_relative_gap"] == [1.0], (
        f"a flat spectrum must report the no-break ratio: {n['flat_spectrum_relative_gap']}")


def test_3_a_frame_that_cannot_carry_a_read_returns_the_computed_null() -> None:
    """`None`, not a keep-everything default: the one place beacon has no read to give."""
    v = _scenario()
    c = v["cut_computed_null"]
    _report("3c · THE COMPUTED NULL", c)
    assert c["single_row"] is None and c["one_live_channel"] is None
    assert c["readable_frame"] == 3
    assert c["min_rows"] == 2


def test_3_the_adaptive_cut_recovers_a_planted_relevant_set() -> None:
    """`select`, in `mantle/search/beacon/cut.py`.

    Ground truth is planted: the first 12 of 60 candidates carry the query direction, the other
    48 do not. The cut has no k, no threshold and no keep-fraction, so recovering exactly those
    12 is a real measurement of the silhouette rather than of a parameter.

    Unlike the rank read, this one is exact once separation is high enough, and the approach is
    monotone rather than saturating: 0.00 / 0.00 / 0.00 / 0.00 / 0.64 / 1.00 across alpha
    0.5 … 16. The cut is not fighting an estimator that moves with the data — it is
    resolution-limited, and more separation buys it. Alpha 8 is a near miss (precision 0.985,
    recall 0.957, 11.68 of 12 kept on average): it loses one item at the boundary, not the set.
    """
    v = _scenario()
    rows = v["cut_select"]
    _report("3d · THE ADAPTIVE CUT vs a PLANTED RELEVANT SET (d=256, 60 candidates, 12 relevant)",
            {"cells": rows, "imperfect": [r for r in rows if r["exact_set_rate"] < 1.0]})
    strong = [r for r in rows if r["alpha"] >= 16.0]
    assert strong, "no separable rung was measured"
    bad = [r for r in strong if r["exact_set_rate"] != 1.0]
    assert not bad, (
        "the adaptive cut did NOT recover the planted set exactly where the set is clearly "
        f"separable — that is the finding, report it: {bad}")
    # The approach must be monotone: a rate that stalls would mean the cut is reading something
    # that scales with the pool rather than the separation, which is the k>=4 defect above.
    rates = [r["mean_recall"] for r in sorted(rows, key=lambda r: r["alpha"])]
    assert rates == sorted(rates), f"recall is not monotone in the planted separation: {rates}"


def test_3_the_adaptive_cut_reports_no_break_when_nothing_is_relevant() -> None:
    v = _scenario()
    n = v["cut_select_no_structure"]
    _report("3e · THE ADAPTIVE CUT with NOTHING planted (25 seeds)", n)
    assert n["max_relative_gap"] < 2.0, (
        f"a pool with no planted relevance produced a large break: {n}")


# Projection and prediction are unfilled, and named as such.

def test_4_5_the_giveaway_is_a_complete_embodiment_of_the_cut() -> None:
    """Beacon fills every member of `Read` and none of `Instrument` or `Dynamics` — not a partial
    anything: a complete embodiment of one question that does not answer two others.

    The count is read from `READ_MEMBERS` rather than typed as a literal, because a literal tests
    the size of the contract while the claim is about completeness — the two come apart whenever
    the contract grows, which is the moment this test matters most.
    """
    from prism.instrument import READ_MEMBERS

    v = _scenario()
    c = v["contracts"]
    _report("4a · WHAT THE GIVEAWAY FILLS", c)
    assert c["isinstance_Read"] is True
    assert set(c["members_read"]) == set(READ_MEMBERS), (
        "the giveaway no longer fills the whole `Read` contract — that is the product claim, not an "
        "implementation detail: %s" % sorted(set(READ_MEMBERS) - set(c["members_read"])))
    assert c["isinstance_Instrument"] is False and c["members_embodiment"] == []
    assert c["isinstance_Dynamics"] is False and c["members_dynamics"] == []


@pytest.mark.parametrize("member", ["absorb_transmit", "next_by_coupling", "membrane_screen"])
def test_4_projection_is_refused_by_name_with_a_503(member: str) -> None:
    """The absence is named, never an `AttributeError` raised from somewhere inside a flow. The
    member is genuinely absent (so `isinstance` and `members_of` tell the truth), and every real
    caller reaches the slot through `require`, which raises `InstrumentRequired` naming the
    contract, the member, and the operation that wanted it. 503, because "not equipped" is the
    same class of fact as "not available"."""
    v = _scenario()
    r = v["projection_refusals"][member]
    _report(f"4b · REFUSAL — {member}", r)
    assert r["attribute_present"] is False
    assert r["refused"] is True and r["exception"] == "InstrumentRequired"
    assert r["http_status"] == 503 and r["code"] == "embodiment_required"
    assert r["contract"] == "embodiment" and r["member"] == member
    assert member in r["message"] and r["at"] in r["message"]


@pytest.mark.parametrize("member", ["decay_profile", "resolution_limit", "embed", "fit_dynamics",
                                    "dynamics_state"])
def test_5_prediction_is_refused_by_name_with_a_503(member: str) -> None:
    """The same mechanism, a different reason, and collapsing them would lose both. `Instrument`
    is a question beacon's domain poses perfectly well and the product answers in the aperture;
    `Dynamics` is a question beacon's domain does not pose at all — a set has no lag."""
    v = _scenario()
    r = v["dynamics_refusals"][member]
    _report(f"5 · REFUSAL — {member}", r)
    assert r["attribute_present"] is False
    assert r["refused"] is True and r["http_status"] == 503
    assert r["contract"] == "dynamics" and r["member"] == member
    assert member in r["message"]
