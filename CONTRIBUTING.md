# Contributing to Agience Mantle

Thank you for considering a contribution. Mantle is the encrypted store beneath the Agience system — the database an organization's trust ultimately rests on — so contributions here follow a stricter discipline than a typical service repo. Read this whole document before writing code.

---

## Before You Start

**Sign off every commit** (`git commit -s`) to certify the [Developer Certificate of Origin](https://developercertificate.org/) (DCO). Sign-off is checked by a maintainer at review, not by automation, so a PR whose commits are unsigned will be sent back rather than flagged on open. By contributing you agree your contribution is licensed under the Apache License 2.0 (per Section 5 of the license), including the Section 3 patent grant for your contribution. For substantial contributions, Ikailo Inc. may additionally request a signed CLA; a maintainer will raise that on the PR and supply the current text.

**Read the security invariants** in [README.md](README.md#security-invariants). They are the contract this repository exists to keep. A change that weakens one will not be merged.

**The design canon is internal to Ikailo** and is not distributed with this repository, so nothing here asks you to read or update it. Everything a contribution must hold is in this repo: the security invariants in the README, the invariant tests listed below, and the architecture section of [README.md](README.md#architecture). Where a change would cross a design decision the README does not cover, open an issue and a maintainer will answer from the canon.

---

## The Workflow

**Mantle is the standalone database** — one SQLite file (`MANTLE_LATTICE_PATH`) plus a filesystem CAS, opened in-process with zero external database processes to provision, so the
test suite runs host-side, from the repo root:

```bash
KEYS_DIR=<tmp-dir> MANTLE_LATTICE_PATH=<tmp-file> python -m pytest tests src/mantle -q
```

Both roots are named because both hold tests. The unit suite lives at `tests/`; `src/mantle/db/` keeps its own in-tree tests because they certify the embeddable surface, and a check that cannot run without the package it exists to prove unnecessary would measure nothing. `pyproject.toml` holds the only pytest config and its `testpaths` names the same two roots, so bare `pytest` from the repo root is equivalent. Point `KEYS_DIR` and `MANTLE_LATTICE_PATH` at throwaway locations so a run does not write a keyset and a store into your checkout.

CI runs two jobs, and the first is the narrower one. `embeddable-surface` installs the **base** distribution only (`pip install .`) and runs `python -m pytest src/mantle/db -q`; it is what holds `pip install agience-mantle` to stdlib plus `cryptography`, so a new module-level import of a third-party or sibling package fails there and nowhere else. `full-suite` then installs `[dev]` alongside a checked-out `agience-prism` and runs both roots. A change that passes locally can still fail `embeddable-surface`, which is the point of it.

The suite needs **`agience-prism`** on the path. It is published on PyPI, so `pip install -e '.[dev]'` resolves it; to work on both at once, clone it beside this repo and install it editable (`pip install -e ../agience-prism/py`), which takes precedence. See the README's Quick Start for the layout.

Mantle ships as a Python distribution and runs as an ordinary process — there is no image to build. `pip install -e '.[service]'` followed by `mantle-serve` is the whole of it, against a throwaway `KEYS_DIR` and `MANTLE_LATTICE_PATH`. That is what an outside contributor is expected to run.

A fuller stack — Mantle behind an OIDC issuer, with an S3-compatible bucket for content and SSE cells — is what maintainers run before merge. It lives in Ikailo's internal deployment repo and is not distributed here, so nothing below asks you to bring it up.

Nothing is quarantined: `collect_ignore` in `tests/conftest.py` is empty, and every test in both roots is collected. `src/mantle/db/test_collect_ignore_is_honest.py` keeps it that way — an entry hides a file from collection entirely, so it is reported as neither run nor skipped, and that test measures each entry rather than taking its stated reason on trust. Adding one means proving the file is genuinely uncollectable. **Record the green/failing baseline before your change and compare after.**

## Live Verification for Auth and Index Changes

Mantle is the database's auth core. Token verification, the per-state encrypted index, light-cone authorization, and trusted-issuer resolution must be exercised against a **running server**, not only in unit tests:

1. Run `mantle-serve` against a throwaway `KEYS_DIR` and lattice.
2. Run an end-to-end check that exercises your change through the real API (index → query → verify scope; grant → revoke → verify the ciphertext goes dark). The blackbox suite in `tests/e2e/` covers most of this surface and is configured entirely by environment — `E2E_MANTLE_URL`, `E2E_ORIGIN_URL`, `E2E_AUTHORITY_ISSUER` — so it runs against whatever stack you have, including your own issuer.
3. Include what you verified, and how, in the PR description.

Maintainers re-run this against the full internal stack before merge. Your job is to show the change works against a real server; the cross-service run is ours.

## Security-Invariant Tests

If your change touches routing, key derivation, the light cone, grants, or cell encryption, it must **add or extend an invariant test**. The canonical assertions:

- the geometry/routing path receives no key material and runs strictly before partitioning, encryption, and any key request;
- an unauthorized context is never routed to, padded in, or key-issued;
- the light-cone resolve never reaches further than `services/dependencies.check_access` would allow — the context walk carries a required authority ceiling and cannot widen the grant-derived set;
- recall cuts at artifact granularity as well as collection, so an artifact-scoped grant exposes one artifact rather than everything filed beside it;
- a tampered, relocated, or wrong-context cell fails authentication before deserialization;
- revoking a grant alone denies routing, key derivation, and decryption — with no re-encryption.

### Call the attenuation operator; do not write a second one

Permission-mask intersection is **single-sourced** in `src/mantle/attenuation.py`. It is a bounded meet-semilattice over the CRUDEASIO action set: `Mask` is the one type, `&` is the meet, deny is absorbing by construction, and both storage encodings — the `edge.propagate` TEXT column and `Grant`'s nine `can_*` booleans — are codecs onto it that round-trip. Use `Mask.from_propagate` / `Mask.from_flags` to decode, `&`, `attenuate()` or `compose()` to combine, `allows()` at an enforcement point, and `propagates()` for an edge column.

Do not intersect masks inline, and do not add a helper named `intersect`, `meet`, `narrow` or `attenuate` anywhere else. `tests/test_attenuation_is_single_sourced.py` sweeps `src/mantle` by AST for both shapes a re-implementation takes, and demonstrates the guard firing on a seeded copy of each; `tests/test_attenuation_algebra.py` proves the laws exhaustively. A second copy does not have to be wrong the day it is written — it only has to drift.

Two things this operator is **not**. Invariant #1 ("geometry never authorizes") is the same principle in a different register: it is an ordering and import-boundary discipline, not a mask intersection, and it stays enforced where it already is. Event *delivery* is not attenuation either — one write notifies many subscribers, so fan-out amplifies; only event *visibility* narrows, and only that half may reach for the meet.

---

## How to Contribute

**Bugs:** open an issue with reproduction steps, expected vs. actual behavior, and environment. **Security vulnerabilities: do not open a public issue** — email **connect@agience.ai**; we respond within 5 business days.

**Features:** open an issue first, describing the problem, how it fits the verify-only database model, and whether it holds the security invariants. A feature PR is accepted after its issue is acknowledged.

**Code:** fork, branch from `main`, sign off every commit (`git commit -s`), and open a PR. Commit messages follow the lightweight conventional format used across Agience repos: `fix:`, `feat(scope):`, `docs:`, `refactor:`, `test:`, `chore:`.

## Code Conventions

- Keep modules small and named after the concept they implement; search the codebase for an existing pattern before inventing an abstraction.
- Strict layering: routers validate and delegate → services orchestrate → DB/S3 adapters. Routers never call adapters directly.
- Mantle never embeds on its own behalf and never fabricates provenance — the data plane is verify-only. Do not add either capability. (The one outbound signature in `services/peer_signing.py` — a service JWT saying "Mantle is calling" — is the transport, not the data plane. Mantle signs no user token and mints no delegation; it only verifies delegations, in `services/dependencies.py`.)
- No new external dependency or network call without prior discussion in an issue.

## What Gets Accepted

Bug fixes with tests; invariant-test hardening; documentation corrections; performance work that leaves the security posture untouched and proves it. **Out of scope without prior discussion:** changes to the crypto path or key-derivation strings, new authorization mechanisms, packaging and release changes, and anything that moves authorization decisions into the geometry path.

## Pull Request Checklist

- [ ] Test suite run (`KEYS_DIR=<tmp> MANTLE_LATTICE_PATH=<tmp-file> python -m pytest tests src/mantle -q` from the repo root), baseline compared
- [ ] Auth/index changes live-verified against a running stack (described in the PR)
- [ ] Invariant tests added/extended where the change touches routing, keys, grants, or cells
- [ ] Any permission-mask intersection goes through `mantle.attenuation`, not a new local copy
- [ ] No new module-level third-party or sibling-package import under `src/mantle/db/` (the `embeddable-surface` CI job)
- [ ] Commit messages follow the convention; every commit signed off (`git commit -s`)

---

## Versioning

Mantle follows [SemVer](https://semver.org/). The version lives in exactly two places, and a version-bump PR updates both together: `project.version` in [pyproject.toml](pyproject.toml) and the version badge at the top of [README.md](README.md).

## Code of Conduct

Be respectful. Contributions, issues, and discussions must remain professional and constructive. Harassment, discrimination, or bad-faith behavior will result in removal from the project.

## License

Mantle is **Apache License 2.0** (see [LICENSE](LICENSE), [NOTICE](NOTICE)). By contributing you agree to the DCO above.

---

*Ikailo Inc., Canada. Questions: connect@agience.ai*
