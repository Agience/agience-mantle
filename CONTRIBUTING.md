# Contributing to Agience Mantle

Thank you for considering a contribution. Mantle is the encrypted store beneath the Agience system — the database an organization's trust ultimately rests on — so contributions here follow a stricter discipline than a typical service repo. Read this whole document before writing code.

---

## Before You Start

**Sign off every commit** (`git commit -s`) to certify the [Developer Certificate of Origin](https://developercertificate.org/) (DCO). By contributing you agree your contribution is licensed under the Apache License 2.0 (per Section 5 of the license), including the Section 3 patent grant for your contribution. For substantial contributions, Ikailo Inc. may additionally request a signed [CLA](https://github.com/Agience/agience-beam/blob/main/CLA.md) — the bot will tell you if it applies.

**Read the security invariants** in [README.md](README.md#security-invariants). They are the contract this repository exists to keep. A change that weakens one will not be merged.

**Canonical design docs live in agience-pharos** under `dev-legacy/dev-features/` (the per-state search index, trust floor + event-driven architecture, trusted issuers as artifacts, the canonical search architecture). Read the relevant doc before changing the index or auth paths, and keep it current when you do.

---

## The Workflow

**Mantle is the standalone database** — one SQLite file (`MANTLE_LATTICE_PATH`) plus a filesystem CAS, opened in-process with zero external database processes to provision, so the
test suite runs host-side:

```bash
cd src/mantle
KEYS_DIR=<tmp-dir> python -m pytest tests db/lattice -q
# baseline 2026-07-22: 1504 passed, 57 skipped
```

The service image is built from the root `Dockerfile` (the shared `core` foundation is supplied as a named build context: `docker build --build-context core=../agience-beam -t agience-mantle .`). The runnable stack — Origin + Mantle + MinIO (the local S3 edge for content/SSE cells) — lives in **agience-bundle** (`docker compose -f docker-compose.local.yml up -d --build`).

A set of inherited tests is quarantined in `tests/conftest.py` at the repo root, which records the measured reason for each. **Record the green/failing baseline before your change and compare after** — a contribution that flips a quarantined test into relevance should say so explicitly.

## Live Verification Is Mandatory for Auth and Index Changes

Mantle is the database's auth core. Token verification, the per-state encrypted index, light-cone authorization, and trusted-issuer resolution must be exercised against a **running stack**:

1. Rebuild the image and recreate the stack (from `agience-bundle`: `docker compose -f docker-compose.local.yml up -d --build --force-recreate`).
2. Run an end-to-end check that exercises your change through the real API (index → query → verify scope; grant → revoke → verify the ciphertext goes dark) — the blackbox suite in `e2e/` covers most of this surface.
3. Include what you verified, and how, in the PR description.

## Security-Invariant Tests

If your change touches routing, key derivation, the light cone, grants, or cell encryption, it must **add or extend an invariant test**. The canonical assertions:

- the geometry/routing path receives no key material and runs strictly before partitioning, encryption, and any key request;
- an unauthorized context is never routed to, padded in, or key-issued;
- a tampered, relocated, or wrong-context cell fails authentication before deserialization;
- revoking a grant alone denies routing, key derivation, and decryption — with no re-encryption.

---

## How to Contribute

**Bugs:** open an issue with reproduction steps, expected vs. actual behavior, and environment. **Security vulnerabilities: do not open a public issue** — email **connect@agience.ai**; we respond within 5 business days.

**Features:** open an issue first, describing the problem, how it fits the verify-only database model, and whether it holds the security invariants. A feature PR is accepted after its issue is acknowledged.

**Code:** fork, branch from `main`, sign off every commit (`git commit -s`), and open a PR. Commit messages follow the lightweight conventional format used across Agience repos (`fix:`, `feat(scope):`, `docs:`, `refactor:`, `test:`, `chore:` — see [agience-beam CONTRIBUTING](https://github.com/Agience/agience-beam/blob/main/CONTRIBUTING.md) for the full standard).

## Code Conventions

- Keep modules small and named after the concept they implement; search the codebase for an existing pattern before inventing an abstraction.
- Strict layering: routers validate and delegate → services orchestrate → DB/S3 adapters. Routers never call adapters directly.
- Mantle never signs peer calls and never embeds on its own behalf — it is verify-only. Do not add either capability.
- No new external dependency or network call without prior discussion in an issue.

## What Gets Accepted

Bug fixes with tests; invariant-test hardening; documentation corrections; performance work that leaves the security posture untouched and proves it. **Out of scope without prior discussion:** changes to the crypto path or key-derivation strings, new authorization mechanisms, Docker/compose changes to the core stack, and anything that moves authorization decisions into the geometry path.

## Pull Request Checklist

- [ ] CLA signed (bot checks on PR open)
- [ ] Test suite run (`cd src/mantle && KEYS_DIR=<tmp> python -m pytest tests db/lattice -q`), baseline compared
- [ ] Auth/index changes live-verified against a running stack (described in the PR)
- [ ] Invariant tests added/extended where the change touches routing, keys, grants, or cells
- [ ] Relevant canonical design doc in agience-beam updated if behavior changed
- [ ] Commit messages follow the convention; commits signed off

---

## Code of Conduct

Be respectful. Contributions, issues, and discussions must remain professional and constructive. Harassment, discrimination, or bad-faith behavior will result in removal from the project.

## License

By contributing you agree to the DCO above.

---

*Ikailo Inc., Canada. Questions: connect@agience.ai*
