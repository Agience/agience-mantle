# Agience Mantle

**The lattice — data + backup.**

Mantle is the lattice the whole universe persists in: the memory of the Agience system — an encrypted-by-default artifact store and search engine where authorization *is* the encryption — and the component responsible for **backup** of that data (backups are Mantle's concern, wherever the scripts currently run from).

Mantle holds knowledge so that it can be trusted by people who weren't there when it was made. Every artifact carries its identity, version history, and provenance inside itself — audit is the data structure. Search runs over an encrypted lexical index (blind-token BM25 / SSE), and access control is enforced **cryptographically**: each cell of the index is encrypted under a per-cell key derived from the owner's master key; authorization is computed as reachability in a typed graph (the *light cone*); and the identifier the traversal must reach is, by construction, the identifier that selects the keys. If you don't hold the grant, the key is mathematically underivable. Revocation is a single grant edit — no re-encryption, no key rotation.

*Storage gets ciphertext. Grants are keys, not metadata — access is enforced cryptographically,
not by an ACL filter applied after the fact.*

Mantle is one of the instruments of the Agience system — the **lattice/memory**, alongside [`agience-beam`](https://github.com/Agience/agience-beam) (the measurement of the energy at a cut — the aperture) and [`agience-entroptics`](https://github.com/Agience/agience-entroptics) (the entropy-matched measurement instrument). This repository is the **production Mantle service**.

## Layout

| Path | Purpose |
|---|---|
| `src/mantle/` | The FastAPI service: the standalone lattice store (one SQLite file + a filesystem CAS, opened in-process — `db/backend.py` → `db/lattice_api` → `db/lattice`), encrypted MANTLE-SSE lexical search (blind-token BM25) over S3 cells, per-state index segments, light-cone authorization, and governable trusted-issuer auth. Depends on the shared `core` foundation (**agience-beam**) and talks to Origin/Chorus over the wire (HTTP/MCP). See [src/mantle/README.md](src/mantle/README.md). |
| `Dockerfile` | The service image. The shared foundation is supplied as a named build context: `docker build --build-context core=../agience-beam -t agience-mantle .` |
| `e2e/` | Blackbox HTTP end-to-end suite — drives a live stack over the wire. See [e2e/README.md](e2e/README.md). |
| `.env.example` | Config template — copy to `.env`. |

*The standalone lattice replaced this repo's compose stack. MinIO lives in **agience-bundle** alone, as the local S3 edge for content and SSE cells.*

## Run it

**Mantle IS the database.** The store is one SQLite file (`MANTLE_LATTICE_PATH`, schema created on open) plus a filesystem CAS, opened in-process — zero external database processes to provision.

Directly, host-side:

```bash
cd src/mantle
pip install -r requirements.txt ../../../agience-beam   # deps + the shared `core` foundation
KEYS_DIR=<keys-dir> MANTLE_LATTICE_PATH=./mantle.db uvicorn main:app --port 8081
```

Or as part of the full stack via **agience-bundle**, where the mantle container keeps the lattice (SQLite file + CAS) on a bind-mounted volume:

```bash
cd ../agience-bundle
docker compose --env-file ../agience-origin/.env -f docker-compose.local.yml up -d --build
```

Mantle boots as a pure database layer with an empty type registry. An
application on top (Agience/Origin) provisions data via Mantle's API. Search is
lexical (BM25/SSE); there is no embeddings provider (no-models rule).

## Architecture

Mantle is a **verify-only encrypted database** — it never signs peer calls and never embeds on its
own behalf: it serves and verifies data and emits a
change-feed event for every artifact write. The canonical design docs live in the **agience-pharos** repo under
`dev-legacy/dev-features/` — notably the per-state search index, the trust floor + event-driven
architecture, and trusted issuers as artifacts (`vnd.agience.issuer+json`). Mantle
verifies tokens from any configured OIDC issuer via one generic verifier, with the
authority manifest as a bootstrap seed.

### Security invariants

Four properties hold across the codebase, are asserted by tests, and must survive every change:

1. **Geometry never authorizes.** Embeddings and routing rank *within* an already-authorized candidate set; they never widen one. The routing path receives no key material and runs strictly before any key request.
2. **Authorization is decided only by the light cone and grants.** An unauthorized context is never routed to, padded in, or key-issued. Access is default-deny; there is no owner fast-path — even the creator holds an explicit, revocable grant.
3. **Ciphertext is bound to its identity.** Every cell is AEAD-encrypted with associated data bound to its context and cluster, so a blob presented under the wrong key or moved to the wrong slot fails authentication before deserialization.
4. **Revocation requires no re-encryption.** Removing a grant alone prevents routing to, deriving keys for, and decrypting the affected cells.

If a contribution weakens any of these, it will not be merged — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Contributing

Bug reports, tests, and hardening contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) first — Mantle has a security-invariant test discipline that contributions must follow.

Security issues → **connect@agience.ai** (do not open a public issue).

## License

**Apache License 2.0.** See [LICENSE](LICENSE) and [NOTICE](NOTICE).
