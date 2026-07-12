# Agience Mantle

**The memory of the Agience system — an encrypted-by-default artifact store and search engine where authorization *is* the encryption.**

Mantle holds knowledge so that it can be trusted by people who weren't there when it was made. Every artifact carries its identity, version history, and provenance inside itself — audit is the data structure, not a bolt-on. Search runs over two encrypted indexes (semantic vector + lexical keyword), and access control is enforced **cryptographically**, not by an ACL filter applied after the fact: each cell of the index is encrypted under a per-cell key derived from the owner's master key; authorization is computed as reachability in a typed graph (the *light cone*); and the identifier the traversal must reach is, by construction, the identifier that selects the keys. If you don't hold the grant, the key is mathematically underivable. Revocation is a single grant edit — no re-encryption, no key rotation.

*Storage gets ciphertext. Grants are keys, not metadata.*

Mantle is one of the three instruments of the Agience system — the **memory**, alongside the **seat** ([`agience-core`](https://github.com/Agience/agience-core), where people and agents work behind a human commit boundary) and the **lens** ([`agience-entroptics`](https://github.com/Agience/agience-entroptics), the entropy-matched measurement instrument). This repository is the **production Mantle service**.

## Layout

| Path | Purpose |
|---|---|
| `mantle/` | The FastAPI service: encrypted MANTLE-SSE lexical + MANTLE vector index over MinIO/S3, hybrid BM25 + kNN retrieval, per-state index segments, light-cone authorization, and governable trusted-issuer auth. Depends on the shared `agience_core` (vendored from **agience-origin**) and talks to Origin/Chorus over the wire (HTTP/MCP). See [mantle/README.md](mantle/README.md). |
| `deploy/` | One-shot key/identity generation + the authority manifest (`deploy/init.py`). |
| `vendor/` | The vendored `agience_core` wheel for offline/local builds. |
| `compose.yaml` | The standalone Mantle stack: init → Arango (graph) + MinIO (content) → Mantle (8081). |
| `compose.override.yaml` | Local dev override (auto-merged): exposes the Arango/MinIO ports for inspection. |
| `.env.example` | Config template — copy to `.env`. |
| `docker-compose.integration.yml` | The full backend stack (Origin + Mantle + gateway) for cross-service integration. |

## Run it

Everything runs in Docker — no host-side installs. One step:

```bash
cp .env.example .env        # adjust if needed (sensible standalone defaults work as-is)
docker compose up -d        # init + Arango + MinIO + Mantle (:8081), with Arango/MinIO exposed
```

(For a portless run that skips the dev override: `docker compose -f compose.yaml up -d`.)

Mantle boots as a pure database layer: no platform seeds, empty type registry. An
application on top (Agience/Origin) provisions data via Mantle's API. Set
`EMBEDDINGS_URI` (in `.env`) to a `/embed` endpoint for vector search; unset = lexical-only.

## Architecture

Mantle is a **verify-only encrypted database**: it serves and verifies data, emits a
change-feed event for every artifact write, and never signs peer calls or embeds on
its own behalf. The canonical design docs live in the **agience-core** repo under
`.dev/features/` — notably the per-state search index, the trust floor + event-driven
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

Bug reports, tests, and hardening contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) first — Mantle has a Docker-only workflow and a security-invariant test discipline that contributions must follow.

Security issues → **connect@agience.ai** (do not open a public issue).

## License

**Apache License 2.0.** See [LICENSE](LICENSE) and [NOTICE](NOTICE).
