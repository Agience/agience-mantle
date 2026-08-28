# Agience Mantle

[![Version](https://img.shields.io/badge/version-0.1.0-blue)](pyproject.toml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![CI](https://github.com/Agience/agience-mantle/actions/workflows/build-image.yml/badge.svg)](https://github.com/Agience/agience-mantle/actions/workflows/build-image.yml)

**The lattice — where the data lives.**

Mantle is the lattice the whole universe persists in: the memory of the Agience system — an encrypted-by-default artifact store and search engine where authorization *is* the encryption.

Mantle **holds** that data; it does not back it up. There is no backup, snapshot, restore or corruption-detection machinery in this repository. Durability is an operator responsibility, and [Backing a node up](#backing-a-node-up) is the procedure — a runbook, not a feature.

Mantle holds knowledge so that it can be trusted by people who weren't there when it was made. Every artifact carries its identity, version history, and provenance inside itself — audit is the data structure, for everything written through the system; a bulk import supplies no provenance and gets none. Search runs over an encrypted lexical index — blind-token MANTLE-SSE, covering **5.9%** of our reference corpus today and growing, with the remainder served by a plaintext lexical index — and access control is enforced **cryptographically**: each cell of the index is encrypted under a per-cell key derived from the owner's master key, and authorization is computed as reachability in a typed graph (the *light cone*).

**Reachability decides which keys are issued — it does not derive them.** A grant is not a rule the storage layer chooses to honour; it is what decides whether a key is handed out at all. The cell key itself is derived on demand and never persisted, so there is nothing at rest to take, and one piece of code decides both what a search may touch and what key is issued — so the two cannot disagree silently. Revocation is a single grant edit — no re-encryption, no key rotation — effective within the authorization cache's window, 30 seconds by default and disableable outright where that matters.

*Grants are keys, not metadata — access is enforced cryptographically, not by an ACL filter applied
after the fact.*

**What is encrypted, and what is not.** Content is encrypted per principal and bound cryptographically to the collection it was written for; on a live store, every content object measured is encrypted at rest, in fact wrapped twice. What the store keeps in the clear is what it must read to *find* things: the offer text the lexical index reads, and the identifiers in the posting store. The blind-token index exists to close exactly that, and covers **5.9%** of our reference corpus today — the rest is served by a plaintext lexical index. That number is small and it is the real one.

Mantle is one of the instruments of the Agience system. This repository is the **production Mantle service**.

## Layout

| Path | Purpose |
|---|---|
| `src/mantle/` | The FastAPI service: the standalone lattice store (one SQLite file + a filesystem CAS, opened in-process — `db/backend.py` → `db/lattice_api.py` → `db/vertex.py` + `db/edge.py`), encrypted retrieval on both arms — MANTLE-SSE blind-token narrowing and anchor-routed vector cells, on object storage or the local disk; the vector arm is inert until an AnchorSet is seeded (see [below](#-semantic-recall-is-inert-until-you-seed-an-anchorset)) — per-state index segments, light-cone authorization, and governable trusted-issuer auth. Depends on **`agience-prism-py`** and talks to Origin over the wire (HTTP/MCP) — `clients/origin_client.py` is the only outbound peer client. |
| `build/Dockerfile` | The service image. The shared foundation is `agience-prism-py`, supplied as a named build context: `docker build -f build/Dockerfile --build-context prism=../agience-prism/py -t agience-mantle .` |
| `docker-compose.yml` | A single-service stack for smoke-testing that image without the rest of the platform. |
| `tests/e2e/` | Blackbox HTTP end-to-end suite — drives a live stack over the wire. See [tests/e2e/README.md](tests/e2e/README.md). |
| `.env.example` | Config template — copy to `.env`. Only `MANTLE_LATTICE_PATH` and `KEYS_DIR` are set outright; everything else is commented out, so an untouched copy runs on defaults. |

*MinIO lives in **agience-observe** alone, as the local S3 edge for content and SSE cells.*

### Subsystems under `src/mantle/`

| Path | Purpose |
|---|---|
| `db/` | The standalone lattice store. `backend.py` is the one import point → `lattice_api.py` → `vertex.py`/`edge.py`/`seq.py`/`schema.py` (SQLite + filesystem CAS), plus the S3 content adapter. `doc_boundary.py` is the write chokepoint: content envelope crypto and the change event, in one place. |
| `routers/` | The five FastAPI routers `main.py` mounts — `artifacts`, `grants`, `events`, `system`, `mcp`. Thin and type-agnostic: validate, delegate, return. |
| `services/` | Orchestration — workspaces, collections, grants, content, contexts, OIDC, seed provisioning, plus `peer_signing.py` (the one outbound signature, a service JWT). `content_crypto.py` is the per-principal content envelope, which is also all there is to a secret. `acting_principal.py` answers who is acting; `principal.py` answers what artifact a principal IS — a person, or a foundation entity for an author that is not a human. |
| `api/` | Pydantic request/response models, grouped by domain — including `api/vectors.py`, the shape validation for writer-supplied vectors. |
| `entities/` | Entity models and serialization. A collection *is* an artifact; `entities/collection.py` says so literally. `context.py` and `subscription.py` are the same move again: a role an artifact plays, discriminated by `content_type`. |
| `search/` | Retrieval. `embeddings.py` and `embeddings_cache.py` are the vector arm's provider facade and its long-term cache; `search/mantle/sse/` is the encrypted lexical arm (blind-token narrowing), which runs on every install; `search/mantle/lightcone.py` is authorization as reachability; `search/anchors/` and `search/beacon/` are the semantic arm and its result cut — `search/anchors/store.py` loads a client-seeded AnchorSet and never derives, grows or reconciles one, so the semantic arm stays dark until a set is seeded; `search/ingest/` is the indexing queue. |
| `attenuation.py` | The one authorization meet — CRUDEASIO masks, deny absorbing, composed along every path. One of the four modules in the package root, beside `__init__.py` (the BLAS pin), `main.py` (the app) and `config.py` (the settings every layer reads). |
| `events/` | The change feed: `event_bus.py` is in-process fan-out and a durable log with cursor replay; `event_backplane.py` is the optional Redis/MQTT back-plane for multi-process nodes. |
| `system/` | Boot and operations — `logging_utils.py` and its `uvicorn_log_config.json`, `runner_hooks.py` (what the store asks of a runner, injected rather than imported), and the `manage_*.py` bootstrap/seed/addon/anchor operations. |
| `ui/` | Server-rendered browse pages — `browse_page.py`. |
| `mesh/` | The peering plane: content-addressed Ed25519-signed shards, anchor-keyed regions, incremental Merkle sync. |
| `oci/` | An OCI registry over the lattice — an image is a collection, a blob is content. No side-car registry. |
| `shard/` | Persistence beneath the store: local cache and its regions, content tiering, curation, and erasure. |
| `clients/` | The wire between Mantle and someone else. `origin_client.py` is the only outbound peer client; `artifact_helpers.py` is the consumer's side — `content_type` ⇄ `mimeType` for the MCP servers that call Mantle, used by `agience-chorus`. |
| `scripts/` | Operator and developer CLIs — `dev_init_keys.py`, `dev_mint_token.py` (the credential a standalone node had no other way to obtain), `serve.py`, `manage_erasure.py`, CAS rekey, usage snapshots. |
| `tools/` | One-shot migrations — `migrate_env_to_db.py`. |

## Run it

**Mantle IS the database.** The store is one SQLite file (`MANTLE_LATTICE_PATH`, schema created on open) plus a filesystem CAS, opened in-process — zero external database processes to provision.

### Prerequisite: a sibling `agience-prism` checkout

`[service]` (and `[semantic]`, and `[dev]`) require **`agience-prism-py`**, which is not on PyPI. Every
delivery path resolves it from a sibling checkout, so clone it beside this repo first — the
directory layout below is what the `docker build` command and the relative `pip install` path both
assume:

```text
<workspace>/
├── agience-mantle/     ← this repo
└── agience-prism/py/   ← the `agience-prism-py` distribution
```

```bash
# The repository is the one `.github/workflows/build-image.yml` checks out as the `prism` build
# context: `<this repo's owner>/agience-prism-py`.
git clone https://github.com/Agience/agience-prism-py ../agience-prism/py
pip install -e ../agience-prism/py
```

Without it, `pip install -e '.[service]'` cannot resolve, and the app fails at import — `prism` is
the trust floor `main.py`'s key initialization calls into.

### Local

```bash
pip install -e ../agience-prism/py    # the prerequisite above
pip install -e '.[service]'           # mantle + its service extra; base dep is just `cryptography`

# An empty KEYS_DIR does not boot: the lifespan loads key material that nothing else in this repo
# writes. This generates a throwaway keyset — local development only, never a deployment.
mantle-init-keys --keys-dir ./.data/keys

KEYS_DIR=./.data/keys MANTLE_LATTICE_PATH=./.data/mantle.db mantle-serve --port 8081
```

`mantle-serve`, `mantle-init-keys` and `mantle-token` are console scripts any install puts on the
path; `mantle-serve` needs `[service]` for uvicorn, the other two need only the base install.
Spelled out they are `uvicorn mantle.main:app`, `python src/mantle/scripts/dev_init_keys.py` and
`python src/mantle/scripts/dev_mint_token.py` — the latter two run straight from a checkout with
nothing installed but `cryptography`.

### Connect a client

Six commands from a fresh install to an MCP client that stores something and finds it again. A
standalone node is a **complete** node: no Origin, no S3, no AnchorSet and no bootstrap step.

```bash
pip install -e ../agience-prism/py                     # the prerequisite above
pip install 'agience-mantle[service]'

mantle-init-keys --keys-dir ./.data/keys

AGIENCE_BASE_DIR=$PWD KEYS_DIR=./.data/keys MANTLE_LATTICE_PATH=./.data/mantle.db \
    mantle-serve --port 8081
```

**Set `AGIENCE_BASE_DIR` on a pip-installed node.** It is the root every derived default hangs
off — the SSE index (`.data/mantle-sse`), the encrypted cells (`.data/mantle-cells`), the embeddings
cache (`.data/mantle/`), `KEYS_DIR` and `MANTLE_LATTICE_PATH`. Unset, an installed node derives it
from **the directory you start it in**, because the alternative — the directory the package was
installed into, `site-packages` — is a tree the next `pip install --upgrade` rewrites, and the
indexes are **data, not cache**: a full rebuild is measured in days-to-weeks. The working directory
is a floor rather than a plan, though. Start the same node from somewhere else and it derives a
different root and comes up healthy serving an empty universe, so say it outright, as the command
above does. `MANTLE_LATTICE_PATH` and `KEYS_DIR` do not save you on their own: they move two of the
four parts a node is made of, and [Backing a node up](#backing-a-node-up) is the list of all four.

A checkout — including an editable install, which is the same tree seen through a finder — keeps
deriving the repo root, so `.data/` stays beside `src/` and nothing about developing here changes.

Then, in a second shell, mint a credential. `KEYS_DIR` already holds one: `mantle-init-keys` wrote
the signing key *and* the trust anchor that verifies it, so the node has trusted its own key since
its first boot — this command is the first thing to sign a **user** token against it.

```bash
mantle-token --keys-dir ./.data/keys
```

```text
Minted a user token, signed by /home/you/node/.data/keys/mantle.private.pem and trusted by this
node's own authority.manifest.json anchor. DEVELOPMENT ONLY - these keys have no custody.

    subject   d99d859b-876e-57f6-b196-7b22fa54335c
    audience  http://localhost:8080   (config.AUTHORITY_ISSUER - what the verifier requires)
    expires   2026-08-12T13:55:49Z   (12 hours)

Add it to Claude Code:

    claude mcp add --transport http mantle http://localhost:8081/mcp --header "Authorization: Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6Im1hbnRsZS0xIiwidHlwIjoiSldUIn0.eyJhdWQiOi..."

Or send the header yourself:

    Authorization: Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6Im1hbnRsZS0xIiwidHlwIjoiSldUIn0.eyJhdWQiOi...

The subject is derived from this keyset's instance.uuid, so re-running mints for the
SAME person and everything the last token stored stays reachable. ...
```

Three claims in that block are worth reading twice. **`audience`** is `config.AUTHORITY_ISSUER`, read
from the same module attribute the verifier compares against — `http://localhost:8080` here is the
*name* a token must carry, not a host anything dials, and nothing needs to be running there.
**`expires`** is `services/auth_service.ACCESS_TOKEN_EXPIRE_HOURS`, this package's own declared
lifetime for an end-user access token, not a number the command picked. **`subject`** is
`uuid5(instance.uuid, "mantle/local-user")` — derived from the keyset, so it is the same person on
every run; a random subject would mint a new principal holding no grants and strand everything the
previous token stored. `--subject <label>` names a second identity on the same keyset.

Paste that `claude mcp add` line and the client is connected. Everything below is the same thing
with `curl`, so you can see the wire:

```bash
TOKEN=$(mantle-token --keys-dir ./.data/keys --token-only)

curl -s http://localhost:8081/mcp \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"create_artifact",
       "arguments":{"name":"Kickoff notes","content_type":"text/markdown",
                    "content":"We agreed to ship the encrypted lexical arm first."}}}'
```

```json
{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"…"}],
 "structuredContent":{"id":"df65a79f-1a57-4a5b-b10d-62f4032557fc","name":"Kickoff notes",
   "content_type":"text/markdown","state":"committed","collection_id":"",
   "created_by":"d99d859b-876e-57f6-b196-7b22fa54335c",
   "created_time":"2026-08-12T01:56:01.828633+00:00"}}}
```

`created_by` is the token's `subject`, so the credential really is the principal — the creator gets
an owner grant, which is why the next call can find it.

**Send `content_type`.** Omitted, it defaults to `application/vnd.agience.collection+json` — the
label for a *container* — so a stored conversation comes back as a collection and every `type:`
filter that would have found it misses. Measured, not assumed:
`create_artifact` with `content_type` left out answers `"content_type":
"application/vnd.agience.collection+json"`.

**Send `identity` for anything you will store more than once.** It names the *thing* the
artifact is of — `file:/repo/README.md`, `session:7c7bcb7b` — and the artifact's id is derived
from it (`services/artifact_identity`), so the write is **idempotent**: storing the same thing
again updates that one artifact instead of leaving a second copy.

```bash
curl -s http://localhost:8081/mcp \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"create_artifact",
       "arguments":{"identity":"file:/repo/NOTES.md","name":"Notes",
                    "content_type":"text/markdown","content":"Second revision."}}}'
```

Run that twice and there is one artifact holding `Second revision.`; run the same pair without
`identity` and there are two, with nothing in either saying which is current. Without it the id
is a fresh `uuid4` per call, so the only way to update rather than duplicate is for the client
to remember the id it was given — and a write whose reply is lost **still succeeds here**,
leaving the client with nothing recorded and the next write creating a second root that nothing
reconciles.

The derivation includes the calling principal, so the same name from a different principal is a
different artifact: two people capturing their own `README.md` cannot collide, and converging on
one artifact stays a deliberate act — a grant — rather than an accident of filenames. `identity`
is top-level only; a member of a collection has a draft/committed lifecycle with more than one
live version, so combining it with `container_id` is a 400 naming the reason rather than a
silently-dropped argument.

```bash
curl -s http://localhost:8081/mcp \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"recall",
       "arguments":{"query_text":"encrypted lexical arm"}}}'
```

```json
{"jsonrpc":"2.0","id":2,"result":{"structuredContent":{
  "hits":[{"id":"df65a79f-1a57-4a5b-b10d-62f4032557fc","score":3.0,
           "root_id":"df65a79f-1a57-4a5b-b10d-62f4032557fc","collection_id":"",
           "title":null,"description":null,
           "content":"We agreed to ship the encrypted lexical arm first."}],
  "total":1,"ordering":"coverage","applied_filters":[],"corrections":[],
  "query_text":"encrypted lexical arm","size":20,"from":0}}}
```

The results are under **`hits`**. `ordering: "coverage"` is the lexical arm answering — `score` is
the count of distinct query stems that hit carries, not a relevance measure. `"semantic"` requires a
seeded AnchorSet ([below](#-semantic-recall-is-inert-until-you-seed-an-anchorset)). `title` is null
because `name` and `title` are different fields: `title` comes from the `context` JSON, which this
call did not send.

**`KEYS_DIR` is the root credential of a standalone node.** `mantle-token` does not create that
exposure, it names it: anyone who can read that directory can already mint this token by hand, and
read access to it is full access to the store — bounded by no grant, no revocation and no expiry.
Back it up separately and under different custody, as [Backing a node up](#backing-a-node-up) says.

**There is no OAuth flow to complete here.** A standalone Mantle serves exactly one document of
the OAuth surface — `/.well-known/oauth-protected-resource` — and none of the endpoints: no
`/.well-known/oauth-authorization-server`, no `/authorize`, no `/token`, no dynamic client
registration. A standards-compliant MCP OAuth flow therefore **cannot** complete against it, and the
static `Authorization` header above is the supported path.

`config.authorization_servers()` names an authority only once one has been **declared** — an
`AUTHORITY_ISSUER` or `ORIGIN_URI` in the environment, a `branding.origin_uri` row that differs from
the default, or a configured `AGIENCE_TRUSTED_ISSUERS`. An undeclared node omits the key rather than
naming a server it cannot serve. Point a node at a real issuer and the document names it.

### Docker

```bash
docker build -f build/Dockerfile --build-context prism=../agience-prism/py -t agience-mantle .
python src/mantle/scripts/dev_init_keys.py --keys-dir ./.data/keys
docker compose up
```

`docker-compose.yml` in this repo runs mantle alone against a bind-mounted `./.data`. The full
platform stack — Origin + Mantle + MinIO — lives in **agience-observe**:

```bash
cd ../agience-observe
docker compose --env-file ../agience-origin/.env -f docker-compose.local.yml up -d --build
```

Mantle boots as a pure database layer with an empty type registry. An
application on top (Agience/Origin) provisions data via Mantle's API.

### Semantic recall is inert until you seed an AnchorSet

**A node you just installed answers `POST /artifacts/recall` from the lexical arm only.** Both
arms are wired and both have somewhere to write — the encrypted vector cells work on local disk
with no bucket — but the semantic arm routes every vector against an **AnchorSet**, the shared
coordinate system, and a fresh node has none. You seed it.

The whole contract is three steps:

1. **Seed a set.** `python -m mantle.system.manage_anchors --action load --path anchors.json`
2. **Send query vectors in that set's space** — the `space_id` you supply must equal the set's
   `model_id`, and the width must equal its `dim`.
3. **Read ranked results.** `POST /artifacts/recall` returns them with `ordering: "semantic"`
   and a cosine on every hit.

**Mantle does not derive, grow, reconcile or crosswalk a coordinate system.** That is deliberate,
for two independent reasons:

- **Locally-derived anchors would not be shared.** An anchor id is content-addressed over
  `(label, model_id, embedding)`, so anchors fitted to whatever corpus a node happens to hold
  mint region ids no other node computes. Two deployments would each route confidently, into
  disjoint cells, with no overlap — an index that looks healthy and shares with nobody.
- **Anchors are vectors, and Mantle runs no model.** The no-models rule (`search/embeddings.py`)
  means this process has nothing that could produce them, or that could fit a projection between
  two spaces. Vectors arrive from a caller.

So the client owns the coordinate system end to end: it authors the set, it names the space, and
one node serves exactly one space. A query in any other space is refused with a 400 naming both.

Until a set is seeded:

| Path | What happens |
|---|---|
| An artifact write | Succeeds and is indexed **lexically**. The vector arm returns `skipped` and logs a `WARNING` naming the missing AnchorSet (`search/ingest/pipeline_unified.py`). |
| `POST /artifacts/recall` | Narrows on the query's terms, then answers most-recently-updated first: `ordering` is `"recency"` and `score` is `null` on every hit. |
| `POST /artifacts/recall` with `vector` + `space_id` | **400.** This node ranks in no space, so the vector names one that does not exist here — the same refusal a foreign `space_id` gets on a seeded node, for the same reason. The message names both ways out: seed the set, or send the recall without `vector`, which is the row above and works. It refuses rather than ignoring the vector, because ignoring it answers with a body identical to that row's — leaving the caller unable to tell their vector had no effect. |
| A full reindex | Runs, and reports `"vector_arm": "off (no AnchorSet)"`. |

**Check which state a node is in.** `KEYS_DIR` must already hold a keyset (the command derives
the platform system principal from it):

```bash
python -m mantle.system.manage_anchors --action inspect
```

It names the live set's anchor count, model, dimension and **fingerprint** — a hash over the
anchor ids, so two operators can establish that their nodes route into the same regions without
either node exporting an anchor, a label or a vector. `/status` carries the same value.

**Seed it.** An anchor **is an artifact** (`application/vnd.agience.anchor+json`) and the
AnchorSet **is the collection** of them (slug `agience-anchorset`, created on first use):

```bash
python -m mantle.system.manage_anchors --action load --path anchors.json
```

`anchors.json` is the single-file form `AnchorSet.save`/`load` emits and `ember ingest --anchors`
consumes; there is no second serialisation and no other way in. `--dry-run` verifies the file and
reports its fingerprint without writing. Reindex afterwards so already-stored artifacts reach the
vector cells.

**Use that command rather than `POST /artifacts`.** An anchor's id is
`uuid5(sha256(label ‖ model_id ‖ embedding))` and that id **is** the cluster id — it names the
cell storage path, the HKDF key `info`, the AEAD associated data and the mesh region.
`CreateArtifactRequest` has no `id` field and the write path assigns a fresh `uuid4`, so posting
anchors one at a time replaces exactly the value that makes two nodes' cells comparable, and
nothing downstream can tell: queries route to regions the writer never produced, cells miss, the
semantic arm returns nothing, the request still answers `200` on lexical results, and mesh sync
transfers nothing and reports success. The load command preserves every id and **verifies** each
one against its own content, refusing a file whole if any anchor disagrees.

**What goes wrong, and what each says to do:**

| Symptom | The error names |
|---|---|
| Recall comes back `ordering: "recency"` with null scores | No set is seeded. `--action inspect` says so and gives the load command. |
| `400` on a query with `vector` | Either the width is not the set's `dim`, or the `space_id` is not the set's `model_id`. The message names the expected value and both spaces. |
| `REFUSED:` from `--action load` | An anchor's stated id disagrees with its own content. Re-export the file with `AnchorSet.save`; the ids cannot be repaired by hand, because the id **is** the content hash. |
| The arm refuses after it worked | The seeded set is a different space from the one this store's cells were written under. Restore that set, or drop the cells and reindex. |

## Architecture

Mantle is a **verify-only encrypted database** on the data plane: it serves and verifies data,
never fabricates provenance, and never embeds on its own behalf — vectors arrive from a caller,
they are not produced here. It emits a change-feed event for every artifact write.

*Verify-only is a claim about the data plane, not about the network.* Mantle signs exactly one
thing (`services/peer_signing.py`): a short-lived, audience-scoped **service JWT** saying "Mantle
is calling", used on its one outbound call to Origin. It never signs a **user** token, and it
mints no delegations — RFC 8693 delegation is inbound-only here: `services/dependencies.py`
accepts one, resolves it to the subject with the acting server recorded in `actor`, and the peer
that issues it is the authority issuer.

**The running service signs no user token, and `mantle-token` is not an exception.** That command is
an offline CLI over `KEYS_DIR` — it holds the same private key the service holds, so it mints
exactly what anyone with read access to that directory could mint by hand, and the service merely
*verifies* the result through the same generic path it verifies every other issuer with. No route
issues a token, and `sign_service_jwt` hard-codes `principal_type: service`.

The canonical design docs live in the **agience-pharos** repo under
`dev-legacy/dev-features/` — notably the per-state search index, the trust floor + event-driven
architecture, and trusted issuers as artifacts (`vnd.agience.issuer+json`). Mantle
verifies tokens from any configured OIDC issuer via one generic verifier, with the
authority manifest as a bootstrap seed.

### The surfaces

Everything is an artifact, so most of what follows is a `content_type` rather than a new plane.

- **HTTP.** Five routers: `/artifacts` (CRUD, children, commits, content, `POST /artifacts/recall`
  for ranked and candidate-set retrieval), `/grants` (grants, invites, grant keys and key
  bundles), `/events`, `/system` (the whole admin namespace behind one predicate —
  issuers, users, seed, admin grants, and `POST /system/erasure/{person_id}`), and `/mcp`.
  `/mcp` is Model Context Protocol over Streamable HTTP, and the surface is ROUND: `create_artifact`
  stores and `recall` finds, beside `update_artifact`, `delete_artifact`, `list_artifacts`,
  `get_artifact` and `get_children` — seven tools. Every tool
  dispatches into the REST handler that owns the verb with the caller's own principal, so the write
  is the one `POST /artifacts` makes and the search is the one `POST /artifacts/recall` makes —
  including the field filters, the coverage ordering, and every 400 and 401 verbatim.
  `create_artifact` takes an optional `identity`, which makes the write idempotent by deriving
  the id from a caller-chosen name for the thing being stored — see
  [above](#connect-a-client).
  `/docs` and `/openapi.json` are always served: the schema is the API's contract, not a secret,
  and every route behind it enforces its own authorization. There is no `/secrets`: a secret is a
  `vnd.agience.credential+json` artifact whose value is its content, so the envelope encrypts it
  at rest and the light cone decides who may read it — one cipher and one authorization path,
  the same two every other artifact gets.
- **The change feed.** Every artifact write emits an event at the persistence chokepoint
  (`db/doc_boundary.py`), so coverage is complete by construction. `event_bus.py` fans out
  in-process and appends to a durable log a consumer resumes from by cursor; a subscription is an
  artifact (`vnd.agience.subscription+json`) carrying a filter and that cursor, created and shared
  through `/artifacts` like anything else. `event_backplane.py` adds optional Redis or MQTT
  distribution across processes — unconfigured is a supported configuration, not a degraded one,
  and the app refuses to boot multi-worker without one rather than dropping events quietly. Live
  delivery re-runs the ACL filter in every receiving process, so a back-plane carries signal and
  never authority.
- **The context lattice.** A context is an artifact too (`entities/context.py`), composed over
  context edges by one bounded, attenuating walk (`services/context_service.py`). The walk takes a
  **required** authority ceiling — the id universe it may not leave — so a context edge only ever
  narrows and can never manufacture reach; `UNCONFINED` is a named opt-out for callers wanting the
  lattice's shape rather than an authorization answer, and there is no default, so no call site
  gets the unsafe one by not thinking about it. It is additive: cell-key derivation is
  `HKDF(master, collection_id ‖ cluster_id)`, unchanged, and no ciphertext moves. Today
  `services/dependencies.check_access` — the gate in front of every read — walks containment only,
  so the resolver holds the context walk to the grant-derived set and it contributes nothing.
  Deliberately: two answers to "what may this principal reach" that disagree is a key issued for
  an artifact the gate would refuse, and the narrower answer is the only safe one.
- **Vector ingress.** An artifact create or update may carry an optional `vector` plus the
  `space_id` it belongs to; `space_id` is required whenever `vector` is present, because two
  vectors are comparable only within a named space. `api/vectors.py` validates shape only —
  finite numbers, a bounded positive dimension, a non-zero norm — never quality. Mantle stores
  what a writer produced and never embeds. `POST /artifacts/recall` takes the same pair as the
  reader's half of that seam, so a caller holding a query vector supplies it the way a writer
  supplies the vector of what it stores. Both halves reach a semantic arm that stays inert
  until an AnchorSet is seeded — see
  [above](#-semantic-recall-is-inert-until-you-seed-an-anchorset).
- **Query syntax.** `query_text` carries terms and filters together. Terms: `+term` (required),
  `!term` (excluded), `~term` (selects what gets embedded), `="phrase"` (exact). Filters narrow
  the result set:

  | | |
  |---|---|
  | Filterable | `id`, `root_id`, `collection_id`, `content_type` (alias `type`), `owner_id`, `title`, `description`, `tags` (alias `tag`), `created_at`, `updated_at` |
  | Operators | `field:value` (case-insensitive; `a,b` is any-of) · `field:="Exact Value"` (case-sensitive, whole) · `!field:value` · `field:>value` / `field:<value` on `created_at` / `updated_at` only |
  | Combining | filters conjoin; each additional one narrows further |
  | Refused, with a 400 naming it and why | `state:`, `content:`, `size:`, `filename:` — fields a caller may reasonably expect that this store cannot answer · `field:~value` · a range on an unordered field · a query of nothing but filters |
  | Not a filter at all | any other word — it searches as an ordinary term |

  **`word:value` is a filter only when `word` is a field on one of those two rows**, so `https://example.com`,
  `meeting at 3:30`, `C:\Users\example` and `ratio 16:9` are ordinary searches: a colon in a token
  the field list does not name is just a character in that token, and it reaches retrieval
  unquoted and unchanged. The parser holds no field list of its own — it asks
  `search/field_filters.is_filter_field`, the same roster the resolver resolves against, so the
  two cannot disagree about what a field is. The cost is that a **misspelled field is a search
  term, not an error**: `titel:foo` searches for the literal text `titel:foo` and finds nothing,
  rather than telling you `titel` is not a field. Check the Filterable row when a `field:value`
  query returns nothing.

  A filter resolves to a set of artifact ids and is intersected with the light cone **before**
  retrieval, so both arms honour it identically and `total` and pagination count filtered
  matches. It can only ever narrow: the predicate is shown docs of authorized artifacts only,
  so no filter can reveal — or hint at — an artifact the caller could not already read, and a
  filter naming an unreadable artifact is indistinguishable from one matching nothing.
  Filterable is everything a doc plainly carries; `content` is not, because it is encrypted at
  rest and the postings are blind tokens. `state` is not either — it selects the index segment,
  which is a separately keyed tree chosen before the query runs, so it stays the `state` request
  field. Both are refused with a 400 naming them, because both are fields a caller can
  reasonably expect — being unfilterable here is a fact about the store, not a spelling
  mistake. Filter tokens never reach the index: retrieval sees the terms only, and
  `applied_filters` on the response lists what actually narrowed the result. Quoting forces a
  term either way, which is how you search for a *field's* name literally (`"type:pdf"`).

### Security invariants

Four properties hold across the codebase, are asserted by tests, and must survive every change:

1. **Geometry never authorizes.** Embeddings and routing rank *within* an already-authorized candidate set; they never widen one. The routing path receives no key material and runs strictly before any key request.
2. **Authorization is decided only by the light cone and grants, and the light cone is bounded above by them.** An unauthorized context is never routed to, padded in, or key-issued. Access is default-deny; there is no owner fast-path — even the creator holds an explicit, revocable grant. Two tighter statements hold inside this one, and both are structural rather than asserted after the fact:
   - *The resolve cannot exceed the read gate.* The context walk is confined to the grant-derived id set, so `resolve(principal, action) ⊆ grants-alone(principal, action)` is a property of how the call is made. A resolver that reached further would hand out a content key for an artifact `check_access` then 404s.
   - *A grant on one artifact means one artifact.* Recall cuts twice: a posting entry must clear the collection cut (which index may be read at all) **and** the artifact-granular cut from the same resolve. Without the second, sharing one document would expose every document filed beside it. An empty authorized set and an absent one are distinct — empty means "the light cone authorized nothing" and returns nothing, never the whole scope.
3. **Ciphertext is bound to its identity.** Every cell is AEAD-encrypted with associated data bound to its context and cluster, so a blob presented under the wrong key or moved to the wrong slot fails authentication before deserialization.
4. **Revocation requires no re-encryption.** Removing a grant alone prevents routing to, deriving keys for, and decrypting the affected cells.

**Composition along a path is monotone and non-amplifying**, and there is exactly one operator
that makes it so. `src/mantle/attenuation.py` holds the meet: a bounded meet-semilattice over the
CRUDEASIO action set, with an absorbing deny and a full-authority identity. Both storage encodings
— the `edge.propagate` TEXT column and `Grant`'s nine `can_*` booleans — are codecs onto the same
`Mask` type and round-trip through it, so the light-cone walk and a grant's own mask cannot
disagree about the zero element. `tests/test_attenuation_algebra.py` proves the laws exhaustively
and `tests/test_attenuation_is_single_sourced.py` sweeps `src/mantle` by AST for a second
implementation. Invariant #1 is the same principle but not this operator: it stays an ordering and
import-boundary discipline, enforced where it already is.

If a contribution weakens any of these, it will not be merged — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Backing a node up

**This section is a runbook rather than a feature.** Mantle ships no backup, snapshot, restore or
corruption-detection code — nothing here runs on a schedule, nothing verifies a copy, and nothing
notices if you never make one. What follows is the procedure an operator runs with the tools their
platform already has. It is written down because the store is a file and a directory, so the
procedure is short and getting it slightly wrong is silent.

A node is **four** things, and a backup missing any one of them does not restore:

| Part | Where | Notes |
|---|---|---|
| The lattice | `MANTLE_LATTICE_PATH` | One SQLite file, in WAL mode — so it is really three files (`.db`, `-wal`, `-shm`). |
| Key material | `KEYS_DIR` | **Without this the rest is unreadable ciphertext.** See below. |
| Content | The local CAS under `AGIENCE_BASE_DIR/.data`, and/or the content bucket | Whichever tiers this node uses — see `CONTENT_*` in `.env.example`. |
| The indexes | `MANTLE_SSE_DIR`, `MANTLE_CELL_DIR` | Derived from the lattice, and a full rebuild is measured in days-to-weeks under S3 write contention (`main.py`), so treat them as data rather than as cache. |

**Copy the lattice with `VACUUM INTO`, not with `cp`.** A plain file copy of a WAL-mode database
while the service is running captures the `.db` without the committed pages still in the `-wal`,
and the result opens without complaint and is missing recent writes. `VACUUM INTO` runs inside a
read transaction and writes one consistent, already-compacted file, with no downtime and without
blocking writers:

```bash
sqlite3 "$MANTLE_LATTICE_PATH" "VACUUM INTO '/backups/mantle-lattice.db'"
```

Copy `KEYS_DIR` and the content tiers with an ordinary file copy, and take the key material
**separately and under different custody**. Grants are keys, not metadata: a copy of the lattice
without `encryption.key` decrypts to nothing, and a copy of both in one place is a single object
that surrenders the whole store.

**Restoring** is placing those parts back where the environment points and starting the service —
there is no import step and no restore command. Restore into a node whose `KEYS_DIR` holds the
*same* keyset the backup was taken under; a different one leaves every secret and platform setting
permanently unreadable, which `.env.example` warns about at length under `KEYS_DIR`.

**Nothing above is verified by this repository.** A backup you have not restored is a hypothesis;
restore into a scratch node and read an artifact back. Scheduling, retention, verification and
off-site replication belong in your platform.

## Contributing

Bug reports, tests, and hardening contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) first — Mantle has a security-invariant test discipline that contributions must follow.

Security issues → **connect@agience.ai** (do not open a public issue).

## License

**Apache License 2.0.** See [LICENSE](LICENSE) and [NOTICE](NOTICE). Contributing: [CONTRIBUTING.md](CONTRIBUTING.md).

**Trademarks.** "Agience" and the Agience logo are trademarks of Ikailo Inc. Apache-2.0 §6 licenses
copyright and patent, **not** the marks — take the code, build on it, ship it; call your product
your own name. Worth stating on a permissive repository precisely because permissive is otherwise
read as "everything is granted".

## Star History

<a href="https://www.star-history.com/?repos=Agience%2Fagience-mantle&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=Agience/agience-mantle&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=Agience/agience-mantle&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=Agience/agience-mantle&type=date&legend=top-left" />
 </picture>
</a>
