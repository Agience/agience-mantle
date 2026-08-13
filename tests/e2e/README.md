# Mantle E2E — blackbox suite

A **true blackbox** end-to-end suite, importing none of the platform's Python packages: it drives
the Agience platform purely over
HTTP (Origin `:8080` for identity, Mantle `:8081` for the data plane). If it can
call it, a real client can.

It exercises the full matrix the platform is meant to guarantee:

| Dimension | How the suite creates it |
|---|---|
| **Multiple issuers** | Origin (native) **+** a self-signed RSA IdP registered via `POST /system/issuers` |
| **Multiple tenants** | the external issuer carries a `namespace` → `uuid5(tenant, sub)` isolates it from native users |
| **Multiple users** | operator (admin) + N registered users, per tenant |
| **Multiple collections** | workspaces/collections with child artifacts, per user |

## What it covers

- **Auth** — bootstrap → operator; register + login; external-issuer tokens; bad/expired/wrong-`aud` → 401; grant keys and grant bundles (`agk_…`) usable on Mantle; retired `agc_` API keys rejected.
- **Users / provisioning** — first `GET /artifacts/visible` seeds baseline grants; same `sub` under two issuers → two distinct users.
- **Collections / artifacts** — every core API: create (top-level=committed / child=draft), read, update, delete, children, order, commits, batch, visible.
- **Sharing (grants in Mantle's own lattice, CRUDEASIO booleans)** — `can_read` grant + propagation; a read-only grantee's write → **404** (confinement masks 403 as not-found); invite → claim; revoke → 404; deny-effect → 404; **cross-tenant isolation**.
- **Search** — `POST /artifacts/recall`, ordered + hydrated and `candidates: true` (the same narrowed set, unordered); `state` = committed vs draft.
- **Commit / first-observation** — draft → `PATCH {state: committed}` → visible in committed search; revert; under `MANTLE_LAZY_INDEX=on`, lazy create → invisible → first `GET` → materialized → visible → `/warm` bulk.
- **Events / Issuers** — `WS /events` delivery + ACL; `/system/issuers` admin-only.
- **Platform admin** — the `/system` namespace behind one predicate: `GET /system/users` is 403 for a non-admin, `POST /system/users/{id}/grant-admin` promotes, `DELETE /system/users/{id}/revoke-admin` demotes, and the operator cannot revoke its own admin.

## Running it

The suite needs a **live stack**, and the compose stack that provides it lives in the sibling
**`agience-bundle`** repo — Origin, Mantle, MinIO and the init container that writes the keyset
and the single-use bootstrap token. From `agience-bundle/`:

```bash
# fresh stack (bootstrap available). Search is lexical BM25/SSE.
# The compose file reads its overrides from Origin's .env.
docker compose --env-file ../agience-origin/.env -f docker-compose.local.yml up -d --build
```

Then, from this directory:

```bash
python -m venv .venv && . .venv/Scripts/activate   # or bin/activate on *nix
pip install -r requirements.txt
pytest
```

This directory carries no pytest config of its own. The repo root's `pyproject.toml` is the
one config — it is where `lazy`, `search` and `external_issuer` are declared — so `pytest`
here and the root run resolve the same markers and the same rootdir, and a green run is
evidence about the code rather than about which directory you typed it in.

### First-observation (lazy) regime

The lazy-materialization assertions only mean something when the stack runs with
`MANTLE_LAZY_INDEX=on`. Bring the stack up with that env set and tell the suite:

```bash
MANTLE_LAZY_INDEX=on docker compose ... up -d
E2E_LAZY_INDEX=1 pytest -m lazy
```

## Configuration (env)

| Var | Default | Meaning |
|---|---|---|
| `E2E_ORIGIN_URL` | `http://localhost:8080` | Origin base URL |
| `E2E_MANTLE_URL` | `http://localhost:8081` | Mantle base URL |
| `E2E_AUTHORITY_ISSUER` | `http://origin:8080` | `aud` on Origin user tokens |
| `E2E_DATA_DIR` | `<workspace>/agience-bundle/.data-local` | where to read `keys/bootstrap.token` — the bundle stack's `LOCAL_DATA` root |
| `E2E_BOOTSTRAP_TOKEN` | — | supply the token directly (remote stack) |
| `E2E_LAZY_INDEX` | off | enable `-m lazy` assertions |
| `E2E_HTTP_TIMEOUT` | `15` | per-request timeout, in seconds |
| `E2E_HAS_EMBEDDINGS` | off | Read by `_config.py`, consulted by no test. There is no semantic ranking to assert: the provider factory has exactly one product, `_UnconfiguredEmbeddings`, which returns empty vectors, per the no-models rule (`mantle/search/embeddings.py`). Setting it changes nothing. |

## Result

The suite collects **48 tests**, of which three are skipped on a standard eager run, all three
intentionally: deny-effect grants are creatable server-side only, and the two first-observation
tests run only under `E2E_LAZY_INDEX=1`. Everything else is expected green against a fresh stack.

Any other skip is the suite reporting a missing precondition rather than a pass: the stack
unreachable, no bootstrap token, the second issuer unregistrable, or a search endpoint answering
503 because the lattice or MinIO is down.

Covered throughout: confinement (404-not-403), search light-cone isolation,
cross-tenant isolation, commit lifecycle, event delivery + ACL, invites/claims.

## Standalone Mantle (Origin-off / sovereign node)

Mantle does not require Origin at runtime. Authorization and secret-artifact
material live entirely in Mantle's own store: Mantle serves `/grants` directly and
mints bearer credentials via `POST /grants/keys` — a grant key IS a grant, so there
is one authorization model rather than a separate API-key system beside it. Origin
stays identity-only, and the platform operator is resolved locally:

1. Mantle's own `platform.operator_id` setting, else
2. **`AGIENCE_OPERATOR_ID`** env — the Mantle user id that holds platform admin
   (for an external-IdP operator, the namespaced `uuid5(tenant, sub)` id), else
3. Origin's `/internal/operator-id` (full-platform fallback, used only when Origin
   is present).

So a sovereign node runs **Mantle + its lattice + MinIO**, with identity
supplied by an external OIDC issuer (registered via `POST /system/issuers`) and the
operator named by `AGIENCE_OPERATOR_ID`. The single remaining Mantle→Origin call
(`get_operator_id`) is optional and non-fatal.

> The couplings are closed and unit-covered: `tests/test_no_origin_service.py` measures what
> Mantle does when no Origin answers — including that the operator falls back through the
> three legs above, that an absent Origin costs one short probe rather than a connect
> timeout per caller, and that an empty `ORIGIN_URI` makes no call at all.
> `tests/test_origin_is_not_a_dependency.py` holds the import boundary.

## Notes / limits

- **Reset:** the operator identity is deterministic, so reruns against a live
  stack are idempotent; for a clean slate wipe `agience-bundle/.data-local` and
  rebuild. Every user/collection is uniquely suffixed, so tests stay isolated.
- **Search backend:** the lattice + MinIO must be up or `POST /artifacts/recall` answers
  503, which the suite treats as a skip. There is no plaintext fallback index to degrade
  to. Recall narrows lexically and NO COSINE ranks what survives, for two independent reasons:
  the only provider Mantle builds returns empty vectors, per the no-models rule, and ranking
  needs a provisioned AnchorSet, which a stack brought up from `docker compose` does not have.
  Supplying a query `vector` does not change that — it removes the first reason and leaves the
  second. What orders the result instead is the narrowing's own answer: `ordering: "coverage"`
  and an integer count of matched query stems as each hit's `score`. Every assertion in
  `test_05_search.py` is about WHICH artifacts come back rather than in what order, because
  the suite's single-term queries put every hit on the same count.
- **Vectors:** an artifact write may carry an optional `vector` plus its `space_id`, which
  Mantle shape-validates and stores. It never embeds, and the suite asserts no ranking
  over them.
- **Router coverage:** limited to the surface `main.py` mounts — `artifacts`, `grants`,
  `events`, `system` and `mcp`. A server has no router: it is an ordinary
  `vnd.agience.server+json` artifact written through `/artifacts`, stored like any other.
  Neither does a secret: a `vnd.agience.credential+json` artifact, whose value is content
  the envelope encrypts and whose reader is decided by the light cone.
