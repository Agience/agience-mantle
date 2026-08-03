# Mantle E2E — blackbox suite

A **true blackbox** end-to-end suite, importing none of the platform's Python packages: it drives
the Agience platform purely over
HTTP (Origin `:8080` for identity, Mantle `:8081` for the data plane). If it can
call it, a real client can.

It exercises the full matrix the platform is meant to guarantee:

| Dimension | How the suite creates it |
|---|---|
| **Multiple issuers** | Origin (native) **+** a self-signed RSA IdP registered via `POST /issuers` |
| **Multiple tenants** | the external issuer carries a `namespace` → `uuid5(tenant, sub)` isolates it from native users |
| **Multiple users** | operator (admin) + N registered users, per tenant |
| **Multiple collections** | workspaces/collections with child artifacts, per user |

## What it covers

- **Auth** — bootstrap → operator; register + login; external-issuer tokens; bad/expired/wrong-`aud` → 401; API keys (`agc_…`) usable on Mantle.
- **Users / provisioning** — first `GET /artifacts/visible` seeds baseline grants; same `sub` under two issuers → two distinct users.
- **Collections / artifacts** — every core API: create (top-level=committed / child=draft), read, update, delete, children, order, commits, batch, visible.
- **Sharing (grants on Origin, CRUDEASIO booleans)** — `can_read` grant + propagation; a read-only grantee's write → **404** (confinement masks 403 as not-found); invite → claim; revoke → 404; deny-effect → 404; **cross-tenant isolation**.
- **Search** — `POST /artifacts/search` (lexical) + `POST /search/query` (light-cone); `state` = committed vs draft.
- **Commit / first-observation** — draft → `PATCH {state: committed}` → visible in committed search; revert; under `MANTLE_LAZY_INDEX=on`, lazy create → invisible → first `GET` → materialized → visible → `/warm` bulk.
- **Secrets / Events / Issuers** — `/secrets` CRUD; `/events` websocket delivery + ACL; `/issuers` admin-only.

## Running it

The suite needs a **live stack**. From `agience-beam/`:

```bash
# fresh stack (bootstrap available). Search is lexical BM25/SSE.
docker compose -f docker-compose.local.yml up -d --build
```

Then, from this directory:

```bash
python -m venv .venv && . .venv/Scripts/activate   # or bin/activate on *nix
pip install -r requirements.txt
pytest
```

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
| `E2E_DATA_DIR` | `../../agience-beam/.data-local` | where to read `keys/bootstrap.token` |
| `E2E_BOOTSTRAP_TOKEN` | — | supply the token directly (remote stack) |
| `E2E_LAZY_INDEX` | off | enable `-m lazy` assertions |
| `E2E_HAS_EMBEDDINGS` | off | ⛔ inert — there is no semantic search to assert. Every embeddings provider was removed under the no-models rule (`mantle/embeddings.py`), so setting this enables assertions against `_UnconfiguredEmbeddings`, which returns empty vectors. A permanently-off flag guarding a permanently-impossible path is a silent skip; the flag and its assertions should go. |

## Result

A fresh-stack run is **green**: 38 passed / 3 skipped eager, + 2 passed lazy, 0 failed.

The 3 **skips** are intentional: deny-effect grants are created server-side only, and
the two lazy first-observation tests run only under `E2E_LAZY_INDEX=1`.

Passing throughout: confinement (404-not-403), search light-cone isolation,
cross-tenant isolation, commit lifecycle, event delivery + ACL, invites/claims.

## Standalone Mantle (Origin-off / sovereign node)

Mantle no longer *requires* Origin at runtime. Authorization (grants + API keys)
and secret-artifact material live entirely in Mantle's own store: Mantle serves
`/grants` and `/api-keys` directly and mints API keys via `POST /api-keys`, Origin
stays identity-only, and the platform operator is resolved locally:

1. Mantle's own `platform.operator_id` setting, else
2. **`AGIENCE_OPERATOR_ID`** env — the Mantle user id that holds platform admin
   (for an external-IdP operator, the namespaced `uuid5(tenant, sub)` id), else
3. Origin's `/internal/operator-id` (full-platform fallback, used only when Origin
   is present).

So a sovereign node runs **Mantle + its lattice + MinIO**, with identity
supplied by an external OIDC issuer (registered via `POST /issuers`) and the
operator named by `AGIENCE_OPERATOR_ID`. The single remaining Mantle→Origin call
(`get_operator_id`) is optional and non-fatal.

> The couplings are closed and unit-covered (`tests/test_operator_and_secret_material.py`).

## Notes / limits

- **Reset:** the operator identity is deterministic, so reruns against a live
  stack are idempotent; for a clean slate wipe `agience-beam/.data-local` and
  rebuild. Every user/collection is uniquely suffixed, so tests stay isolated.
- **Search backend:** the lattice + MinIO must be up or search endpoints 503 (no
  plaintext fallback). Search is lexical — there is no semantic arm to degrade FROM:
  every embeddings provider was removed under the no-models rule. The suite asserts
  lexical behaviour; the `skip`ped semantic checks guard a path that cannot exist.
- **Router coverage:** limited to the routers `main.py` registers — `beacon`,
  `stream` and `server_credentials` sit outside it.
