# Agience Mantle — the lattice (data + backup)

Status: **Reference**
Date: 2026-03-31 (header aligned to the canonical component definitions 2026-07-23)

This directory contains the FastAPI service for Agience Mantle — the lattice: authentication, artifact CRUD, workspace and collection lifecycle, search, storage, and MCP infrastructure.

For the broader project map, use:
- `.dev/README.md` for internal developer documentation
- `docs/README.md` for public-facing docs
- `CLAUDE.md` and `backend/CLAUDE.md` for coding-agent guidance

## Layer Boundary

The backend is the Core platform layer. Keep it type-blind.

- Routers, services, and DB adapters should not hardcode content-type-specific behavior
- New domain behavior lives on persona servers, in `agience-chorus/src/<persona>/`
- Type-specific logic belongs on the persona server that owns the type, never in Core

Authoritative boundary rules live in `agience-pharos/dev-legacy/dev-features/layered-architecture.md`.

## Directory Guide

```text
backend/
├── api/           # Pydantic request/response schemas by domain
├── db/            # The standalone lattice store: backend.py (the one import point) → lattice_api over db/lattice (SQLite + FS CAS), plus the S3 content adapter
├── entities/      # Dual-context entity models and serialization helpers
├── routers/       # FastAPI routers; keep them thin and type-agnostic
├── schemas/       # Package stub; the lattice schema is created on open
├── search/        # Query parsing, weights, and lexical (SSE/BM25) search support
├── services/      # Core orchestration and platform services
├── tests/         # Pytest suites
└── main.py        # App entry point and lifespan wiring
```

## Request Flow

```text
HTTP request
  -> router
  -> api module
  -> service
  -> db/search/storage adapter
```

Routers should not call DB adapters directly.

## Key Surfaces

- `main.py` initializes FastAPI, CORS, startup schema loading, and router registration
- `services/chorus_client.py` is the only outbound-MCP path — JSON-RPC over HTTP to Chorus's universal gateway (`/{server_uuid}/mcp`)
- `services/workspace_service.py` orchestrates workspace lifecycle and commit flow
- `services/collection_service.py` owns committed collection lifecycle and indexing hooks
- Type-declared operations are resolved and dispatched by kind (`mcp_tool` / `native` / `artifact_crud`) in the **crystal gateway**, not here. Mantle's op-dispatch route was removed deliberately; `db/lattice/test_op_dispatch_route_is_gone.py` is the guard that keeps it out.

## Local Development

```bash
cd src/mantle
KEYS_DIR=<keys-dir> MANTLE_LATTICE_PATH=./mantle.db uvicorn main:app --port 8081
```

Recommended validation:

```bash
cd src/mantle
KEYS_DIR=<tmp-dir> python -m pytest tests db/lattice -q
```

## Related Docs

- `.dev/features/layered-architecture.md`
- `.dev/features/artifact-model-and-referencing.md`
- `docs/mcp/overview.md`
- `.dev/testing/test-suite-summary.md`