# agience-mantle — the encrypted memory service, FastAPI on :8081.
# The shared foundation is supplied as a named build context:
#   docker build --build-context core=../agience-core -t agience-mantle .

# ---- builder: compile/install deps with the toolchain, kept OUT of runtime ----
FROM python:3.12-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /build
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc && \
    rm -rf /var/lib/apt/lists/*
COPY src/mantle/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
# Shared foundation (agience-core → provides the `core` import; self-contained).
COPY --from=core . /tmp/agience-core
RUN pip install --no-cache-dir --prefix=/install /tmp/agience-core

# ---- runtime: slim image, no compiler/toolchain → smaller attack surface ----
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    AGIENCE_BASE_DIR=/app
WORKDIR /app
# Installed Python packages only (build-essential/gcc stay in the builder stage).
COPY --from=builder /install /usr/local
# App code
COPY src/mantle/ ./mantle/
EXPOSE 8081
# ⭐ RUN AS A PACKAGE: `mantle.main:app` on `PYTHONPATH=/app`, NOT `main:app` on `/app/mantle`.
#
# The service used to run with `src/mantle` ITSELF as the root, so `main`, `db`, `services` were
# top-level modules there while a library consumer imported them as `mantle.db`, `mantle.services`.
# Two entry points, two import styles for one codebase — and mixing them in one process yields
# DIFFERENT CLASS OBJECTS for the same class, so `except SomeError` silently stops matching across
# the seam. (Measured: the same exception imported both ways compared unequal.)
#
# One shape now. `pip install agience-mantle` and the running service are the same artifact, which
# is what makes the wheel worth shipping — an embedding consumer (EREA) and this container import
# identically. Changed 2026-07-29 [John: "whatever is easiest for a user. least friction"].
CMD ["uvicorn", "mantle.main:app", "--host", "0.0.0.0", "--port", "8081", "--timeout-graceful-shutdown", "10"]
