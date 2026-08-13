# agience-mantle — the encrypted memory service, FastAPI on :8081.

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
# The trust floor (agience-prism/py → the `prism` package: trust.key_manager, trust.service_identity,
# mass, canonical, envelope, grounding, law, vector).
COPY --from=prism . /tmp/agience-prism
RUN pip install --no-cache-dir --prefix=/install /tmp/agience-prism

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

ARG MANTLE_REVISION=unknown
ARG MANTLE_BUILD_TIME=unknown
RUN printf '{"version":"%s","build_time":"%s"}\n' "$MANTLE_REVISION" "$MANTLE_BUILD_TIME" \
      > /app/build_info.json

# ---- drop root ----
# ⭐ THE SERVICE DOES NOT NEED root AND MUST NOT HOLD IT. It binds 8081 (unprivileged), installs
# nothing at runtime, and writes only under AGIENCE_BASE_DIR — so root buys nothing here except
# the blast radius of a code-execution bug: with it, an escape writes anywhere in the container
# filesystem, including the site-packages this process imports from on the next start.
#
# EVERY RUNTIME WRITE IS UNDER /app, which is why one `chown` covers all of them: `.data/` holds
# the lattice (MANTLE_LATTICE_PATH), the encrypted lexical index (MANTLE_SSE_DIR), the vector
# cells (MANTLE_CELL_DIR) and the local content CAS, all derived from AGIENCE_BASE_DIR above;
# KEYS_DIR is written once, when `delete_setup_token()` removes `setup.token`. `.data` is created
# HERE rather than left to the first write, so a named volume mounted over it inherits this
# ownership at creation instead of being made by the daemon as root and refused by the process.
#
# A BIND MOUNT DOES NOT INHERIT IT — the host's own ownership wins — so a host directory mounted
# at /app/.data must be readable and writable by uid 10001. A fixed high uid rather than one the
# distro picks: the number is part of the image's contract with whatever mounts a volume into it,
# and a uid that moved between rebuilds would break that mount instead of the image.
RUN groupadd --system --gid 10001 mantle \
 && useradd --system --uid 10001 --gid 10001 --home-dir /app --shell /usr/sbin/nologin mantle \
 && mkdir -p /app/.data \
 && chown -R mantle:mantle /app
USER mantle

EXPOSE 8081

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8081/', timeout=4).status==200 else 1)"]

  # ⭐ RUN AS A PACKAGE: `mantle.main:app` on `PYTHONPATH=/app`
  #
  # NOT `mantle-serve`, even though `[project.scripts]` declares it. This image COPIES the source
  # tree rather than pip-installing the distribution, so no console script is generated — the
  # entry point exists for `pip install agience-mantle[service]`, which is a different delivery
  # path. The two stay equivalent: `scripts/serve.py` carries these same defaults (port 8081, a
  # 10-second graceful shutdown), so a change to one belongs in the other.
CMD ["uvicorn", "mantle.main:app", "--host", "0.0.0.0", "--port", "8081", "--timeout-graceful-shutdown", "10"]
