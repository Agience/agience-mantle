# agience-mantle — the encrypted memory service, FastAPI on :8081.
# The shared foundation is supplied as a named build context:
#   docker build --build-context core=../agience-core -t agience-mantle .
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/mantle \
    AGIENCE_BASE_DIR=/app
WORKDIR /app
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc && \
    rm -rf /var/lib/apt/lists/*
# App dependencies (cacheable)
COPY src/mantle/requirements.txt ./mantle/requirements.txt
RUN pip install --no-cache-dir -r mantle/requirements.txt
# Shared foundation (agience-core → provides the `platform` import; self-contained).
COPY --from=core . /tmp/agience-core
RUN pip install --no-cache-dir /tmp/agience-core
# App code
COPY src/mantle/ ./mantle/
WORKDIR /app/mantle
EXPOSE 8081
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8081", "--timeout-graceful-shutdown", "10"]
