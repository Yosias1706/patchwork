FROM node:20-alpine AS web-build

WORKDIR /build/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
# An empty URL makes the dashboard call the API at its own origin.
ARG VITE_API_URL=
ENV VITE_API_URL=${VITE_API_URL}
RUN npm run build


FROM python:3.13-slim AS application

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATCHWORK_WEB_DIST=/app/web-dist

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .
COPY --from=web-build /build/web/dist /app/web-dist

EXPOSE 8000
# Hosted platforms commonly provide an HTTP port at runtime (for example,
# Render uses PORT=10000). `exec` keeps Uvicorn as PID 1 for graceful shutdown.
CMD ["/bin/sh", "-c", "exec uvicorn rag_document_search.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
