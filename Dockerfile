# Blindspot Market: one container serving the API and the built web app.
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev
# Download Banking77 and embed it once at build time (a few minutes on CPU), so the server starts quickly.
RUN uv run --no-dev python -c "from pathlib import Path; from blindspot.learner import build_cache; build_cache(Path('data'))"
COPY experiment/results.json ./experiment/results.json
COPY scripts ./scripts
COPY --from=web /web/dist ./web/dist
ENV HOST=0.0.0.0 PORT=8077 BLINDSPOT_DB=/data/blindspot.db
VOLUME /data
EXPOSE 8077
CMD ["uv", "run", "--no-dev", "python", "-c", "from blindspot.api import main; main()"]
