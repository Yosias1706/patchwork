# Patchwork

Patchwork is a repository-aware RAG application for software engineers. Connect a GitHub repository, index its merged bug-fix pull requests and diffs, then describe a new incident to find grounded examples of how that codebase solved similar problems before.

It is deliberately evidence-first: an investigation returns the matching pull request, commit, changed files, regression tests, diff excerpts, and source links. It does **not** autonomously change a repository.

## What it demonstrates

- Repository-scoped retrieval: results from one connected repository cannot leak into another.
- GitHub ingestion: a balanced mix of `bug`-labelled PRs and unlabelled repair-like merged PRs (for example, regressions, races, timeouts, cleanup, and validation fixes) becomes searchable patch cards.
- GitHub account linking: users can optionally authorize their own GitHub account for private-repository reads; access tokens are encrypted at rest and never reach the browser.
- Grounded RAG: answers cite the indexed pull requests that support them.
- An evidence-first investigation agent: it extracts error signatures, paths, and symbols; runs a fixed set of repository-scoped searches; fuses results; and produces a cited implementation plan with files and regression coverage to review.
- Operational state: PostgreSQL persists repository connections, documents, chunks, sync runs, and evaluation runs.
- Retrieval evaluation: durable GitHub issue → fixing-PR manifests are measured with Recall@1, Recall@3, MRR, and retrieval-miss samples without indexing the evaluation query into its target PR.
- A React + Tailwind dashboard and a CLI, backed by FastAPI.

## Architecture

```text
GitHub repository URL
        │
        ▼
GitHub REST API ──► merged bug-fix PRs + changed files + tests
        │
        ▼
Patch-card builder ──► chunking + embeddings ──► PostgreSQL
                                                   │
New bug report ───────────────────────────────────┤
                                                   ▼
                      evidence agent + repository-scoped retrieval + grounded answer
                                                   │
                                                   ▼
                         matching PRs, commits, diffs, tests, and citations
```

The default retrieval stack is completely local and deterministic: hashing embeddings, cosine similarity, and extractive answers. Set the optional compatible-provider variables if you want hosted embeddings and generative answers.

## Quick start

Requirements: Python 3.11+, Node.js 20+, npm, Docker Desktop, and a GitHub.com network connection.

```bash
docker compose up -d database
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
uvicorn rag_document_search.api:app --reload
```

In another terminal, start the dashboard:

```bash
cd web
npm install
npm run dev
```

Open the URL Vite prints (normally `http://127.0.0.1:5173`). Paste a public URL such as `https://github.com/fastapi/fastapi`, wait for its sync to finish, and describe an issue to investigate. Private repositories need the optional GitHub account-linking setup below.

The API is documented at `http://127.0.0.1:8000/docs`.

## Accounts and saved work

Patchwork uses local email/password accounts. Register from the dashboard, then sign in to access an account-scoped workspace. Passwords are stored as salted `scrypt` hashes; browser sessions use opaque bearer tokens whose hashes are stored server-side and can be revoked on sign-out.

Repositories, sync runs, evaluation runs, and investigations belong to the signed-in account. Every investigation is saved automatically and can be reopened from **Saved investigations** on the Investigate view. The first account created after upgrading from an earlier single-user local install claims any existing unowned local repositories; new repositories are always created under the active account.

## GitHub account linking and private repositories

Public repositories can be connected without GitHub authentication. To let an account connect repositories it can privately read, configure a GitHub OAuth App and add these settings to `.env`:

```dotenv
PATCHWORK_GITHUB_OAUTH_CLIENT_ID=...
PATCHWORK_GITHUB_OAUTH_CLIENT_SECRET=...
PATCHWORK_GITHUB_OAUTH_REDIRECT_URL=https://patchwork.example.com/auth/github/callback
PATCHWORK_TOKEN_ENCRYPTION_KEY=...
PATCHWORK_WEB_URL=https://patchwork.example.com
```

Register the same callback URL in the GitHub OAuth App settings. Generate the encryption key once with:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

After deployment, a signed-in user selects **Link GitHub** from the dashboard account menu. The browser is redirected to GitHub, but the authorization code exchange and encrypted token storage happen only on the Patchwork API. Patchwork uses that account’s credentials only for read-only repository, pull-request, issue, and diff retrieval. Signing out does not remove the connection; **Unlink GitHub** removes its stored credentials.

GitHub OAuth Apps require the `repo` scope to read private repositories. The Patchwork application never calls GitHub write endpoints, but teams that require installation-scoped permissions should use a GitHub App with read-only repository permissions as the next hardening step.

## Dashboard

The React dashboard has three working views:

- **Investigate**: choose a repository, describe a regression, then review a concise finding and linked historical PR cards. Each card lists affected files and regression tests without exposing raw diff fragments or retrieval internals.
- **Repositories**: add GitHub URLs, view every connected repository, select one for investigation, refresh its patch history, open its evaluation, or remove its local indexed data.
- **Evaluation lab**: select a repository and measure Recall@1, Recall@3, and MRR against a durable manifest of GitHub issue-to-fix pairs captured during sync. It identifies verified data versus a non-reportable PR-description smoke test.

When a repository does not expose linked GitHub issues, the evaluation lab runs clearly labeled PR-description retrieval checks instead of returning an empty evaluation. These confirm indexing and repository scoping but are not presented as a real incident-retrieval benchmark.

## Using Patchwork

1. Connect a GitHub.com repository in the dashboard. Patchwork verifies the repository and queues a sync.
2. The sync scans recent closed PRs carrying the `bug` label, retains merged fixes, and indexes their descriptions, metadata, changed-file list, test-file list, and patch text.
3. Paste a bug report, stack trace, or incident description into **Investigate a new issue**.
4. Review the generated summary together with the historical PR cards. Follow the PR URL, inspect the diff, and reuse the test pattern as appropriate.
5. Run **Evaluation lab** to measure retrieval over issue/PR pairs. GitHub-linked pairs are persisted with their source and provenance at sync time. Patchwork keeps issue text out of the indexed target patch card, which prevents the target document from winning merely because it contains the test question.

For GitHub's unauthenticated rate limit, the dashboard defaults to 15 candidate PRs per sync. Use a fine-grained GitHub token for larger public syncs or private repositories.

## CLI

The `patchwork` CLI performs the same workflow without the web interface:

```bash
patchwork --session-token TOKEN connect https://github.com/fastapi/fastapi --pull-limit 10
patchwork --session-token TOKEN repositories
patchwork --session-token TOKEN investigate REPOSITORY_ID "A streaming endpoint loses type information after include_router"
patchwork --session-token TOKEN evaluate REPOSITORY_ID
patchwork --session-token TOKEN sync REPOSITORY_ID --pull-limit 20
patchwork --session-token TOKEN remove REPOSITORY_ID
```

`rag-search` remains available as a small general-purpose local document-search CLI:

```bash
rag-search ingest examples/documents
rag-search ask "How long are documents retained?"
```

## API overview

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/auth/github/authorize` | Start a signed-in user’s GitHub OAuth account-linking flow. |
| `GET` | `/auth/github/status` | Return safe GitHub connection status; never returns credentials. |
| `POST` | `/patchwork/repositories` | Validate, store, and queue a durable GitHub repository sync. |
| `GET` | `/patchwork/repositories` | List repositories and latest sync/evaluation state. |
| `POST` | `/patchwork/repositories/{id}/sync` | Start a fresh sync. |
| `POST` | `/patchwork/repositories/{id}/investigate` | Retrieve analogous historical fixes for an incident. |
| `POST` | `/patchwork/repositories/{id}/evaluate` | Measure repository retrieval with linked issue/PR pairs. |
| `DELETE` | `/patchwork/repositories/{id}` | Remove the repository and its indexed patch cards. |

## Deployment

`compose.production.yaml` packages the API, compiled React dashboard, PostgreSQL volume, health checks, restart policy, and a single durable-sync worker process. The worker claims queued rows with PostgreSQL row locking; a run whose lease expires after an API failure becomes eligible for another worker.

1. Create a production `.env` from `.env.example` and set a long, unique `PATCHWORK_POSTGRES_PASSWORD` plus `PATCHWORK_PUBLIC_URL` (for example `https://patchwork.example.com`).
2. Configure `RAG_CORS_ORIGINS` only if the frontend is hosted separately; the included container serves the dashboard and API from one origin.
3. Configure the GitHub OAuth values above before enabling private repositories.
4. Start the durable stack:

```bash
docker compose -f compose.production.yaml up --build -d
docker compose -f compose.production.yaml ps
```

The dashboard is served at `PATCHWORK_PUBLIC_URL` and `/health` is the service health endpoint. Keep the PostgreSQL volume and `PATCHWORK_TOKEN_ENCRYPTION_KEY` stable across restarts; changing the key makes existing encrypted GitHub connections intentionally unreadable until users link again.

### Free hosted portfolio deployment

The repository also includes [`render.yaml`](render.yaml) for a Docker-based Render web
service. It deliberately uses a separately managed PostgreSQL URL rather than Render's
free Postgres offering: a free Render Postgres database expires after 30 days. A Neon
Free database is a better no-cost portfolio default because it persists data while
automatically suspending unused compute.

1. Push this project to a GitHub repository.
2. Create a Neon PostgreSQL database and retain its pooled or direct PostgreSQL URL.
3. In Render, create a **Blueprint** from the GitHub repository. Render reads
   `render.yaml`, builds the Docker image, creates a free web service, and asks for the
   `RAG_DATABASE_URL` secret.
4. Paste the Neon connection string only into that Render secret field. Do not commit it
   or add it to the frontend.
5. Render assigns an `https://…onrender.com` URL. Patchwork automatically uses it for
   OAuth return navigation. After creating the GitHub OAuth App, set its callback to
   `https://YOUR-RENDER-HOST/auth/github/callback`, then add the client ID and client
   secret to the Render service environment variables and redeploy.

The free web service is appropriate for a portfolio preview, but may sleep when idle.
For an always-on commercial deployment, use paid compute and managed database backups.

Example investigation request:

```bash
curl -X POST http://127.0.0.1:8000/patchwork/repositories/REPOSITORY_ID/investigate \
  -H 'Content-Type: application/json' \
  -d '{"issue":"Streaming requests hang after a client disconnects. Find a similar fix and test pattern.","limit":5}'
```

## Configuration

Copy `.env.example` to `.env`. Environment variables take precedence.

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAG_DATABASE_URL` | `postgresql://rag:rag@localhost:5432/rag_document_search` | PostgreSQL connection string. |
| `PATCHWORK_GITHUB_TOKEN` | unset | Fine-grained token for higher GitHub limits or private-repository read access. Keep it out of source control. |
| `PATCHWORK_GITHUB_OAUTH_CLIENT_ID` | unset | GitHub OAuth App client ID; enables the optional per-account authorization flow. |
| `PATCHWORK_GITHUB_OAUTH_CLIENT_SECRET` | unset | GitHub OAuth App client secret. Store only in deployment secrets. |
| `PATCHWORK_GITHUB_OAUTH_REDIRECT_URL` | unset | Exact GitHub OAuth callback URL: `/auth/github/callback`. |
| `PATCHWORK_TOKEN_ENCRYPTION_KEY` | unset | Stable Fernet key for encrypting stored GitHub access/refresh tokens. |
| `PATCHWORK_WEB_URL` | local Vite URL | Dashboard URL used after a GitHub OAuth callback. |
| `PATCHWORK_DEFAULT_PULL_LIMIT` | `300` | Number of selected repair-history PRs indexed per sync, 1–300 through the API. Patchwork paginates closed-PR summaries across up to 10,000 PRs, then retrieves detailed diffs only for repair candidates. |
| `PATCHWORK_SYNC_LEASE_SECONDS` | `900` | Time after which an abandoned sync can be safely reclaimed. |
| `RAG_CORS_ORIGINS` | local Vite origins | Browser origins allowed to call the API. |
| `RAG_EMBEDDING_PROVIDER` | `hashing` | `hashing` or `openai_compatible`. |
| `RAG_LLM_PROVIDER` | `extractive` | `extractive` produces evidence-guided implementation plans; `openai_compatible` additionally produces a grounded model-assisted plan from retrieved Patchwork evidence. |
| `RAG_OPENAI_BASE_URL` | `https://api.openai.com/v1` | Compatible API base URL. |
| `RAG_OPENAI_API_KEY` | unset | Required when using the compatible provider. |

For a hosted provider, select `openai_compatible` for both provider settings and set `RAG_OPENAI_API_KEY`, `RAG_EMBEDDING_MODEL`, and `RAG_CHAT_MODEL`.

## Project layout

```text
src/rag_document_search/
  api.py                  FastAPI endpoints and CORS configuration
  github.py               Safe GitHub URL parsing, REST/GraphQL client, patch cards
  github_oauth.py         OAuth code flow and encrypted GitHub account credentials
  investigator.py         Bounded evidence-agent planning, search fusion, confidence
  patchwork.py            Sync, investigate, evaluate, and lifecycle orchestration
  repository.py           PostgreSQL schema, storage, scoped search, run history
  worker.py               PostgreSQL-claimed durable sync worker
  service.py              Ingestion, retrieval, and grounded answer service
  chunking.py             Text splitting
  embeddings.py           Local hashing and compatible API embeddings
  loaders.py              General local-document loaders
  cli.py                  Patchwork and generic-document CLI commands
web/
  src/App.tsx             React Patchwork dashboard
  src/styles.css          Tailwind theme and shared presentation details
tests/                    Offline unit tests
compose.yaml              Local PostgreSQL service
compose.production.yaml   Durable API, web dashboard, and PostgreSQL deployment
examples/documents/       Small generic-RAG sample corpus
```

## Development

```bash
pytest
ruff check .
cd web && npm run build
```

The local storage implementation uses PostgreSQL `REAL[]` vectors and scores similarity in the application, which keeps setup simple and transparent. For higher-throughput deployments, retain the service interfaces and use pgvector or a dedicated vector index for database-side ranking. The current deployment includes account scoping, GitHub OAuth linking, encrypted credential storage, and a durable sync worker. The highest-leverage next upgrade is a GitHub App installation flow with narrow read-only permissions, followed by pgvector, GitHub webhooks, and retrieval/reranking experiments measured against the stored evaluation manifest.
