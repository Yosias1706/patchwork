"""FastAPI application exposing the RAG service."""

from __future__ import annotations

import os
import shutil
import urllib.parse
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .auth import AuthenticationError, AuthenticationService
from .github import GitHubError
from .github_oauth import GitHubOAuthError
from .patchwork import PatchworkService
from .service import RAGService
from .worker import SyncWorker

service = RAGService()
patchwork = PatchworkService(service)
authentication = AuthenticationService(patchwork.index)
sync_worker = SyncWorker(patchwork)
bearer_scheme = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    sync_worker.start()
    try:
        yield
    finally:
        sync_worker.stop()


app = FastAPI(title="Patchwork", version="0.3.0", lifespan=lifespan)
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "RAG_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
class DirectoryIngestRequest(BaseModel):
    path: str = Field(description="Server-local directory containing supported documents")
    continue_on_error: bool = True


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int | None = Field(default=None, ge=1, le=50)


class RepositoryConnectRequest(BaseModel):
    github_url: str = Field(description="GitHub repository URL, for example https://github.com/fastapi/fastapi")
    pull_limit: int | None = Field(default=None, ge=1, le=10_000)


class InvestigateRequest(BaseModel):
    issue: str = Field(
        min_length=8,
        description="Bug report, stack trace, issue URL, or incident description",
    )
    limit: int = Field(default=5, ge=1, le=10)


class RepositorySyncRequest(BaseModel):
    pull_limit: int | None = Field(default=None, ge=1, le=10_000)


class CredentialsRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=10, max_length=256)


def current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict[str, object]:
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sign in required")
    user = authentication.user_for_token(credentials.credentials)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired")
    return user


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "product": "patchwork",
        "sync_worker": "ready",
        "github_oauth": "configured" if patchwork.github_oauth.configured else "optional",
    }


@app.post("/auth/register", status_code=status.HTTP_201_CREATED)
def register(request: CredentialsRequest) -> dict[str, object]:
    try:
        return authentication.register(request.email, request.password)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/auth/login")
def login(request: CredentialsRequest) -> dict[str, object]:
    try:
        return authentication.login(request.email, request.password)
    except AuthenticationError as error:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/auth/me")
def auth_me(user: dict[str, object] = Depends(current_user)) -> dict[str, object]:
    return user


@app.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)) -> None:
    if credentials and credentials.scheme.lower() == "bearer":
        authentication.logout(credentials.credentials)


@app.get("/auth/github/status")
def github_connection_status(
    user: dict[str, object] = Depends(current_user),
) -> dict[str, object]:
    return patchwork.github_oauth.status(str(user["id"]))


@app.post("/auth/github/authorize")
def authorize_github(
    user: dict[str, object] = Depends(current_user),
) -> dict[str, str]:
    try:
        return {"authorization_url": patchwork.github_oauth.authorization_url(str(user["id"]))}
    except GitHubOAuthError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/auth/github/callback")
def github_oauth_callback(
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
) -> RedirectResponse:
    result = "connected"
    if error or not code or not state:
        result = "cancelled"
    else:
        try:
            patchwork.github_oauth.complete_callback(code=code, state=state)
        except GitHubOAuthError:
            result = "failed"
    target = f"{service.settings.patchwork_web_url}?{urllib.parse.urlencode({'github': result})}"
    return RedirectResponse(target, status_code=status.HTTP_303_SEE_OTHER)


@app.delete("/auth/github/connection")
def disconnect_github(
    user: dict[str, object] = Depends(current_user),
) -> dict[str, bool]:
    return {"disconnected": patchwork.github_oauth.disconnect(str(user["id"]))}


@app.get("/patchwork/repositories")
def patchwork_repositories(
    user: dict[str, object] = Depends(current_user),
) -> list[dict[str, object]]:
    return patchwork.repositories(user_id=str(user["id"]))


@app.post("/patchwork/repositories", status_code=202)
def connect_patchwork_repository(
    request: RepositoryConnectRequest,
    user: dict[str, object] = Depends(current_user),
) -> dict[str, object]:
    try:
        repository, run = patchwork.connect_repository(
            request.github_url, user_id=str(user["id"]), pull_limit=request.pull_limit
        )
        sync_worker.notify()
        return {"repository": repository, "sync_run": run}
    except (GitHubError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/patchwork/repositories/{repository_id}")
def patchwork_repository(
    repository_id: str, user: dict[str, object] = Depends(current_user)
) -> dict[str, object]:
    repository = patchwork.repository(repository_id, user_id=str(user["id"]))
    if not repository:
        raise HTTPException(status_code=404, detail="Repository not found")
    return repository


@app.post("/patchwork/repositories/{repository_id}/sync", status_code=202)
def sync_patchwork_repository(
    repository_id: str,
    request: RepositorySyncRequest,
    user: dict[str, object] = Depends(current_user),
) -> dict[str, object]:
    user_id = str(user["id"])
    existing = patchwork.repository(repository_id, user_id=user_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Repository not found")
    try:
        _, run = patchwork.connect_repository(
            str(existing["github_url"]), user_id=user_id, pull_limit=request.pull_limit
        )
        sync_worker.notify()
        return {"repository": existing, "sync_run": run}
    except (GitHubError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/patchwork/repositories/{repository_id}/investigate")
def investigate_patchwork_repository(
    repository_id: str,
    request: InvestigateRequest,
    user: dict[str, object] = Depends(current_user),
) -> dict[str, object]:
    try:
        return patchwork.investigate(
            repository_id, request.issue, user_id=str(user["id"]), limit=request.limit
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Repository not found") from error
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/patchwork/repositories/{repository_id}/investigations")
def patchwork_investigations(
    repository_id: str, user: dict[str, object] = Depends(current_user)
) -> list[dict[str, object]]:
    try:
        return patchwork.investigations(repository_id, user_id=str(user["id"]))
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Repository not found") from error


@app.post("/patchwork/repositories/{repository_id}/evaluate")
def evaluate_patchwork_repository(
    repository_id: str, user: dict[str, object] = Depends(current_user)
) -> dict[str, object]:
    try:
        return patchwork.evaluate(repository_id, user_id=str(user["id"]))
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Repository not found") from error
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.delete("/patchwork/repositories/{repository_id}")
def delete_patchwork_repository(
    repository_id: str, user: dict[str, object] = Depends(current_user)
) -> dict[str, bool]:
    if not patchwork.delete_repository(repository_id, user_id=str(user["id"])):
        raise HTTPException(status_code=404, detail="Repository not found")
    return {"deleted": True}


@app.post("/ingest/directory")
def ingest_directory(request: DirectoryIngestRequest) -> dict[str, object]:
    try:
        report = service.ingest_directory(
            Path(request.path), continue_on_error=request.continue_on_error
        )
        return asdict(report)
    except (NotADirectoryError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/ingest/file")
def ingest_file(file: Annotated[UploadFile, File(...)]) -> dict[str, object]:
    filename = Path(file.filename or "upload.txt").name
    suffix = Path(filename).suffix
    try:
        with NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            shutil.copyfileobj(file.file, temporary)
            path = Path(temporary.name)
        try:
            return asdict(service.ingest_file(path, source=f"upload:///{filename}"))
        finally:
            path.unlink(missing_ok=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/search")
def search(request: SearchRequest) -> list[dict[str, object]]:
    try:
        return [asdict(hit) for hit in service.search(request.query, limit=request.limit)]
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/ask")
def ask(request: SearchRequest) -> dict[str, object]:
    try:
        return asdict(service.ask(request.query, limit=request.limit))
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/documents")
def documents() -> list[dict[str, object]]:
    return service.list_documents()


@app.delete("/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, bool]:
    if not service.delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": True}


static_directory = os.getenv("PATCHWORK_WEB_DIST")
if static_directory and Path(static_directory).is_dir():
    app.mount("/", StaticFiles(directory=static_directory, html=True), name="dashboard")
