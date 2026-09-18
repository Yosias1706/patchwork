"""Command-line workflows for Patchwork and the generic RAG service."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .auth import AuthenticationService
from .github import GitHubError
from .patchwork import PatchworkService
from .service import RAGService


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="patchwork", description="Repository-aware historical bug-fix retrieval"
    )
    parser.add_argument(
        "--session-token",
        default=os.getenv("PATCHWORK_SESSION_TOKEN"),
        help="Authenticated session token from the local Patchwork API",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    connect = commands.add_parser("connect", help="Connect and synchronously index a repository")
    connect.add_argument("github_url", help="GitHub.com repository URL")
    connect.add_argument("--pull-limit", type=int, default=None)

    sync = commands.add_parser("sync", help="Synchronously refresh a repository's patch cards")
    sync.add_argument("repository_id")
    sync.add_argument("--pull-limit", type=int, default=None)

    commands.add_parser("repositories", help="List connected repositories and latest run state")

    investigate = commands.add_parser("investigate", help="Find analogous historical fixes")
    investigate.add_argument("repository_id")
    investigate.add_argument("issue", help="Bug report, stack trace, or incident description")
    investigate.add_argument("--limit", type=int, default=5)

    evaluate = commands.add_parser("evaluate", help="Evaluate linked issue-to-fix retrieval")
    evaluate.add_argument("repository_id")
    evaluate.add_argument("--limit", type=int, default=5)

    remove = commands.add_parser("remove", help="Remove a repository and its patch cards")
    remove.add_argument("repository_id")

    ingest = commands.add_parser("ingest", help="Index a generic local file or directory")
    ingest.add_argument("path", type=Path)
    ingest.add_argument(
        "--fail-fast", action="store_true", help="Stop at the first unreadable file"
    )

    search = commands.add_parser("search", help="Retrieve generic indexed passages")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=None)

    ask = commands.add_parser("ask", help="Answer from generic indexed documents")
    ask.add_argument("question")
    ask.add_argument("--limit", type=int, default=None)

    commands.add_parser("documents", help="List generic indexed documents")
    delete = commands.add_parser("delete", help="Delete one generic indexed document")
    delete.add_argument("document_id")
    return parser


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, default=str))


def main() -> None:
    args = _parser().parse_args()
    service = RAGService()
    try:
        patchwork_commands = {
            "connect",
            "sync",
            "repositories",
            "investigate",
            "evaluate",
            "remove",
        }
        patchwork = PatchworkService(service) if args.command in patchwork_commands else None
        user: dict[str, object] | None = None
        if patchwork:
            if not args.session_token:
                raise ValueError(
                    "Patchwork commands require --session-token or PATCHWORK_SESSION_TOKEN"
                )
            user = AuthenticationService(patchwork.index).user_for_token(args.session_token)
            if not user:
                raise ValueError("Patchwork session is missing or expired")
        user_id = str(user["id"]) if user else ""
        if args.command == "connect":
            repository, run = patchwork.connect_repository(
                args.github_url, user_id=user_id, pull_limit=args.pull_limit
            )
            patchwork.sync_repository(str(repository["id"]), str(run["id"]))
            _print_json(
                {
                    "repository": patchwork.repository(str(repository["id"]), user_id=user_id),
                    "sync_run": patchwork.index.get_patchwork_sync_run(str(run["id"])),
                }
            )
        elif args.command == "sync":
            existing = patchwork.repository(args.repository_id, user_id=user_id)
            if not existing:
                raise KeyError(f"Repository not found: {args.repository_id}")
            repository, run = patchwork.connect_repository(
                str(existing["github_url"]), user_id=user_id, pull_limit=args.pull_limit
            )
            patchwork.sync_repository(str(repository["id"]), str(run["id"]))
            _print_json(patchwork.index.get_patchwork_sync_run(str(run["id"])))
        elif args.command == "repositories":
            _print_json(patchwork.repositories(user_id=user_id))
        elif args.command == "investigate":
            _print_json(
                patchwork.investigate(
                    args.repository_id, args.issue, user_id=user_id, limit=args.limit
                )
            )
        elif args.command == "evaluate":
            _print_json(patchwork.evaluate(args.repository_id, user_id=user_id, limit=args.limit))
        elif args.command == "remove":
            if not patchwork.delete_repository(args.repository_id, user_id=user_id):
                raise KeyError(f"Repository not found: {args.repository_id}")
            print(f"Removed {args.repository_id}")
        elif args.command == "ingest":
            report = (
                service.ingest_directory(args.path, continue_on_error=not args.fail_fast)
                if args.path.is_dir()
                else service.ingest_file(args.path)
            )
            _print_json(asdict(report))
        elif args.command == "search":
            _print_json([asdict(hit) for hit in service.search(args.query, limit=args.limit)])
        elif args.command == "ask":
            answer = service.ask(args.question, limit=args.limit)
            print(answer.text)
            if answer.citations:
                print("\nSources:")
                for number, hit in enumerate(answer.citations, start=1):
                    print(f"[{number}] {hit.source} (score {hit.score:.3f})")
        elif args.command == "documents":
            _print_json(service.list_documents())
        elif args.command == "delete":
            if not service.delete_document(args.document_id):
                raise KeyError(f"Document not found: {args.document_id}")
            print(f"Deleted {args.document_id}")
    except (GitHubError, KeyError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
