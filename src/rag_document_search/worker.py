"""Database-backed sync worker for durable Patchwork ingestion jobs."""

from __future__ import annotations

import threading

from .patchwork import PatchworkService


class SyncWorker:
    """Claim persisted sync runs until stopped by the FastAPI lifespan manager."""

    def __init__(self, patchwork: PatchworkService) -> None:
        self.patchwork = patchwork
        self._stopped = threading.Event()
        self._wakeup = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stopped.clear()
        self._thread = threading.Thread(
            target=self._run, name="patchwork-sync-worker", daemon=True
        )
        self._thread.start()

    def notify(self) -> None:
        self._wakeup.set()

    def stop(self) -> None:
        self._stopped.set()
        self._wakeup.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        settings = self.patchwork.rag.settings
        while not self._stopped.is_set():
            run = self.patchwork.index.claim_next_patchwork_sync_run(
                lease_seconds=settings.sync_lease_seconds
            )
            if run:
                try:
                    self.patchwork.sync_repository(
                        str(run["repository_id"]), str(run["id"])
                    )
                except Exception as error:  # Keep a single failed job from killing the worker.
                    self.patchwork.index.update_patchwork_sync_run(
                        str(run["id"]), status="failed", error=str(error)[:1000]
                    )
                continue
            self._wakeup.wait(timeout=settings.sync_worker_poll_seconds)
            self._wakeup.clear()
