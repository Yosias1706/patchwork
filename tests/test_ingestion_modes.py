from unittest.mock import Mock

import pytest

from rag_document_search.patchwork import PatchworkService


@pytest.fixture
def service() -> PatchworkService:
    service = object.__new__(PatchworkService)
    service.rag = Mock()
    service.rag.settings.patchwork_default_pull_limit = 15
    service.index = Mock()
    service.index.upsert_patchwork_repository.return_value = {"id": "repo"}
    service._github_client = Mock()
    return service


@pytest.mark.parametrize("options", [{}, {"ingestion_mode": "detailed"}])
def test_connect_accepts_default_and_explicit_detailed_ingestion(service, options) -> None:
    repository, run = service.connect_repository(
        "https://github.com/example/api", user_id="user", **options
    )

    assert repository == {"id": "repo"}
    assert run is service.index.create_patchwork_sync_run.return_value
    service.index.create_patchwork_sync_run.assert_called_once_with("repo", 15)
    service._github_client.assert_called_once_with("user")


@pytest.mark.parametrize("mode", ["bulk", "unknown"])
def test_unimplemented_ingestion_modes_fail_before_network_or_storage(service, mode) -> None:
    with pytest.raises(ValueError, match="Use detailed ingestion"):
        service.connect_repository(
            "https://github.com/example/api", user_id="user", ingestion_mode=mode
        )

    service._github_client.assert_not_called()
    service.index.upsert_patchwork_repository.assert_not_called()
    service.index.create_patchwork_sync_run.assert_not_called()
