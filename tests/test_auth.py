from datetime import UTC, datetime

import pytest
from rag_document_search.auth import AuthenticationError, AuthenticationService


class FakeAuthIndex:
    def __init__(self) -> None:
        self.users: dict[str, dict[str, object]] = {}
        self.sessions: dict[str, str] = {}

    def create_user(self, email: str, password_hash: str) -> dict[str, object]:
        if email in self.users:
            raise ValueError("An account already exists for that email address")
        user = {
            "id": str(len(self.users) + 1),
            "email": email,
            "password_hash": password_hash,
            "created_at": "now",
        }
        self.users[email] = user
        return user

    def get_user_by_email(self, email: str) -> dict[str, object] | None:
        return self.users.get(email)

    def claim_unowned_repositories(self, _user_id: str) -> int:
        return 0

    def create_session(self, user_id: str, token_hash: str, expires_at: object) -> None:
        assert expires_at > datetime.now(UTC)
        self.sessions[token_hash] = user_id

    def get_user_for_session(self, token_hash: str) -> dict[str, object] | None:
        user_id = self.sessions.get(token_hash)
        return next((user for user in self.users.values() if user["id"] == user_id), None)

    def revoke_session(self, token_hash: str) -> None:
        self.sessions.pop(token_hash, None)


def test_authentication_register_login_and_logout() -> None:
    auth = AuthenticationService(FakeAuthIndex())  # type: ignore[arg-type]
    registration = auth.register("Engineer@Example.com", "secure-password")

    assert registration["user"]["email"] == "engineer@example.com"
    assert auth.user_for_token(str(registration["token"]))["id"] == "1"

    auth.logout(str(registration["token"]))
    assert auth.user_for_token(str(registration["token"])) is None
    assert auth.login("engineer@example.com", "secure-password")["user"]["id"] == "1"


def test_authentication_rejects_wrong_password() -> None:
    auth = AuthenticationService(FakeAuthIndex())  # type: ignore[arg-type]
    auth.register("engineer@example.com", "secure-password")

    with pytest.raises(AuthenticationError):
        auth.login("engineer@example.com", "wrong-password")
