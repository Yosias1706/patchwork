"""GitHub OAuth account linking with encrypted, server-side credential storage."""

from __future__ import annotations

import hashlib
import json
import secrets
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from .config import Settings
from .github import GitHubClient, GitHubError
from .repository import PostgresIndex

_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
_TOKEN_URL = "https://github.com/login/oauth/access_token"


class GitHubOAuthError(ValueError):
    """A safe, actionable failure during GitHub account linking."""


class GitHubOAuthService:
    """Own the OAuth code flow and never expose an access token to the browser."""

    def __init__(self, index: PostgresIndex, settings: Settings) -> None:
        self.index = index
        self.settings = settings
        self._cipher = self._build_cipher()

    @property
    def configured(self) -> bool:
        return self._cipher is not None

    def status(self, user_id: str) -> dict[str, object]:
        connection = self.index.get_github_connection(user_id)
        if not connection:
            return {
                "configured": self.configured,
                "connected": False,
                "github_login": None,
                "scopes": [],
            }
        return {
            "configured": self.configured,
            "connected": True,
            "github_login": connection["github_login"],
            "scopes": connection["scopes"],
            "access_token_expires_at": connection["access_token_expires_at"],
            "updated_at": connection["updated_at"],
        }

    def authorization_url(self, user_id: str) -> str:
        self._require_configuration()
        state = secrets.token_urlsafe(32)
        self.index.create_github_oauth_state(
            state_hash=_hash_secret(state),
            user_id=user_id,
            expires_at=datetime.now(UTC) + timedelta(minutes=10),
        )
        query = urllib.parse.urlencode(
            {
                "client_id": self.settings.github_oauth_client_id,
                "redirect_uri": self.settings.github_oauth_redirect_url,
                "scope": " ".join(self.settings.github_oauth_scopes),
                "state": state,
            }
        )
        return f"{_AUTHORIZE_URL}?{query}"

    def complete_callback(self, *, code: str, state: str) -> dict[str, object]:
        self._require_configuration()
        user_id = self.index.consume_github_oauth_state(_hash_secret(state))
        if not user_id:
            raise GitHubOAuthError("This GitHub authorization link expired or was already used")

        token_payload = self._exchange_code(code)
        access_token = _required_text(token_payload, "access_token")
        try:
            viewer = GitHubClient(
                self.settings.github_api_url,
                token=access_token,
                timeout=self.settings.request_timeout_seconds,
                graphql_url=self.settings.github_graphql_url,
            ).viewer()
        except GitHubError as error:
            raise GitHubOAuthError(
                "GitHub authorization succeeded but account verification failed"
            ) from error

        expires_at = _access_token_expiry(token_payload)
        refresh_token = _optional_text(token_payload.get("refresh_token"))
        try:
            connection = self.index.upsert_github_connection(
                user_id=user_id,
                github_user_id=str(viewer["id"]),
                github_login=str(viewer["login"]),
                encrypted_access_token=self._encrypt(access_token),
                encrypted_refresh_token=self._encrypt(refresh_token) if refresh_token else None,
                access_token_expires_at=expires_at,
                scopes=_scopes(token_payload.get("scope"), self.settings.github_oauth_scopes),
            )
        except Exception as error:
            # Avoid leaking whether a GitHub identity is already linked elsewhere.
            raise GitHubOAuthError(
                "This GitHub account cannot be linked to this workspace"
            ) from error
        return {
            "connected": True,
            "github_login": connection["github_login"],
            "scopes": connection["scopes"],
        }

    def access_token_for_user(self, user_id: str) -> str | None:
        """Return a usable account token, refreshing expiring OAuth credentials when possible."""
        connection = self.index.get_github_connection(user_id)
        if not connection:
            return None
        expires_at = connection.get("access_token_expires_at")
        if _expires_soon(expires_at):
            return self._refresh_connection(user_id, connection)
        try:
            return self._decrypt(str(connection["encrypted_access_token"]))
        except InvalidToken as error:
            raise GitHubOAuthError(
                "Stored GitHub credentials are unreadable. Link GitHub again."
            ) from error

    def disconnect(self, user_id: str) -> bool:
        return self.index.delete_github_connection(user_id)

    def _refresh_connection(self, user_id: str, connection: dict[str, object]) -> str:
        encrypted_refresh = connection.get("encrypted_refresh_token")
        if not encrypted_refresh:
            raise GitHubOAuthError(
                "Your GitHub authorization expired. Link GitHub again to continue."
            )
        try:
            refresh_token = self._decrypt(str(encrypted_refresh))
        except InvalidToken as error:
            raise GitHubOAuthError(
                "Stored GitHub credentials are unreadable. Link GitHub again."
            ) from error
        payload = self._post_form(
            {
                "client_id": self.settings.github_oauth_client_id,
                "client_secret": self.settings.github_oauth_client_secret,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            }
        )
        access_token = _required_text(payload, "access_token")
        next_refresh = _optional_text(payload.get("refresh_token")) or refresh_token
        self.index.upsert_github_connection(
            user_id=user_id,
            github_user_id=str(connection["github_user_id"]),
            github_login=str(connection["github_login"]),
            encrypted_access_token=self._encrypt(access_token),
            encrypted_refresh_token=self._encrypt(next_refresh),
            access_token_expires_at=_access_token_expiry(payload),
            scopes=_scopes(
                payload.get("scope"), tuple(str(scope) for scope in connection["scopes"])
            ),
        )
        return access_token

    def _exchange_code(self, code: str) -> dict[str, Any]:
        return self._post_form(
            {
                "client_id": self.settings.github_oauth_client_id,
                "client_secret": self.settings.github_oauth_client_secret,
                "code": code,
                "redirect_uri": self.settings.github_oauth_redirect_url,
            }
        )

    def _post_form(self, values: dict[str, str | None]) -> dict[str, Any]:
        form_values = {key: value for key, value in values.items() if value}
        encoded = urllib.parse.urlencode(form_values).encode("utf-8")
        request = urllib.request.Request(
            _TOKEN_URL,
            data=encoded,
            method="POST",
            headers={"Accept": "application/json", "User-Agent": "Patchwork-RAG/0.2"},
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.settings.request_timeout_seconds
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (TimeoutError, urllib.error.URLError) as error:
            raise GitHubOAuthError("GitHub did not respond while linking your account") from error
        except (ValueError, UnicodeDecodeError) as error:
            raise GitHubOAuthError("GitHub returned an invalid authorization response") from error
        if not isinstance(payload, dict):
            raise GitHubOAuthError("GitHub returned an invalid authorization response")
        if payload.get("error"):
            raise GitHubOAuthError("GitHub did not approve this authorization request")
        return payload

    def _build_cipher(self) -> Fernet | None:
        client_credentials = (
            self.settings.github_oauth_client_id,
            self.settings.github_oauth_client_secret,
        )
        if not any(client_credentials):
            # A deployment may pre-provision its callback and encryption key
            # before OAuth credentials are issued. OAuth remains disabled until
            # GitHub supplies both client credentials.
            return None
        configured_values = (
            *client_credentials,
            self.settings.github_oauth_redirect_url,
            self.settings.token_encryption_key,
        )
        if not all(configured_values):
            raise ValueError(
                "GitHub OAuth requires client ID, client secret, redirect URL, and token "
                "encryption key"
            )
        try:
            return Fernet(str(self.settings.token_encryption_key).encode("utf-8"))
        except (ValueError, TypeError) as error:
            raise ValueError("PATCHWORK_TOKEN_ENCRYPTION_KEY must be a valid Fernet key") from error

    def _require_configuration(self) -> None:
        if not self._cipher:
            raise GitHubOAuthError(
                "GitHub account linking is not configured on this deployment yet"
            )

    def _encrypt(self, value: str) -> str:
        self._require_configuration()
        assert self._cipher is not None
        return self._cipher.encrypt(value.encode("utf-8")).decode("utf-8")

    def _decrypt(self, value: str) -> str:
        self._require_configuration()
        assert self._cipher is not None
        return self._cipher.decrypt(value.encode("utf-8")).decode("utf-8")


def _hash_secret(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = _optional_text(payload.get(key))
    if not value:
        raise GitHubOAuthError("GitHub did not return a usable access token")
    return value


def _optional_text(value: object) -> str | None:
    return str(value) if value is not None else None


def _access_token_expiry(payload: dict[str, Any]) -> datetime | None:
    value = payload.get("expires_in")
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        return None
    return datetime.now(UTC) + timedelta(seconds=seconds) if seconds > 0 else None


def _expires_soon(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        expiry = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=UTC)
    return expiry <= datetime.now(UTC) + timedelta(minutes=2)


def _scopes(value: object, fallback: tuple[str, ...]) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [scope for scope in value.replace(",", " ").split() if scope]
    return list(fallback)
