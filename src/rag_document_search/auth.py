"""Password and opaque-session authentication for the local Patchwork app."""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
from datetime import UTC, datetime, timedelta

from .repository import PostgresIndex

_EMAIL = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1


class AuthenticationError(ValueError):
    """Authentication failed without disclosing account-specific details."""


class AuthenticationService:
    """Owns password verification and revocable, database-backed bearer sessions."""

    def __init__(self, index: PostgresIndex, *, session_days: int = 14) -> None:
        self.index = index
        self.session_days = session_days

    def register(self, email: str, password: str) -> dict[str, object]:
        normalized_email = _normalize_email(email)
        _validate_password(password)
        user = self.index.create_user(normalized_email, _hash_password(password))
        self.index.claim_unowned_repositories(str(user["id"]))
        return self._session_response(user)

    def login(self, email: str, password: str) -> dict[str, object]:
        normalized_email = _normalize_email(email)
        user = self.index.get_user_by_email(normalized_email)
        if not user or not _verify_password(password, str(user["password_hash"])):
            raise AuthenticationError("Email or password is incorrect")
        return self._session_response(user)

    def user_for_token(self, token: str) -> dict[str, object] | None:
        return self.index.get_user_for_session(_token_hash(token))

    def logout(self, token: str) -> None:
        self.index.revoke_session(_token_hash(token))

    def _session_response(self, user: dict[str, object]) -> dict[str, object]:
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now(UTC) + timedelta(days=self.session_days)
        self.index.create_session(str(user["id"]), _token_hash(token), expires_at)
        return {
            "token": token,
            "expires_at": expires_at.isoformat(),
            "user": {key: user[key] for key in ("id", "email", "created_at")},
        }


def _normalize_email(email: str) -> str:
    value = email.strip().lower()
    if not _EMAIL.fullmatch(value):
        raise ValueError("Enter a valid email address")
    return value


def _validate_password(password: str) -> None:
    if len(password) < 10:
        raise ValueError("Password must be at least 10 characters")


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    derived = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P
    )
    return f"scrypt${_encode(salt)}${_encode(derived)}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, encoded_salt, encoded_hash = stored.split("$", maxsplit=2)
        if algorithm != "scrypt":
            return False
        expected = _decode(encoded_hash)
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=_decode(encoded_salt),
            n=_SCRYPT_N,
            r=_SCRYPT_R,
            p=_SCRYPT_P,
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value.encode("ascii"))
