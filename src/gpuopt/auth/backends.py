from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


@dataclass
class AuthResult:
    authenticated: bool = False
    username: str = ""
    email: str = ""
    groups: list[str] = ()
    metadata: dict[str, Any] = ()
    error: str = ""


class AuthenticationBackend(ABC):
    @abstractmethod
    def authenticate(self, token: str) -> AuthResult:
        ...


class OAuth2PasswordBackend(AuthenticationBackend):
    def __init__(
        self,
        token_url: str = "",
        client_id: str = "",
        client_secret: str = "",
        verify_url: str = "",
    ) -> None:
        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._verify_url = verify_url

    def authenticate(self, token: str) -> AuthResult:
        if not self._verify_url:
            return AuthResult(error="OAuth2 not configured")
        if not token.startswith("Bearer "):
            token = f"Bearer {token}"
        try:
            req = Request(
                self._verify_url,
                headers={"Authorization": token, "Accept": "application/json"},
                method="GET",
            )
            with urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read().decode())
            username = body.get("preferred_username") or body.get("sub", "")
            return AuthResult(
                authenticated=True,
                username=username,
                email=body.get("email", ""),
                groups=body.get("groups", []),
                metadata=body,
            )
        except Exception as exc:
            logger.warning("OAuth2 verification failed: %s", exc)
            return AuthResult(error=str(exc))


class OIDCBackend(AuthenticationBackend):
    def __init__(
        self,
        issuer_url: str = "",
        client_id: str = "",
    ) -> None:
        self._issuer_url = issuer_url.rstrip("/")
        self._client_id = client_id
        self._jwks_uri = ""
        self._load_config()

    def _load_config(self) -> None:
        if not self._issuer_url:
            return
        try:
            req = Request(f"{self._issuer_url}/.well-known/openid-configuration")
            with urlopen(req, timeout=10) as resp:
                config = json.loads(resp.read().decode())
            self._jwks_uri = config.get("jwks_uri", "")
            logger.info("OIDC configured: issuer=%s jwks_uri=%s", self._issuer_url, self._jwks_uri)
        except Exception as exc:
            logger.warning("Failed to load OIDC config from %s: %s", self._issuer_url, exc)

    def authenticate(self, token: str) -> AuthResult:
        if not self._issuer_url:
            return AuthResult(error="OIDC not configured")
        if not self._jwks_uri:
            return AuthResult(error="OIDC issuer unreachable")
        try:
            req = Request(
                f"{self._issuer_url}/userinfo",
                headers={"Authorization": f"Bearer {token}"},
                method="GET",
            )
            with urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read().decode())
            username = body.get("preferred_username") or body.get("sub", "")
            return AuthResult(
                authenticated=True,
                username=username,
                email=body.get("email", ""),
                groups=body.get("groups", []),
                metadata=body,
            )
        except Exception as exc:
            logger.warning("OIDC userinfo failed: %s", exc)
            return AuthResult(error=str(exc))


class LDAPBackend(AuthenticationBackend):
    def __init__(
        self,
        server_url: str = "",
        bind_dn: str = "",
        bind_password: str = "",
        search_base: str = "",
        search_filter: str = "(uid={username})",
    ) -> None:
        self._server_url = server_url
        self._bind_dn = bind_dn
        self._bind_password = bind_password
        self._search_base = search_base
        self._search_filter = search_filter

    def authenticate(self, token: str) -> AuthResult:
        if not self._server_url:
            return AuthResult(error="LDAP not configured")
        try:
            import ldap3

            server = ldap3.Server(self._server_url, get_info=ldap3.ALL)
            conn = ldap3.Connection(server, self._bind_dn, self._bind_password, auto_bind=True)
            conn.search(
                search_base=self._search_base,
                search_filter=self._search_filter.format(username=token),
                attributes=["cn", "mail", "memberOf"],
            )
            if not conn.entries:
                return AuthResult(error="User not found in LDAP")
            entry = conn.entries[0]
            groups = []
            if "memberOf" in entry:
                groups = [str(g) for g in entry.memberOf]
            return AuthResult(
                authenticated=True,
                username=str(getattr(entry, "cn", entry.entry_dn)),
                email=str(getattr(entry, "mail", "")),
                groups=groups,
            )
        except ImportError:
            return AuthResult(error="ldap3 package not installed")
        except Exception as exc:
            logger.warning("LDAP authentication failed: %s", exc)
            return AuthResult(error=str(exc))
