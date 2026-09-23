"""Small OAuth server for remote MCP clients.

Implements the same public shape used by the user's Excalimate deployment:
OAuth protected-resource metadata, authorization-server metadata, dynamic client
registration, Authorization Code + PKCE, refresh tokens, revocation, and a
password-gated authorization screen. State is persisted in SQLite.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import html
import json
import os
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route

SCOPE = "mcp"
OFFLINE_SCOPE = "offline_access"
SUPPORTED_SCOPES = {SCOPE, OFFLINE_SCOPE}
COOKIE_NAME = "video_analyzer_oauth_session"


def _now() -> int:
    return int(time.time())


def _random(prefix: str, size: int = 32) -> str:
    return prefix + secrets.token_urlsafe(size)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_equal(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    return hmac.compare_digest(a.encode(), b.encode())


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    value = int(raw)
    if value <= 0:
        raise RuntimeError(f"{name} must be > 0")
    return value


def _validate_redirect_uri(value: str) -> None:
    parsed = urlparse(value)
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError("Unsafe redirect URI")
    if parsed.scheme == "https" and parsed.netloc:
        return
    if parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
        return
    raise ValueError("Redirect URI must be HTTPS, except loopback native-client callbacks")


def _parse_form_bytes(body: bytes) -> dict[str, str]:
    parsed = parse_qs(body.decode("utf-8"), keep_blank_values=True)
    return {key: values[-1] for key, values in parsed.items() if values}


def _normalize_scope(raw: str | None) -> str:
    parts = list(dict.fromkeys(p for p in (raw or SCOPE).split() if p))
    if SCOPE not in parts or any(part not in SUPPORTED_SCOPES for part in parts):
        raise ValueError(f"Supported scopes: {SCOPE} {OFFLINE_SCOPE}")
    return " ".join(parts)


def _pkce_ok(verifier: str, expected: str) -> bool:
    if not 43 <= len(verifier) <= 128:
        return False
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    actual = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return _safe_equal(actual, expected)


def _signed_session(secret: str, ttl: int) -> str:
    expires = str(_now() + ttl)
    sig = hmac.new(secret.encode(), expires.encode(), hashlib.sha256).digest()
    return f"{expires}.{base64.urlsafe_b64encode(sig).decode().rstrip('=')}"


def _session_valid(value: str | None, secret: str) -> bool:
    if not value or "." not in value:
        return False
    expires, signature = value.split(".", 1)
    try:
        if int(expires) <= _now():
            return False
    except ValueError:
        return False
    expected = hmac.new(secret.encode(), expires.encode(), hashlib.sha256).digest()
    expected_b64 = base64.urlsafe_b64encode(expected).decode().rstrip("=")
    return _safe_equal(signature, expected_b64)


class Store:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA busy_timeout=30000")
        return db

    def _init(self) -> None:
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS oauth_clients (
                  client_id TEXT PRIMARY KEY,
                  redirect_uris TEXT NOT NULL,
                  client_name TEXT,
                  metadata TEXT NOT NULL,
                  created_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS oauth_codes (
                  code_hash TEXT PRIMARY KEY,
                  client_id TEXT NOT NULL,
                  redirect_uri TEXT NOT NULL,
                  code_challenge TEXT NOT NULL,
                  scope TEXT NOT NULL,
                  resource TEXT NOT NULL,
                  expires_at INTEGER NOT NULL,
                  used_at INTEGER
                );
                CREATE TABLE IF NOT EXISTS oauth_tokens (
                  access_token_hash TEXT PRIMARY KEY,
                  refresh_token_hash TEXT UNIQUE NOT NULL,
                  client_id TEXT NOT NULL,
                  scope TEXT NOT NULL,
                  resource TEXT NOT NULL,
                  expires_at INTEGER NOT NULL,
                  refresh_expires_at INTEGER NOT NULL,
                  revoked_at INTEGER
                );
            """)

    def cleanup(self) -> None:
        cutoff = _now() - 86400
        with self.connect() as db:
            db.execute("DELETE FROM oauth_codes WHERE expires_at < ? OR used_at < ?", (cutoff, cutoff))
            db.execute("DELETE FROM oauth_tokens WHERE refresh_expires_at < ? OR revoked_at < ?", (cutoff, cutoff))

    def register_client(self, redirect_uris: list[str], client_name: str | None, metadata: dict[str, Any]) -> str:
        client_id = _random("mcp_client_", 24)
        with self.connect() as db:
            db.execute(
                "INSERT INTO oauth_clients(client_id,redirect_uris,client_name,metadata,created_at) VALUES(?,?,?,?,?)",
                (client_id, json.dumps(redirect_uris), client_name, json.dumps(metadata), _now()),
            )
        return client_id

    def client(self, client_id: str) -> sqlite3.Row | None:
        with self.connect() as db:
            return db.execute("SELECT * FROM oauth_clients WHERE client_id=?", (client_id,)).fetchone()

    def issue_code(self, *, client_id: str, redirect_uri: str, challenge: str, scope: str, resource: str, ttl: int) -> str:
        code = _random("va_code_", 32)
        with self.connect() as db:
            db.execute(
                "INSERT INTO oauth_codes(code_hash,client_id,redirect_uri,code_challenge,scope,resource,expires_at,used_at) VALUES(?,?,?,?,?,?,?,NULL)",
                (_hash(code), client_id, redirect_uri, challenge, scope, resource, _now() + ttl),
            )
        return code

    def exchange_code(self, code: str, client_id: str, redirect_uri: str, verifier: str, resource: str) -> sqlite3.Row | None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM oauth_codes WHERE code_hash=?", (_hash(code),)).fetchone()
            if not row:
                db.rollback()
                return None
            if row["used_at"] or row["expires_at"] <= _now() or row["client_id"] != client_id or row["redirect_uri"] != redirect_uri or row["resource"] != resource or not _pkce_ok(verifier, row["code_challenge"]):
                db.rollback()
                return None
            db.execute("UPDATE oauth_codes SET used_at=? WHERE code_hash=?", (_now(), _hash(code)))
            db.commit()
            return row

    def issue_tokens(self, *, client_id: str, scope: str, resource: str, access_ttl: int, refresh_ttl: int) -> dict[str, Any]:
        access = _random("va_at_", 32)
        refresh = _random("va_rt_", 40)
        now = _now()
        with self.connect() as db:
            db.execute(
                "INSERT INTO oauth_tokens(access_token_hash,refresh_token_hash,client_id,scope,resource,expires_at,refresh_expires_at,revoked_at) VALUES(?,?,?,?,?,?,?,NULL)",
                (_hash(access), _hash(refresh), client_id, scope, resource, now + access_ttl, now + refresh_ttl),
            )
        return {"access_token": access, "token_type": "Bearer", "expires_in": access_ttl, "refresh_token": refresh, "scope": scope, "resource": resource}

    def rotate_refresh(self, refresh: str, client_id: str, resource: str, access_ttl: int, refresh_ttl: int) -> dict[str, Any] | None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM oauth_tokens WHERE refresh_token_hash=?", (_hash(refresh),)).fetchone()
            if not row or row["revoked_at"] or row["refresh_expires_at"] <= _now() or row["client_id"] != client_id or row["resource"] != resource:
                db.rollback()
                return None
            access = _random("va_at_", 32)
            next_refresh = _random("va_rt_", 40)
            now = _now()
            db.execute(
                "UPDATE oauth_tokens SET access_token_hash=?,refresh_token_hash=?,expires_at=?,refresh_expires_at=? WHERE refresh_token_hash=?",
                (_hash(access), _hash(next_refresh), now + access_ttl, now + refresh_ttl, _hash(refresh)),
            )
            db.commit()
            return {"access_token": access, "token_type": "Bearer", "expires_in": access_ttl, "refresh_token": next_refresh, "scope": row["scope"], "resource": row["resource"]}

    def validate_access(self, token: str, resource: str) -> bool:
        with self.connect() as db:
            row = db.execute("SELECT scope,resource,expires_at,revoked_at FROM oauth_tokens WHERE access_token_hash=?", (_hash(token),)).fetchone()
            return bool(row and not row["revoked_at"] and row["expires_at"] > _now() and row["resource"] == resource and SCOPE in row["scope"].split())

    def revoke(self, token: str) -> None:
        hashed = _hash(token)
        with self.connect() as db:
            db.execute("UPDATE oauth_tokens SET revoked_at=COALESCE(revoked_at,?) WHERE access_token_hash=? OR refresh_token_hash=?", (_now(), hashed, hashed))


@dataclass
class OAuth:
    issuer: str
    resource: str
    password: str
    session_secret: str
    store: Store
    access_ttl: int
    refresh_ttl: int
    code_ttl: int
    session_ttl: int
    static_api_key: str | None

    @classmethod
    def from_env(cls) -> "OAuth":
        issuer = _required_env("PUBLIC_BASE_URL").rstrip("/")
        if not issuer.startswith("https://"):
            raise RuntimeError("PUBLIC_BASE_URL must use https://")
        resource = issuer + "/mcp"
        password = _required_env("OAUTH_LOGIN_PASSWORD")
        secret = _required_env("OAUTH_SESSION_SECRET")
        if len(secret) < 32:
            raise RuntimeError("OAUTH_SESSION_SECRET must be at least 32 characters")
        db_path = os.getenv("OAUTH_DB_PATH", "/data/oauth.sqlite3")
        return cls(
            issuer=issuer,
            resource=resource,
            password=password,
            session_secret=secret,
            store=Store(db_path),
            access_ttl=_int_env("OAUTH_ACCESS_TOKEN_TTL_SECONDS", 3600),
            refresh_ttl=_int_env("OAUTH_REFRESH_TOKEN_TTL_SECONDS", 30 * 86400),
            code_ttl=_int_env("OAUTH_AUTH_CODE_TTL_SECONDS", 300),
            session_ttl=_int_env("OAUTH_SESSION_TTL_SECONDS", 30 * 86400),
            static_api_key=os.getenv("MCP_API_KEY") or None,
        )

    def challenge_headers(self, invalid: bool = False) -> dict[str, str]:
        value = f'Bearer resource_metadata="{self.issuer}/.well-known/oauth-protected-resource", scope="{SCOPE}"'
        if invalid:
            value += ', error="invalid_token"'
        return {"WWW-Authenticate": value}

    def token_valid(self, token: str | None) -> bool:
        if not token:
            return False
        if self.static_api_key and _safe_equal(token, self.static_api_key):
            return True
        return self.store.validate_access(token, self.resource)

    def _client(self, client_id: str) -> tuple[list[str], str | None]:
        row = self.store.client(client_id)
        if not row:
            raise ValueError("Unknown OAuth client")
        return json.loads(row["redirect_uris"]), row["client_name"]

    def _auth_request(self, values: dict[str, str]) -> dict[str, str]:
        if values.get("response_type") != "code":
            raise ValueError("Only response_type=code is supported")
        client_id = values.get("client_id", "")
        redirect_uri = values.get("redirect_uri", "")
        challenge = values.get("code_challenge", "")
        method = values.get("code_challenge_method", "")
        if not client_id or not redirect_uri or not challenge:
            raise ValueError("Missing required OAuth authorization parameters")
        if method != "S256" or not 43 <= len(challenge) <= 128:
            raise ValueError("PKCE code_challenge_method=S256 is required")
        redirect_uris, _ = self._client(client_id)
        if redirect_uri not in redirect_uris:
            raise ValueError("redirect_uri is not registered")
        requested_resource = values.get("resource") or self.resource
        if requested_resource != self.resource:
            raise ValueError("OAuth resource mismatch")
        return {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "state": values.get("state", ""),
            "scope": _normalize_scope(values.get("scope")),
            "resource": requested_resource,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }

    def _page(self, request_data: dict[str, str], client_name: str | None, *, invalid: bool, authenticated: bool) -> str:
        hidden = "".join(
            f'<input type="hidden" name="{html.escape(k)}" value="{html.escape(v)}">'
            for k, v in request_data.items() if v
        )
        display = html.escape(client_name or "your MCP client")
        password = "" if authenticated else '<label>Password</label><input name="password" type="password" autocomplete="current-password" required autofocus>'
        error = '<div class="error">Incorrect password.</div>' if invalid else ""
        button = f"Authorize {display}" if authenticated else "Continue securely"
        return f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Connect Video Analyzer</title><style>body{{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#111;color:#f4f4f4;display:grid;place-items:center;min-height:100vh;margin:0}}.card{{width:min(420px,calc(100% - 36px));background:#1c1c1c;border:1px solid #333;border-radius:18px;padding:28px;box-shadow:0 20px 70px #0008}}h1{{margin:0 0 8px;font-size:22px}}p{{color:#aaa;line-height:1.5}}label{{display:block;margin:22px 0 8px;font-size:13px;color:#bbb}}input{{width:100%;box-sizing:border-box;padding:13px;border-radius:10px;border:1px solid #444;background:#111;color:#fff;font-size:16px}}button{{width:100%;margin-top:18px;padding:13px;border:0;border-radius:10px;background:#fff;color:#111;font-weight:700;font-size:15px;cursor:pointer}}.error{{background:#3a1e1e;color:#ffb2b2;padding:10px;border-radius:9px;margin-top:16px}}</style></head><body><div class="card"><h1>Video Analyzer</h1><p>Authorize <strong>{display}</strong> to use the Gemini video-analysis MCP.</p>{error}<form method="post" action="/oauth/authorize">{hidden}{password}<button type="submit">{button}</button></form></div></body></html>'''

    def routes(self) -> list[Route]:
        async def protected_resource(_request: Request) -> Response:
            return JSONResponse({
                "resource": self.resource,
                "authorization_servers": [self.issuer],
                "scopes_supported": [SCOPE],
                "bearer_methods_supported": ["header"],
                "resource_name": "Video Analyzer MCP",
            }, headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "public, max-age=300"})

        async def auth_server(_request: Request) -> Response:
            return JSONResponse({
                "issuer": self.issuer,
                "authorization_endpoint": self.issuer + "/oauth/authorize",
                "token_endpoint": self.issuer + "/oauth/token",
                "registration_endpoint": self.issuer + "/oauth/register",
                "revocation_endpoint": self.issuer + "/oauth/revoke",
                "authorization_response_iss_parameter_supported": True,
                "response_types_supported": ["code"],
                "response_modes_supported": ["query"],
                "grant_types_supported": ["authorization_code", "refresh_token"],
                "token_endpoint_auth_methods_supported": ["none"],
                "code_challenge_methods_supported": ["S256"],
                "scopes_supported": [SCOPE, OFFLINE_SCOPE],
            }, headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "public, max-age=300"})

        async def register(request: Request) -> Response:
            try:
                body = await request.json()
                redirect_uris = body.get("redirect_uris")
                if not isinstance(redirect_uris, list) or not 1 <= len(redirect_uris) <= 20:
                    raise ValueError("redirect_uris must contain 1-20 entries")
                uris: list[str] = []
                for uri in redirect_uris:
                    if not isinstance(uri, str):
                        raise ValueError("Invalid redirect URI")
                    _validate_redirect_uri(uri)
                    uris.append(uri)
                if body.get("token_endpoint_auth_method", "none") != "none":
                    raise ValueError("Only token_endpoint_auth_method=none is supported")
                client_name = body.get("client_name") if isinstance(body.get("client_name"), str) else None
                client_id = self.store.register_client(uris, client_name, body)
                return JSONResponse({
                    "client_id": client_id,
                    "client_id_issued_at": _now(),
                    "redirect_uris": uris,
                    "client_name": client_name,
                    "token_endpoint_auth_method": "none",
                    "grant_types": ["authorization_code", "refresh_token"],
                    "response_types": ["code"],
                }, status_code=201)
            except Exception as exc:
                return JSONResponse({"error": "invalid_client_metadata", "error_description": str(exc)}, status_code=400)

        async def authorize_get(request: Request) -> Response:
            try:
                data = self._auth_request(dict(request.query_params))
                _, name = self._client(data["client_id"])
                authenticated = _session_valid(request.cookies.get(COOKIE_NAME), self.session_secret)
                return HTMLResponse(self._page(data, name, invalid=False, authenticated=authenticated), headers={"Cache-Control": "no-store"})
            except Exception as exc:
                return JSONResponse({"error": "invalid_request", "error_description": str(exc)}, status_code=400)

        async def authorize_post(request: Request) -> Response:
            try:
                values = _parse_form_bytes(await request.body())
                data = self._auth_request(values)
                _, name = self._client(data["client_id"])
                authenticated = _session_valid(request.cookies.get(COOKIE_NAME), self.session_secret)
                if not authenticated and not _safe_equal(values.get("password"), self.password):
                    return HTMLResponse(self._page(data, name, invalid=True, authenticated=False), status_code=401, headers={"Cache-Control": "no-store"})
                code = self.store.issue_code(client_id=data["client_id"], redirect_uri=data["redirect_uri"], challenge=data["code_challenge"], scope=data["scope"], resource=data["resource"], ttl=self.code_ttl)
                parsed = urlparse(data["redirect_uri"])
                query = parse_qs(parsed.query, keep_blank_values=True)
                query["code"] = [code]
                if data.get("state"):
                    query["state"] = [data["state"]]
                query["iss"] = [self.issuer]
                redirect = urlunparse(parsed._replace(query=urlencode(query, doseq=True)))
                response = RedirectResponse(redirect, status_code=302)
                if not authenticated:
                    response.set_cookie(COOKIE_NAME, _signed_session(self.session_secret, self.session_ttl), max_age=self.session_ttl, httponly=True, secure=True, samesite="lax", path="/")
                return response
            except Exception as exc:
                return JSONResponse({"error": "invalid_request", "error_description": str(exc)}, status_code=400)

        async def token(request: Request) -> Response:
            try:
                values = _parse_form_bytes(await request.body())
                grant = values.get("grant_type")
                requested_resource = values.get("resource") or self.resource
                if requested_resource != self.resource:
                    raise ValueError("OAuth resource mismatch")
                if grant == "authorization_code":
                    row = self.store.exchange_code(values.get("code", ""), values.get("client_id", ""), values.get("redirect_uri", ""), values.get("code_verifier", ""), self.resource)
                    if not row:
                        return JSONResponse({"error": "invalid_grant", "error_description": "Authorization code is invalid, expired, consumed, or PKCE verification failed."}, status_code=400)
                    payload = self.store.issue_tokens(client_id=row["client_id"], scope=row["scope"], resource=row["resource"], access_ttl=self.access_ttl, refresh_ttl=self.refresh_ttl)
                    return JSONResponse(payload, headers={"Cache-Control": "no-store", "Pragma": "no-cache"})
                if grant == "refresh_token":
                    payload = self.store.rotate_refresh(values.get("refresh_token", ""), values.get("client_id", ""), self.resource, self.access_ttl, self.refresh_ttl)
                    if not payload:
                        return JSONResponse({"error": "invalid_grant", "error_description": "Refresh token is invalid, expired, or revoked."}, status_code=400)
                    return JSONResponse(payload, headers={"Cache-Control": "no-store", "Pragma": "no-cache"})
                return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)
            except Exception as exc:
                return JSONResponse({"error": "invalid_request", "error_description": str(exc)}, status_code=400)

        async def revoke(request: Request) -> Response:
            values = _parse_form_bytes(await request.body())
            if values.get("token"):
                self.store.revoke(values["token"])
            return Response(status_code=200)

        return [
            Route("/.well-known/oauth-protected-resource", protected_resource, methods=["GET"]),
            Route("/.well-known/oauth-protected-resource/mcp", protected_resource, methods=["GET"]),
            Route("/.well-known/oauth-authorization-server", auth_server, methods=["GET"]),
            Route("/oauth/register", register, methods=["POST"]),
            Route("/register", register, methods=["POST"]),
            Route("/oauth/authorize", authorize_get, methods=["GET"]),
            Route("/authorize", authorize_get, methods=["GET"]),
            Route("/oauth/authorize", authorize_post, methods=["POST"]),
            Route("/authorize", authorize_post, methods=["POST"]),
            Route("/oauth/token", token, methods=["POST"]),
            Route("/token", token, methods=["POST"]),
            Route("/oauth/revoke", revoke, methods=["POST"]),
        ]
