"""
aarav-lm · Web accounts: signup, login, sessions, and self-service API keys

    GET  /app/config              which sign-in mode the server is in (no auth): none | key | account, and signup policy
    POST /account/signup          {email, password, invite_code}   -> session cookie
    POST /account/login           {email, password}                -> session cookie
    POST /account/logout
    GET  /account/me              {email, csrf_token}
    POST /account/password        {current, new}                   ends every other session
    GET  /account/keys            the caller's keys (never the secrets)
    POST /account/keys            {label}   -> the new key, shown once
    POST /account/keys/{id}/revoke
    POST /account/keys/{id}/rotate                                   -> the replacement key, shown once
    GET  /account/usage?days=14   totals and a per-day series

Security model
* Passwords: salted scrypt (api/store.py), 10-128 characters, never logged or returned. Login and signup are rate limited
  per client and per email, and a wrong email takes as long as a wrong password.
* Sessions: a random cookie value (only its SHA-256 is stored), HttpOnly, SameSite=Strict, `Secure` on HTTPS, 7 days.
  Every state-changing request made with a cookie must carry the session's CSRF token in `X-CSRF-Token`; JSON-only bodies
  mean a cross-site form cannot forge one either.
* Signup is closed unless the owner sets an invite code (or opens it). Email verification would need an email service, so
  the invite code is the abuse control; new keys get conservative limits.
* Errors are deliberately vague ("invalid email or password") so they cannot be used to find out who has an account.
"""

from __future__ import annotations

import hmac
import re
from typing import Callable, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from aarav_lm.api.auth import RateLimiter
from aarav_lm.api.store import AccountStore, KeyInfo, SessionInfo

COOKIE = "aarav_session"
SESSION_DAYS = 7.0
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class SignupBody(BaseModel):
    email: str = Field(..., max_length=254)
    password: str = Field(..., max_length=256)
    invite_code: Optional[str] = Field(None, max_length=200)


class LoginBody(BaseModel):
    email: str = Field(..., max_length=254)
    password: str = Field(..., max_length=256)


class PasswordBody(BaseModel):
    current: str = Field(..., max_length=256)
    new: str = Field(..., max_length=256)


class KeyBody(BaseModel):
    label: str = Field("", max_length=40)


def key_json(k: KeyInfo) -> dict:
    return {"id": k.id, "prefix": k.prefix, "label": k.label, "created_at": k.created_at, "last_used_at": k.last_used_at,
            "expires_at": k.expires_at, "revoked_at": k.revoked_at, "active": k.active,
            "rpm_limit": k.rpm_limit, "tpm_limit": k.tpm_limit}


def client_ip(request: Request, trust_proxy: bool) -> str:
    """The caller's address. Behind a reverse proxy every request comes from the proxy, so with `trust_proxy` the last
    X-Forwarded-For entry (the one the proxy itself appended) is used. Never enable it without a proxy: it is spoofable."""
    if trust_proxy:
        fwd = request.headers.get("x-forwarded-for", "")
        if fwd:
            return fwd.split(",")[-1].strip() or "unknown"
    return request.client.host if request.client else "unknown"


def session_from(request: Request, store: AccountStore) -> Optional[SessionInfo]:
    return store.get_session(request.cookies.get(COOKIE))


def csrf_ok(request: Request, session: SessionInfo) -> bool:
    return hmac.compare_digest(request.headers.get("x-csrf-token", ""), session.csrf)


def add_account_routes(
    app: FastAPI, store: AccountStore, *, model_name: str, auth_mode: str, signup_code: Optional[str], open_signup: bool,
    default_key_rpm: float, default_key_tpm: int, web_rpm: float, web_tpm: int, max_keys_per_user: int,
    cookie_secure: Optional[bool], trust_proxy: bool,
) -> None:
    login_by_ip = RateLimiter(per_minute=10, burst=10)
    login_by_email = RateLimiter(per_minute=0.5, burst=5)             # five tries, then one every two minutes
    signup_by_ip = RateLimiter(per_minute=0.2, burst=5)               # five, then one every five minutes

    def secure_cookie(request: Request) -> bool:
        if cookie_secure is not None:
            return cookie_secure
        return request.url.scheme == "https" or (trust_proxy and request.headers.get("x-forwarded-proto") == "https")

    def start_session(request: Request, response: Response, user_id: int) -> SessionInfo:
        token, csrf = store.create_session(user_id, SESSION_DAYS)
        response.set_cookie(COOKIE, token, max_age=int(SESSION_DAYS * 86400), httponly=True, samesite="strict",
                            secure=secure_cookie(request), path="/")
        return SessionInfo(user_id, store.user_info(user_id)["email"], csrf)

    def throttle(limiter: RateLimiter, identity: str, what: str) -> None:
        ok, retry = limiter.allow(identity)
        if not ok:
            raise HTTPException(429, f"too many {what} attempts, try again later", headers={"Retry-After": str(int(retry) + 1)})

    def need_session(request: Request) -> SessionInfo:
        s = session_from(request, store)
        if s is None:
            raise HTTPException(401, "not signed in")
        if request.method not in SAFE_METHODS and not csrf_ok(request, s):
            raise HTTPException(403, "missing or invalid CSRF token")
        return s

    def me_json(s: SessionInfo) -> dict:
        return {"email": s.email, "csrf_token": s.csrf}

    @app.get("/app/config")
    async def app_config():
        signup = "open" if open_signup else ("invite" if signup_code else "closed")
        return {"auth": auth_mode, "signup": signup, "model": model_name,
                "limits": {"web_requests_per_minute": web_rpm, "web_tokens_per_minute": web_tpm,
                           "key_requests_per_minute": default_key_rpm, "key_tokens_per_minute": default_key_tpm,
                           "max_keys": max_keys_per_user}}

    @app.post("/account/signup")
    def signup(body: SignupBody, request: Request, response: Response):
        throttle(signup_by_ip, client_ip(request, trust_proxy), "signup")
        if not open_signup:
            if not signup_code:
                raise HTTPException(403, "signup is closed; ask the owner for an invite")
            if not hmac.compare_digest((body.invite_code or "").encode(), signup_code.encode()):
                raise HTTPException(403, "that invite code is not valid")
        try:
            uid = store.signup(body.email, body.password)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        return me_json(start_session(request, response, uid))

    @app.post("/account/login")
    def login(body: LoginBody, request: Request, response: Response):
        throttle(login_by_ip, client_ip(request, trust_proxy), "sign-in")
        throttle(login_by_email, body.email.strip().lower(), "sign-in")
        uid = store.check_password(body.email, body.password)
        if uid is None:
            raise HTTPException(401, "invalid email or password")
        return me_json(start_session(request, response, uid))

    @app.post("/account/logout")
    def logout(request: Request, response: Response, s: SessionInfo = Depends(need_session)):
        store.end_session(request.cookies[COOKIE])
        response.delete_cookie(COOKIE, path="/")
        return {"ok": True}

    @app.get("/account/me")
    def me(s: SessionInfo = Depends(need_session)):
        return me_json(s)

    @app.post("/account/password")
    def change_password(body: PasswordBody, request: Request, s: SessionInfo = Depends(need_session)):
        if store.check_password(s.email, body.current) is None:
            raise HTTPException(401, "the current password is wrong")
        try:
            store.set_password(s.user_id, body.new)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        store.end_sessions(s.user_id, keep_token=request.cookies[COOKIE])
        return {"ok": True}

    def owned_key(s: SessionInfo, key_id: int) -> KeyInfo:
        for k in store.list_keys(s.user_id):
            if k.id == key_id:
                return k
        raise HTTPException(404, "no such key")                        # someone else's key looks exactly like a missing one

    @app.get("/account/keys")
    def list_keys(s: SessionInfo = Depends(need_session)) -> List[dict]:
        return [key_json(k) for k in store.list_keys(s.user_id)]

    @app.post("/account/keys")
    def create_key(body: KeyBody, s: SessionInfo = Depends(need_session)):
        if store.count_active_keys(s.user_id) >= max_keys_per_user:
            raise HTTPException(409, f"you already have {max_keys_per_user} active keys; revoke one first")
        label = _CONTROL.sub("", body.label).strip()
        key = store.create_key(s.user_id, label, rpm_limit=default_key_rpm, tpm_limit=default_key_tpm)
        info = next(k for k in store.list_keys(s.user_id) if key.startswith(k.prefix[:-1]) and k.active and k.label == label)
        return {"key": key, "info": key_json(info)}

    @app.post("/account/keys/{key_id}/revoke")
    def revoke_key(key_id: int, s: SessionInfo = Depends(need_session)):
        owned_key(s, key_id)
        store.revoke_key(key_id)
        return {"ok": True}

    @app.post("/account/keys/{key_id}/rotate")
    def rotate_key(key_id: int, s: SessionInfo = Depends(need_session)):
        k = owned_key(s, key_id)
        if not k.active:
            raise HTTPException(409, "that key is already revoked or expired")
        new = store.rotate_key(key_id)
        info = next(x for x in store.list_keys(s.user_id) if x.rotated_from == key_id)
        return {"key": new, "info": key_json(info)}

    @app.get("/account/usage")
    def usage(days: int = 14, s: SessionInfo = Depends(need_session)):
        if not 1 <= days <= 90:
            raise HTTPException(400, "days must be between 1 and 90")
        return {"totals": store.usage_summary(user_id=s.user_id, days=days), "daily": store.usage_daily(s.user_id, days),
                "limits": {"tokens_per_minute": web_tpm, "requests_per_minute": web_rpm}}
