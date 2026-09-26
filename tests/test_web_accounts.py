"""Web accounts over HTTP: signup policy, login, cookie sessions, CSRF, self-service keys, and the attacks on them."""

import pytest
from fastapi.testclient import TestClient

from aarav_lm.api.app import create_app
from aarav_lm.api.store import AccountStore

PW = "a long enough password"


@pytest.fixture()
def accounts(tmp_path):
    s = AccountStore(tmp_path / "accounts.db")
    yield s
    s.close()


def make(engine, accounts, base="http://testserver", **kw):
    kw.setdefault("signup_code", "invite-123")
    kw.setdefault("model_name", "exp_test")
    kw.setdefault("rate_limit_per_min", 600)
    kw.setdefault("rate_limit_burst", 100)
    kw.setdefault("queue_timeout", 5.0)
    return TestClient(create_app(engine, accounts, **kw), base_url=base)


def signup(client, email="a@example.com", pw=PW, code="invite-123"):
    r = client.post("/account/signup", json={"email": email, "password": pw, "invite_code": code})
    return r


def csrf(client):
    return {"X-CSRF-Token": client.get("/account/me").json()["csrf_token"]}


@pytest.fixture()
def client(engine, accounts):
    c = make(engine, accounts)
    assert signup(c).status_code == 200
    return c


# ── configuration and signup policy ──────────────────────────────────────────

def test_server_starts_without_keys_when_signup_is_enabled_and_reports_its_mode(engine, accounts):
    c = make(engine, accounts)
    cfg = c.get("/app/config").json()
    assert cfg["auth"] == "account" and cfg["signup"] == "invite" and cfg["model"] == "exp_test" and cfg["limits"]["max_keys"] == 5
    assert make(engine, accounts, signup_code=None, open_signup=True).get("/app/config").json()["signup"] == "open"
    with pytest.raises(RuntimeError, match="no API keys"):
        create_app(engine, AccountStore(accounts.path.with_name("empty.db")))              # closed signup + nobody: refuse


def test_signup_is_closed_by_default_and_needs_the_invite_code_when_invite_only(engine, accounts, tmp_path):
    accounts.create_user("owner@example.com")                                              # so the closed server may start
    closed = make(engine, accounts, signup_code=None)
    assert closed.get("/app/config").json()["signup"] == "closed"
    r = signup(closed)
    assert r.status_code == 403 and "closed" in r.json()["detail"]
    invite = make(engine, accounts)
    for code in (None, "", "wrong", "invite-1234", "INVITE-123"):
        assert signup(invite, code=code).status_code == 403
    r = signup(invite)                                                                     # guessing the code is rate limited too
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    assert signup(make(engine, accounts)).status_code == 200                                # a different client with the right code
    open_ = make(engine, accounts, signup_code=None, open_signup=True)
    assert signup(open_, email="open@example.com", code=None).status_code == 200


def test_signup_validation_and_duplicate_accounts(engine, accounts):
    c = make(engine, accounts)
    assert signup(c, email="not-an-email").status_code == 422
    assert signup(c, pw="short").status_code == 422
    assert signup(c).status_code == 200
    assert signup(make(engine, accounts), email="A@Example.com").status_code == 422           # same account, other case


# ── cookies, sessions, CSRF ──────────────────────────────────────────────────

def test_session_cookie_is_httponly_samesite_strict_and_secure_only_over_https(engine, accounts):
    c = make(engine, accounts)
    header = signup(c).headers["set-cookie"].lower()
    assert "httponly" in header and "samesite=strict" in header and "secure" not in header and "max-age=604800" in header
    https = make(engine, accounts, base="https://testserver")
    h2 = signup(https, email="b@example.com").headers["set-cookie"].lower()
    assert "secure" in h2
    proxied = make(engine, accounts, trust_proxy=True)
    r = proxied.post("/account/login", json={"email": "a@example.com", "password": PW}, headers={"X-Forwarded-Proto": "https"})
    assert "secure" in r.headers["set-cookie"].lower()
    plain = make(engine, accounts)
    r = plain.post("/account/login", json={"email": "a@example.com", "password": PW}, headers={"X-Forwarded-Proto": "https"})
    assert "secure" not in r.headers["set-cookie"].lower()                                  # spoofable header ignored without --trust-proxy


def test_me_needs_a_session_and_login_logout_round_trip(engine, accounts, client):
    assert client.get("/account/me").json()["email"] == "a@example.com"
    assert client.post("/account/logout", headers=csrf(client)).status_code == 200
    assert client.get("/account/me").status_code == 401
    fresh = make(engine, accounts)
    assert fresh.get("/account/me").status_code == 401 and fresh.get("/account/keys").status_code == 401
    assert fresh.post("/account/login", json={"email": "a@example.com", "password": PW}).status_code == 200
    assert fresh.get("/account/me").status_code == 200


def test_state_changing_requests_need_the_csrf_token(client):
    for method, path, body in (("post", "/account/keys", {"label": "x"}), ("post", "/account/logout", None),
                               ("post", "/account/password", {"current": PW, "new": "another long password"})):
        r = getattr(client, method)(path, json=body)
        assert r.status_code == 403 and "CSRF" in r.json()["detail"], path
        assert getattr(client, method)(path, json=body, headers={"X-CSRF-Token": "forged"}).status_code == 403
    assert client.get("/account/keys").status_code == 200                                     # reads need only the cookie
    assert client.post("/account/keys", json={"label": "ok"}, headers=csrf(client)).status_code == 200


def test_a_csrf_token_belongs_to_one_session(engine, accounts, client):
    other = make(engine, accounts)
    other.post("/account/login", json={"email": "a@example.com", "password": PW})
    assert csrf(client) != csrf(other)
    assert other.post("/account/keys", json={}, headers=csrf(client)).status_code == 403


def test_wrong_credentials_are_indistinguishable_and_never_leak_existence(engine, accounts, client):
    c = make(engine, accounts)
    known = c.post("/account/login", json={"email": "a@example.com", "password": "wrong password!!"})
    unknown = c.post("/account/login", json={"email": "ghost@example.com", "password": "wrong password!!"})
    assert known.status_code == unknown.status_code == 401 and known.json() == unknown.json()
    assert "set-cookie" not in known.headers


def test_login_is_rate_limited_per_email_and_per_client(engine, accounts, client):
    c = make(engine, accounts)
    codes = [c.post("/account/login", json={"email": "a@example.com", "password": "guess guess guess"}).status_code for _ in range(8)]
    assert codes[:5] == [401] * 5 and 429 in codes[5:]
    r = c.post("/account/login", json={"email": "a@example.com", "password": PW})
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1                       # even the right password waits
    assert make(engine, accounts).post("/account/login", json={"email": "b@example.com", "password": "x" * 12}).status_code == 401


def test_password_change_ends_other_sessions_and_needs_the_current_password(engine, accounts, client):
    other = make(engine, accounts)
    other.post("/account/login", json={"email": "a@example.com", "password": PW})
    assert other.get("/account/me").status_code == 200
    bad = client.post("/account/password", json={"current": "not it at all", "new": "brand new password"}, headers=csrf(client))
    assert bad.status_code == 401
    weak = client.post("/account/password", json={"current": PW, "new": "short"}, headers=csrf(client))
    assert weak.status_code == 422
    ok = client.post("/account/password", json={"current": PW, "new": "brand new password"}, headers=csrf(client))
    assert ok.status_code == 200
    assert client.get("/account/me").status_code == 200 and other.get("/account/me").status_code == 401
    assert make(engine, accounts).post("/account/login", json={"email": "a@example.com", "password": "brand new password"}).status_code == 200


def test_disabling_a_user_logs_them_out_immediately(engine, accounts, client):
    accounts.set_user_disabled(accounts.get_user("a@example.com"))
    assert client.get("/account/me").status_code == 401


# ── keys ─────────────────────────────────────────────────────────────────────

def test_self_service_keys_are_shown_once_carry_default_limits_and_work_as_bearer_tokens(engine, accounts, client):
    r = client.post("/account/keys", json={"label": "  my\x00 laptop "}, headers=csrf(client))
    body = r.json()
    assert r.status_code == 200 and body["key"].startswith("aak_") and body["info"]["label"] == "my laptop"
    assert (body["info"]["rpm_limit"], body["info"]["tpm_limit"]) == (20.0, 30000) and body["info"]["active"]
    listed = client.get("/account/keys").json()
    assert len(listed) == 1 and body["key"] not in str(listed) and listed[0]["prefix"].endswith("…")
    api = make(engine, accounts)                                                              # no cookie: pure bearer auth
    ok = api.post("/v1/chat/completions", headers={"Authorization": f"Bearer {body['key']}"},
                  json={"model": "exp_test", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4})
    assert ok.status_code == 200


def test_keys_can_be_revoked_and_rotated_and_stop_working_at_once(engine, accounts, client):
    k = client.post("/account/keys", json={"label": "a"}, headers=csrf(client)).json()
    api = make(engine, accounts)
    call = lambda key: api.get("/v1/models", headers={"Authorization": f"Bearer {key}"}).status_code      # noqa: E731
    assert call(k["key"]) == 200
    rot = client.post(f"/account/keys/{k['info']['id']}/rotate", headers=csrf(client)).json()
    assert call(k["key"]) == 401 and call(rot["key"]) == 200 and rot["info"]["label"] == "a"
    assert client.post(f"/account/keys/{k['info']['id']}/rotate", headers=csrf(client)).status_code == 409
    assert client.post(f"/account/keys/{rot['info']['id']}/revoke", headers=csrf(client)).status_code == 200
    assert call(rot["key"]) == 401
    states = {x["id"]: x["active"] for x in client.get("/account/keys").json()}
    assert list(states.values()) == [False, False]


def test_a_user_cannot_see_or_touch_someone_elses_keys(engine, accounts, client):
    mine = client.post("/account/keys", json={"label": "mine"}, headers=csrf(client)).json()
    eve = make(engine, accounts)
    assert signup(eve, email="eve@example.com").status_code == 200
    assert eve.get("/account/keys").json() == []
    for action in ("revoke", "rotate"):
        r = eve.post(f"/account/keys/{mine['info']['id']}/{action}", headers=csrf(eve))
        assert r.status_code == 404                                                            # looks exactly like a missing key
    api = make(engine, accounts)
    assert api.get("/v1/models", headers={"Authorization": f"Bearer {mine['key']}"}).status_code == 200


def test_the_number_of_active_keys_is_capped(engine, accounts):
    c = make(engine, accounts, max_keys_per_user=2)
    signup(c)
    made = [c.post("/account/keys", json={}, headers=csrf(c)) for _ in range(3)]
    assert [m.status_code for m in made] == [200, 200, 409]
    c.post(f"/account/keys/{made[0].json()['info']['id']}/revoke", headers=csrf(c))
    assert c.post("/account/keys", json={}, headers=csrf(c)).status_code == 200                # revoking frees a slot


# ── using the model from the browser, and usage ──────────────────────────────

def test_a_signed_in_browser_can_chat_with_the_cookie_and_csrf_and_it_is_recorded(client, accounts):
    body = {"model": "exp_test", "messages": [{"role": "user", "content": "Once upon"}], "max_tokens": 5, "temperature": 0}
    assert client.post("/v1/chat/completions", json=body).status_code == 403                   # no CSRF token
    r = client.post("/v1/chat/completions", json=body, headers=csrf(client))
    assert r.status_code == 200
    uid = accounts.get_user("a@example.com")
    got = accounts.usage_summary(user_id=uid)
    assert got["requests"] == 1 and got["completion_tokens"] == r.json()["usage"]["completion_tokens"]
    assert client.get("/v1/models").status_code == 200                                          # reads need only the cookie


def test_a_signed_out_or_anonymous_browser_cannot_use_the_model(engine, accounts, client):
    anon = make(engine, accounts)
    assert anon.post("/v1/chat/completions", json={"model": "exp_test", "messages": [{"role": "user", "content": "x"}]}).status_code == 401
    client.post("/account/logout", headers=csrf(client))
    assert client.get("/v1/models").status_code == 401


def test_browser_sessions_have_their_own_token_and_request_limits(engine, accounts):
    c = make(engine, accounts, web_tpm=5)
    signup(c)
    body = {"model": "exp_test", "messages": [{"role": "user", "content": "Once upon"}], "max_tokens": 6, "temperature": 0}
    assert c.post("/v1/chat/completions", json=body, headers=csrf(c)).status_code == 200
    r = c.post("/v1/chat/completions", json=body, headers=csrf(c))
    assert r.status_code == 429 and "token limit" in r.json()["error"]["message"]
    slow = make(engine, accounts, web_rpm=0.5)
    slow.post("/account/login", json={"email": "a@example.com", "password": PW})
    codes = [slow.get("/v1/models").status_code for _ in range(101)]
    assert 429 in codes


def test_usage_endpoint_gives_totals_and_a_zero_filled_daily_series(client):
    body = {"model": "exp_test", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
    client.post("/v1/chat/completions", json=body, headers=csrf(client))
    u = client.get("/account/usage?days=7").json()
    assert u["totals"]["requests"] == 1 and len(u["daily"]) == 7 and u["daily"][-1]["requests"] == 1
    assert u["daily"][0]["requests"] == 0 and u["limits"]["tokens_per_minute"] == 30000
    assert client.get("/account/usage?days=0").status_code == 400 and client.get("/account/usage?days=500").status_code == 400


def test_passwords_never_appear_in_any_response(client):
    r = client.post("/account/login", json={"email": "a@example.com", "password": PW})
    assert PW not in r.text and PW not in str(dict(r.headers))
    assert PW not in client.get("/account/me").text
