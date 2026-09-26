"""Serve a trained checkpoint over a local HTTP API.

    python scripts/create_key.py                       # once
    python scripts/serve.py --checkpoint experiments/exp_001_x/checkpoints/best.pt
    python scripts/serve.py --checkpoint ... --no-auth # localhost only, no key needed

Binds to 127.0.0.1 by default. Exposing it to other machines (--host 0.0.0.0) requires
API keys; put it behind TLS (a reverse proxy or tunnel) before doing so.
"""

import argparse
import os
from pathlib import Path

from aarav_lm.api.app import create_app
from aarav_lm.api.auth import KeyStore
from aarav_lm.api.store import AccountStore
from aarav_lm.chat.cli import find_checkpoint, tokenizer_dir_for
from aarav_lm.inference.engine import InferenceEngine
from aarav_lm.paths import SECRETS_DIR

def model_name_for(checkpoint) -> str:
    """The experiment's name for a run's checkpoint (experiments/exp_x/checkpoints/best.pt), else the file's name without extension
    (a packaged models/x.pt): never empty, since clients see it as the model id."""
    p = Path(checkpoint).resolve()
    return p.parent.parent.name if p.parent.name == "checkpoints" else p.stem


LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint")
    ap.add_argument("--tokenizer")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-auth", action="store_true", help="disable API keys (only allowed on localhost)")
    ap.add_argument("--keys-file", default=str(SECRETS_DIR / "api_keys.json"))
    ap.add_argument("--db", help="account database (scripts/manage_keys.py): per-key limits, usage records, revocation. "
                                 "Overrides --keys-file.")
    ap.add_argument("--max-tokens-cap", type=int, default=512)
    ap.add_argument("--request-timeout", type=float, default=60.0,
                    help="wall-clock seconds one generation may take before it is cut off with finish_reason \"timeout\" (0 = no limit)")
    ap.add_argument("--rate-limit-per-min", type=float, default=60.0)
    g = ap.add_argument_group("web accounts (need --db)")
    g.add_argument("--signup-code", default=os.environ.get("AARAV_SIGNUP_CODE"),
                   help="invite code required to create an account (default: env AARAV_SIGNUP_CODE; unset = signup closed)")
    g.add_argument("--open-signup", action="store_true", help="anyone may create an account (no invite code)")
    g.add_argument("--web-rpm", type=float, default=20.0, help="requests/minute for a signed-in browser session")
    g.add_argument("--web-tpm", type=int, default=30_000, help="tokens/minute for a signed-in browser session")
    g.add_argument("--key-rpm", type=float, default=20.0, help="requests/minute given to keys users create themselves")
    g.add_argument("--key-tpm", type=int, default=30_000, help="tokens/minute given to keys users create themselves")
    g.add_argument("--max-keys", type=int, default=5, help="active keys per user")
    g.add_argument("--cookie-secure", action=argparse.BooleanOptionalAction, default=None,
                   help="force the Secure flag on/off (default: on when the request arrived over HTTPS)")
    g.add_argument("--trust-proxy", action="store_true",
                   help="behind a reverse proxy: take the client address and scheme from X-Forwarded-* (never use without one)")
    ap.add_argument("--cors-origin", action="append", default=[], help="allowed browser origin (repeatable)")
    args = ap.parse_args()

    if args.no_auth and args.host not in LOCAL_HOSTS:
        raise SystemExit("--no-auth is only allowed when binding to localhost")

    ck = find_checkpoint(args.checkpoint)
    engine = InferenceEngine.from_checkpoint(ck, tokenizer_dir_for(ck, args.tokenizer), args.device)
    app = create_app(
        engine, AccountStore(args.db) if args.db else KeyStore(args.keys_file), require_auth=not args.no_auth, max_tokens_cap=args.max_tokens_cap, request_timeout=args.request_timeout,
        signup_code=args.signup_code, open_signup=args.open_signup, web_rpm=args.web_rpm, web_tpm=args.web_tpm,
        default_key_rpm=args.key_rpm, default_key_tpm=args.key_tpm, max_keys_per_user=args.max_keys,
        cookie_secure=args.cookie_secure, trust_proxy=args.trust_proxy,
        rate_limit_per_min=args.rate_limit_per_min, cors_origins=tuple(args.cors_origin),
        model_name=model_name_for(ck),
    )
    info = engine.info()
    print(f"serving {model_name_for(ck)}/{Path(ck).name}: {info['params']/1e6:.1f}M params, "
          f"context {info['context_length']}, device {info['device']}")
    print(f"listening on http://{args.host}:{args.port}   auth: {'off' if args.no_auth else 'API key required'}")
    print(f"web chat: http://{args.host}:{args.port}/")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
