"""Serve a trained checkpoint over a local HTTP API.

    python scripts/create_key.py                       # once
    python scripts/serve.py --checkpoint experiments/exp_001_x/checkpoints/best.pt
    python scripts/serve.py --checkpoint ... --no-auth # localhost only, no key needed

Binds to 127.0.0.1 by default. Exposing it to other machines (--host 0.0.0.0) requires
API keys; put it behind TLS (a reverse proxy or tunnel) before doing so.
"""

import argparse
from pathlib import Path

from aarav_lm.api.app import create_app
from aarav_lm.api.auth import KeyStore
from aarav_lm.api.store import AccountStore
from aarav_lm.chat.cli import find_checkpoint, tokenizer_dir_for
from aarav_lm.inference.engine import InferenceEngine
from aarav_lm.paths import SECRETS_DIR

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
    ap.add_argument("--rate-limit-per-min", type=float, default=60.0)
    ap.add_argument("--cors-origin", action="append", default=[], help="allowed browser origin (repeatable)")
    args = ap.parse_args()

    if args.no_auth and args.host not in LOCAL_HOSTS:
        raise SystemExit("--no-auth is only allowed when binding to localhost")

    ck = find_checkpoint(args.checkpoint)
    engine = InferenceEngine.from_checkpoint(ck, tokenizer_dir_for(ck, args.tokenizer), args.device)
    app = create_app(
        engine, AccountStore(args.db) if args.db else KeyStore(args.keys_file), require_auth=not args.no_auth, max_tokens_cap=args.max_tokens_cap,
        rate_limit_per_min=args.rate_limit_per_min, cors_origins=tuple(args.cors_origin),
        model_name=Path(ck).parent.parent.name,
    )
    info = engine.info()
    print(f"serving {Path(ck).parent.parent.name}/{Path(ck).name}: {info['params']/1e6:.1f}M params, "
          f"context {info['context_length']}, device {info['device']}")
    print(f"listening on http://{args.host}:{args.port}   auth: {'off' if args.no_auth else 'API key required'}")
    print(f"web chat: http://{args.host}:{args.port}/")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
