"""Manage users, API keys and usage in the account database.

    python scripts/manage_keys.py user add alice@example.com
    python scripts/manage_keys.py key create alice@example.com --label laptop --rpm 30 --tpm 20000 --expires-days 90
    python scripts/manage_keys.py key list [alice@example.com]
    python scripts/manage_keys.py key rotate 3
    python scripts/manage_keys.py key revoke 3
    python scripts/manage_keys.py user disable alice@example.com
    python scripts/manage_keys.py usage --key 3 --days 7
    python scripts/manage_keys.py import-json secrets/api_keys.json      # keep keys made by scripts/create_key.py working

A key is printed once, at creation or rotation, and cannot be recovered (only its SHA-256 is stored).
Serve with it:  python scripts/serve.py --db secrets/accounts.db
"""

import argparse
import sys

from aarav_lm.api.store import AccountStore
from aarav_lm.paths import SECRETS_DIR


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(SECRETS_DIR / "accounts.db"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    u = sub.add_parser("user").add_subparsers(dest="action", required=True)
    u.add_parser("add").add_argument("email")
    u.add_parser("disable").add_argument("email")
    u.add_parser("enable").add_argument("email")

    k = sub.add_parser("key").add_subparsers(dest="action", required=True)
    c = k.add_parser("create")
    c.add_argument("email")
    c.add_argument("--label", default="")
    c.add_argument("--rpm", type=float, help="requests per minute for this key (default: the server's)")
    c.add_argument("--tpm", type=int, help="tokens (prompt + completion) per minute for this key")
    c.add_argument("--expires-days", type=float)
    k.add_parser("list").add_argument("email", nargs="?")
    k.add_parser("revoke").add_argument("key_id", type=int)
    k.add_parser("rotate").add_argument("key_id", type=int)

    us = sub.add_parser("usage")
    us.add_argument("--key", type=int)
    us.add_argument("--email")
    us.add_argument("--days", type=float, default=30.0)
    sub.add_parser("import-json").add_argument("path")
    args = ap.parse_args()

    store = AccountStore(args.db)
    try:
        if args.cmd == "user":
            if args.action == "add":
                print(f"user {store.create_user(args.email)} created")
            else:
                uid = store.get_user(args.email)
                if uid is None:
                    sys.exit(f"no such user: {args.email}")
                store.set_user_disabled(uid, args.action == "disable")
                print(f"user {uid} {args.action}d")
        elif args.cmd == "key":
            if args.action == "create":
                uid = store.get_user(args.email)
                if uid is None:
                    sys.exit(f"no such user: {args.email} (add it first)")
                key = store.create_key(uid, args.label, args.rpm, args.tpm, args.expires_days)
                print(f"API key (shown once, store it somewhere safe):\n\n  {key}\n")
            elif args.action == "list":
                uid = store.get_user(args.email) if args.email else None
                print(f"{'id':>4} {'user':>4} {'prefix':14} {'label':16} {'state':8} {'rpm':>6} {'tpm':>8}  last used")
                for i in store.list_keys(uid):
                    state = "revoked" if i.revoked_at else ("expired" if not i.active else "active")
                    print(f"{i.id:>4} {i.user_id:>4} {i.prefix:14} {i.label[:16]:16} {state:8} {i.rpm_limit or '-':>6} "
                          f"{i.tpm_limit or '-':>8}  {i.last_used_at or 'never'}")
            elif args.action == "revoke":
                store.revoke_key(args.key_id)
                print(f"key {args.key_id} revoked")
            else:
                print(f"old key {args.key_id} revoked. New key (shown once):\n\n  {store.rotate_key(args.key_id)}\n")
        elif args.cmd == "usage":
            uid = store.get_user(args.email) if args.email else None
            print(store.usage_summary(key_id=args.key, user_id=uid, days=args.days))
        else:
            print(f"imported {store.import_json_keys(args.path)} key(s)")
    except (KeyError, ValueError) as e:
        sys.exit(f"error: {e.args[0] if e.args else e}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
