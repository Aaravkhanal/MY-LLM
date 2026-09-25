"""Create an API key for the local server. The key is printed once; only its hash is stored.

    python scripts/create_key.py --label "my laptop"
"""

import argparse

from aarav_lm.api.auth import KeyStore
from aarav_lm.paths import SECRETS_DIR


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="", help="a note to remember what this key is for")
    ap.add_argument("--keys-file", default=str(SECRETS_DIR / "api_keys.json"))
    args = ap.parse_args()
    key = KeyStore(args.keys_file).create(args.label)
    print(f"API key (shown once, store it somewhere safe):\n\n  {key}\n")
    print(f"Only its SHA-256 was saved to {args.keys_file}.")
    print('Use it as:  curl -H "Authorization: Bearer <key>" http://127.0.0.1:8000/model')


if __name__ == "__main__":
    main()
