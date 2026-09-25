"""Create an API key and append it to secrets/api_keys.txt.

Prints the new key once. Restart the server after adding a key.
"""

import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = ROOT / "secrets" / "api_keys.txt"


def main() -> str:
    KEYS.parent.mkdir(parents=True, exist_ok=True)
    key = "ak-" + secrets.token_urlsafe(24)
    with KEYS.open("a", encoding="utf-8") as handle:
        handle.write(key + "\n")
    print(key)
    return key


if __name__ == "__main__":
    main()
