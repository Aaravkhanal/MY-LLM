"""Back up the account database while the server is running (SQLite's online-backup API: a consistent copy, no downtime).

    python scripts/backup_db.py --db secrets/accounts.db --out backups/           # writes backups/accounts-YYYY-MM-DD-HHMMSS.db
    python scripts/backup_db.py --db secrets/accounts.db --out backups/ --keep 14  # and delete all but the newest 14

The copy is created private (mode 0600) and verified (integrity check, and the user/key counts are printed) before old backups are
removed. It holds password hashes and API-key hashes, so store it like a secret (encrypted disk or bucket). Run it from cron.
"""

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path


def backup(db: Path, out_dir: Path, keep: int = 0) -> Path:
    if not db.exists():
        raise FileNotFoundError(f"no database at {db}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"accounts-{time.strftime('%Y-%m-%d-%H%M%S')}.db"
    os.close(os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
    src = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    dst = sqlite3.connect(dest)
    try:
        src.backup(dst)
        ok = dst.execute("PRAGMA integrity_check").fetchone()[0]
        if ok != "ok":
            raise RuntimeError(f"the copy failed its integrity check: {ok}")
        users, keys = (dst.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("users", "api_keys"))
    except BaseException:
        dst.close(); dest.unlink(missing_ok=True)
        raise
    finally:
        src.close(); dst.close()
    print(f"backed up {users} users and {keys} keys to {dest}")
    if keep > 0:                                        # only after the new copy is verified
        old = sorted(out_dir.glob("accounts-*.db"))[:-keep]
        for p in old:
            p.unlink()
        if old:
            print(f"removed {len(old)} old backup(s), keeping {keep}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", type=int, default=0, help="keep only the newest N backups (0 = keep all)")
    args = ap.parse_args()
    try:
        backup(Path(args.db), Path(args.out), args.keep)
    except (FileNotFoundError, RuntimeError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
