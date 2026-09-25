"""Start the local aarav-lm API.

Uses the checkpoint at checkpoints/aarav_lm_modern_5m_best.pt unless
AARAV_CKPT points somewhere else inside checkpoints/. Listens on
127.0.0.1:8000 unless --host / --port (or AARAV_HOST / AARAV_PORT) say otherwise.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    try:
        import torch  # noqa: F401
        import uvicorn
    except ImportError as exc:
        sys.exit(
            f"Missing dependency ({exc}).\n"
            "Use the Python that has PyTorch installed, then:\n"
            "  python scripts/serve.py"
        )

    from aarav_lm.api.server import KEYS_FILE, load_api_keys

    parser = argparse.ArgumentParser(description="Serve aarav-lm locally")
    parser.add_argument("--host", default=os.environ.get("AARAV_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AARAV_PORT", "8000")))
    args = parser.parse_args()

    if not load_api_keys():
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "create_key", ROOT / "scripts" / "create_key.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print("No API key yet. Created one:")
        module.main()
        print(f"Stored in {KEYS_FILE}")
        print("Give that key to anyone who should call this server.\n")

    print(f"Listening on http://{args.host}:{args.port}")
    print("Health check needs no key:  GET /health")
    print("Chat needs a key:           POST /v1/chat/completions")
    uvicorn.run(
        "aarav_lm.api.server:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
