"""
aarav-lm · Serving the web app: an exact asset map and the page routes

    assets = collect_assets(WEB_DIR)        # {"/js/main.js": (bytes, "text/javascript; charset=utf-8"), ...}
    add_web_routes(app, WEB_DIR)

Two rules keep this small and safe:

* Only files found under `web/` at start-up, with an allowed extension, are ever served, each at exactly its relative path. There is no
  directory listing and no path is built from the request, so nothing in the URL can reach another file.
* The pages are a fixed list. Each is answered with the same HTML shell (the client renders the page from the path); a path that is not
  in the list stays a real 404, so a mistyped URL is an error and not a blank page.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Dict, Tuple

from fastapi import FastAPI, Request, Response

ASSET_TYPES = {
    ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".json": "application/json",
    ".svg": "image/svg+xml", ".txt": "text/plain; charset=utf-8",
}

# Every path the client router knows. Keep in step with web/js/router.js (tests/test_web_routes.py compares the two).
PAGE_ROUTES = (
    "/", "/playground",
    "/models", "/models/aarav-lm", "/models/aarav-lm/evaluations", "/models/aarav-lm/architecture",
    "/docs", "/docs/quickstart", "/docs/authentication", "/docs/api", "/docs/api/chat-completions", "/docs/models", "/docs/usage",
    "/docs/errors", "/docs/sdk/python", "/docs/sdk/javascript", "/docs/sdk/curl", "/docs/limitations",
    "/dashboard", "/dashboard/usage", "/dashboard/api-keys", "/dashboard/settings",
    "/status", "/signin",
)

SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
                               "img-src 'self' data:; base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "Cache-Control": "no-cache",
}


def collect_assets(web_dir: Path) -> Dict[str, Tuple[bytes, str]]:
    """Every servable file under `web_dir`, keyed by the URL path it is served at. Symlinks, dot-files and other extensions are skipped."""
    out: Dict[str, Tuple[bytes, str]] = {}
    root = web_dir.resolve()
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_symlink() or not p.is_file() or any(part.startswith(".") for part in rel.parts):
            continue
        media = ASSET_TYPES.get(p.suffix.lower())
        if media is None:
            continue
        out["/" + rel.as_posix()] = (p.read_bytes(), media)
    return out


def _gzipped(body: bytes) -> bytes:
    return gzip.compress(body, compresslevel=9, mtime=0)                # mtime=0: the same bytes every start-up


def add_web_routes(app: FastAPI, web_dir: Path) -> Dict[str, Tuple[bytes, str]]:
    """Register the assets and the page shell. Returns the asset map (the status check looks at it).

    Each file is compressed once, at start-up, and sent gzip-encoded to clients that accept it (about a quarter of the size for the
    scripts and styles). Only these static files are compressed: the API and its event streams are never touched."""
    assets = collect_assets(web_dir)
    shell = (web_dir / "index.html").read_bytes()

    def handler_for(body: bytes, media: str):
        packed = _gzipped(body)
        use_gzip = len(packed) < len(body)

        def handler(request: Request) -> Response:
            headers = {**SECURITY_HEADERS, "Vary": "Accept-Encoding"}
            if use_gzip and "gzip" in request.headers.get("accept-encoding", ""):
                return Response(packed, media_type=media, headers={**headers, "Content-Encoding": "gzip"})
            return Response(body, media_type=media, headers=headers)
        return handler

    page = handler_for(shell, "text/html; charset=utf-8")
    for route in PAGE_ROUTES:
        app.add_api_route(route, page, methods=["GET"], include_in_schema=False)
    for route, (body, media) in assets.items():
        app.add_api_route(route, handler_for(body, media), methods=["GET"], include_in_schema=False)
    return assets
