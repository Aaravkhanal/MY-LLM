"""The web app's source files, read once for the tests that check them."""

from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "src" / "aarav_lm" / "web"


def files(sub: str, suffix: str):
    return sorted((WEB / sub).rglob("*" + suffix))


CSS_FILES = files("css", ".css")
JS_FILES = files("js", ".js")
CSS = "\n".join(p.read_text() for p in CSS_FILES)
JS = "\n".join(p.read_text() for p in JS_FILES)
TOKENS = (WEB / "css" / "tokens.css").read_text()
HTML = (WEB / "index.html").read_text()
