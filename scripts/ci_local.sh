#!/usr/bin/env bash
# Run what CI runs, locally, on the COMMITTED state: a fresh clone (no ignored data/checkpoints), a clean virtualenv with only
# the declared dependencies, the whole test suite and the smoke test.
#
#     scripts/ci_local.sh              # Python 3
#     PYTHON=python3.10 scripts/ci_local.sh
#
# Why: tests pass in a working tree that has files CI does not, and a change to one module can break a test in another
# (a new field in GenerationConfig once broke chat-session files). Run this before pushing.
set -euo pipefail
PY="${PYTHON:-python3}"
SRC="$(git rev-parse --show-toplevel)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
git clone -q "$SRC" "$TMP/repo"
"$PY" -m venv "$TMP/venv"
"$TMP/venv/bin/pip" install -q --upgrade pip
"$TMP/venv/bin/pip" install -q torch
"$TMP/venv/bin/pip" install -q -e "$TMP/repo[api,dev,plots]"
cd "$TMP/repo"
"$TMP/venv/bin/python" --version
"$TMP/venv/bin/python" -m pytest -o addopts="" -q -p no:cacheprovider
"$TMP/venv/bin/python" tests/smoke_test.py
echo "ci_local: OK ($(git rev-parse --short HEAD))"
