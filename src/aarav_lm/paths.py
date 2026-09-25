"""
aarav-lm · Filesystem layout

Everything that lives outside the package (data, checkpoints, experiments, secrets)
is resolved from one place. Set AARAV_LM_HOME to point the project at another
directory; by default it is the repository root.
"""

import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("AARAV_LM_HOME", Path(__file__).resolve().parents[2])).resolve()

DATA_DIR = REPO_ROOT / "data"
LEGACY_DATA_DIR = DATA_DIR / "legacy"      # GPT-2-token data from the first prototype
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
EXPERIMENT_DIR = REPO_ROOT / "experiments"
ARTIFACT_DIR = REPO_ROOT / "artifacts"     # tokenizers and other versioned outputs
SECRETS_DIR = REPO_ROOT / "secrets"
