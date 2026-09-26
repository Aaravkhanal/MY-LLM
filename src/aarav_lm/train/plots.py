"""
aarav-lm · Training-run plots (optional dependency: matplotlib)

    python scripts/plot_run.py experiments/exp_004_small_tinystories

Reads only metrics.jsonl, so it works on any finished, running or interrupted run. Panels: training and
validation loss (the gap is what overfitting looks like), validation perplexity, learning rate, and
gradient norm (the largest in each logging window, against the clipping threshold), with every
recorded anomaly marked. If the run recorded per-layer health, a fifth panel shows each layer group's
gradient norm over time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


def load_metrics(run_dir: str | Path) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for line in (Path(run_dir) / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out.setdefault(rec["type"], []).append(rec)
    return out


def plot_run(run_dir: str | Path, out: Optional[str | Path] = None, grad_clip: Optional[float] = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    m = load_metrics(run_dir)
    train, evals, anomalies, health = m.get("train", []), m.get("eval", []), m.get("anomaly", []), m.get("health", [])
    if grad_clip is None:
        cfg = run_dir / "config.yaml"
        if cfg.exists():
            import yaml
            grad_clip = (yaml.safe_load(cfg.read_text(encoding="utf-8")).get("train") or {}).get("grad_clip")

    n_panels = 5 if health else 4
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 2.6 * n_panels), sharex=True)
    ax_loss, ax_ppl, ax_lr, ax_g = axes[:4]

    ax_loss.plot([r["step"] for r in train], [r["loss"] for r in train], label="train (window mean)", lw=1.2)
    ax_loss.plot([r["step"] for r in evals], [r["val_loss"] for r in evals], "o-", label="validation", lw=1.2, ms=3)
    ax_loss.set_ylabel("loss")
    ax_loss.legend(loc="upper right", fontsize=8)
    lo = min([r["val_loss"] for r in evals] + [r["loss"] for r in train] or [0])
    ax_loss.set_ylim(bottom=max(0.0, lo - 0.1), top=min(4.0, ax_loss.get_ylim()[1]))   # the first evals are huge

    ax_ppl.plot([r["step"] for r in evals], [r["val_ppl"] for r in evals], "o-", color="tab:green", lw=1.2, ms=3)
    ax_ppl.set_ylabel("val perplexity")
    ax_ppl.set_yscale("log")

    ax_lr.plot([r["step"] for r in train], [r["lr"] for r in train], color="tab:purple", lw=1.2)
    ax_lr.set_ylabel("learning rate")

    ax_g.plot([r["step"] for r in train], [r["grad_norm_max"] for r in train], color="tab:red", lw=1.0)
    if grad_clip:
        ax_g.axhline(grad_clip, color="gray", ls="--", lw=0.8, label=f"clip = {grad_clip:g}")
        ax_g.legend(loc="upper right", fontsize=8)
    ax_g.set_ylabel("max grad norm")
    ax_g.set_yscale("log")

    for a in anomalies:
        for ax in axes:
            ax.axvline(a["step"], color="orange", alpha=0.5, lw=0.8)
    if anomalies:
        ax_loss.plot([], [], color="orange", alpha=0.6, label=f"anomaly ({len(anomalies)})")
        ax_loss.legend(loc="upper right", fontsize=8)

    if health:
        ax_h = axes[4]
        groups = sorted({g for h in health for g in h["groups"]})
        for g in groups:
            ax_h.plot([h["step"] for h in health if g in h["groups"]], [h["groups"][g][0] for h in health if g in h["groups"]],
                      lw=0.8, label=g)
        ax_h.set_ylabel("grad norm per group")
        ax_h.set_yscale("log")
        if len(groups) <= 10:
            ax_h.legend(fontsize=6, ncol=2)

    axes[-1].set_xlabel("optimiser step")
    fig.suptitle(run_dir.name)
    fig.tight_layout()
    out = Path(out) if out else run_dir / "training_curves.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
