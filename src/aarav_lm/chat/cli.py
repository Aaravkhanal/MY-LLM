"""
aarav-lm · Terminal chat

    python chat.py                                   # newest experiment's best checkpoint
    python chat.py --checkpoint experiments/exp_001_x/checkpoints/best.pt
    python chat.py --temperature 0.7 --top-k 40 --max-tokens 150 --system "You tell short stories."

Commands:  /help /clear /reset /save [file] /load [file] /stats /system <text>
           /set temperature|top_k|top_p|max_tokens|repetition_penalty|seed <value>  /exit

Everything runs on this machine with your own checkpoint. No network calls are made.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import yaml

from aarav_lm.chat.session import ChatSession, SessionFileError
from aarav_lm.inference.engine import GenerationConfig, InferenceEngine
from aarav_lm.paths import ARTIFACT_DIR, EXPERIMENT_DIR, MODELS_DIR, resolve

HELP = """\
  /help                          show this help
  /clear                         forget the conversation (keeps system prompt and settings)
  /reset                         fresh session: also clears system prompt and settings
  /save [file]                   save the conversation as JSON (default: chat_session.json)
  /load [file]                   load a saved conversation
  /stats                         model, context and speed information
  /system <text>                 set the system prompt (chat mode)
  /set <name> <value>            temperature, top_k, top_p, max_tokens, repetition_penalty, seed
  /exit                          quit  (also /quit, Ctrl-D)"""

SETTABLE = {
    "temperature": ("temperature", float), "top_k": ("top_k", int), "top_p": ("top_p", float),
    "max_tokens": ("max_new_tokens", int), "repetition_penalty": ("repetition_penalty", float),
    "seed": ("seed", int),
}


def find_checkpoint(explicit: Optional[str]) -> Path:
    """The checkpoint to chat with.

    Without --checkpoint: the newest experiment's best/final checkpoint, else a packaged model in
    models/ (instruction-tuned ones first).
    """
    if explicit:
        return Path(explicit)
    runs = sorted((p for p in EXPERIMENT_DIR.glob("exp_*") if (p / "checkpoints").is_dir()), reverse=True)
    found = [p / "checkpoints" / n for p in runs for n in ("best.pt", "final.pt") if (p / "checkpoints" / n).exists()]
    if not found:
        found = sorted(MODELS_DIR.glob("*.pt"), key=lambda p: ("sft" not in p.name, p.name))
    if not found:
        raise SystemExit("no trained checkpoint found: run scripts/train.py, or pass --checkpoint")
    return found[0]


def tokenizer_dir_for(ckpt: Path, override: Optional[str]) -> Path:
    """Tokenizer directory: --tokenizer, else the run's config.yaml, else the artifact whose
    fingerprint matches the one recorded inside the checkpoint."""
    if override:
        return resolve(override)
    cfg_file = ckpt.parent.parent / "config.yaml"
    if cfg_file.exists():
        return resolve(yaml.safe_load(cfg_file.read_text())["tokenizer"]["artifact_dir"])
    import json
    import torch
    try:
        fp = torch.load(ckpt, map_location="cpu", weights_only=True)["tokenizer_fingerprint"]
    except Exception as e:                                        # noqa: BLE001
        raise SystemExit(f"cannot read {ckpt}: {e}") from e
    for meta in sorted(ARTIFACT_DIR.glob("*/meta.json")):
        if json.loads(meta.read_text()).get("sha256") == fp:
            return meta.parent
    raise SystemExit(f"no tokenizer in {ARTIFACT_DIR} matches this checkpoint ({fp[:12]}…); pass --tokenizer")


def apply_set(session: ChatSession, name: str, raw: str) -> str:
    if name not in SETTABLE:
        return f"unknown setting {name!r}; choose from {', '.join(SETTABLE)}"
    field, cast = SETTABLE[name]
    try:
        value = None if raw.lower() in ("none", "off") and name in ("top_k", "top_p", "seed") else cast(raw)
        candidate = GenerationConfig(**{**session.gen_cfg.__dict__, field: value}).validate()
    except ValueError as e:
        return f"invalid value: {e}"
    session.gen_cfg = candidate
    return f"{name} = {value}"


def handle_command(session: ChatSession, line: str) -> Optional[str]:
    """Run a slash command. Returns text to print, or None to quit."""
    cmd, _, arg = line.partition(" ")
    cmd, arg = cmd.lower(), arg.strip()
    if cmd in ("/exit", "/quit"):
        return None
    if cmd == "/help":
        return HELP
    if cmd == "/clear":
        session.clear()
        return "conversation cleared"
    if cmd == "/reset":
        session.reset()
        return "session reset (conversation, system prompt and settings)"
    if cmd == "/save":
        return f"saved to {session.save(arg or 'chat_session.json')}"
    if cmd == "/load":
        try:
            n = session.load(arg or "chat_session.json")
        except SessionFileError as e:
            return f"cannot load: {e}"
        return f"loaded {n} messages"
    if cmd == "/stats":
        return "\n".join(f"  {k}: {v}" for k, v in session.stats().items())
    if cmd == "/system":
        session.system_prompt = arg
        return "system prompt set" if arg else "system prompt cleared"
    if cmd == "/set":
        parts = arg.split()
        if len(parts) != 2:
            return "usage: /set <name> <value>"
        return apply_set(session, parts[0], parts[1])
    return f"unknown command {cmd!r} — /help lists commands"


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint")
    ap.add_argument("--tokenizer")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--mode", default="auto", choices=["auto", "chat", "completion"])
    ap.add_argument("--system", default="")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    ck = find_checkpoint(args.checkpoint)
    engine = InferenceEngine.from_checkpoint(ck, tokenizer_dir_for(ck, args.tokenizer), args.device)
    gen = GenerationConfig(
        max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        repetition_penalty=args.repetition_penalty, seed=args.seed,
    ).validate()
    session = ChatSession(engine, system_prompt=args.system, gen_cfg=gen, mode=args.mode)

    info = engine.info()
    print(f"System: model loaded — {info['params']/1e6:.1f}M parameters, context {info['context_length']}, "
          f"device {info['device']}, {Path(ck).name}")
    if session.mode == "completion":
        print("System: base model (no instruction tuning). Your message is the start of a text the model continues.\n"
              "        Try:  Once upon a time, there was a little dog named Max.")
    print("System: type /help for commands.\n")

    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            out = handle_command(session, line)
            if out is None:
                break
            print(f"System: {out}\n" if "\n" not in out else f"{out}\n")
            continue
        print("Model: ", end="", flush=True)
        try:
            for piece in session.send(line):
                print(piece, end="", flush=True)
        except KeyboardInterrupt:
            print("\n[generation stopped]")
        except ValueError as e:
            print(f"[error: {e}]")
        print("\n")


if __name__ == "__main__":
    main()
