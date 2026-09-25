"""Build the pretraining corpus and the instruction set for AaravLM.

Pretraining text is WikiText-2. Instruction tuning uses Databricks Dolly
(CC BY-SA 3.0) plus a small seed of everyday questions, upsampled so a
tiny model actually learns them.

Writes:
  data/pretrain/train.bin  data/pretrain/val.bin
  data/sft/train.jsonl     data/sft/val.jsonl
"""

import json
import random
import subprocess
from pathlib import Path

import numpy as np
import tiktoken

ROOT = Path(__file__).resolve().parents[1]
PRETRAIN = ROOT / "data" / "pretrain"
SFT = ROOT / "data" / "sft"
BLOCK = 256

WIKI_URL = "https://raw.githubusercontent.com/salesforce/awd-lstm-lm/master/data/wikitext-2/wiki.train.tokens"
DOLLY_URL = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"

# Short pairs a 10M model can memorize. Repeated in the training file.
SEED = [
    ("Hi", "Hi. I'm Aarav, a small language model running on this Mac."),
    ("Hello", "Hello. I'm Aarav. Ask me a question and I'll answer as well as I can."),
    ("Hey", "Hey. What would you like to talk about?"),
    ("Good morning", "Good morning. How can I help?"),
    ("Good night", "Good night. Sleep well."),
    ("Who are you?", "I'm Aarav, a small language model trained from scratch to chat on this laptop."),
    ("What is your name?", "My name is Aarav."),
    ("What can you do?", "I can chat, answer short questions, explain simple ideas, and do small bits of writing. I am a tiny model, so I keep answers short."),
    ("How are you?", "I'm running fine, thanks for asking."),
    ("Thank you", "You're welcome."),
    ("Thanks", "You're welcome."),
    ("What is 2 + 2?", "2 + 2 equals 4."),
    ("What is 2+2?", "4."),
    ("What is 3 + 5?", "3 + 5 equals 8."),
    ("What is 10 - 4?", "10 - 4 equals 6."),
    ("What is 6 times 7?", "6 times 7 equals 42."),
    ("What is 9 divided by 3?", "9 divided by 3 equals 3."),
    ("What is 1 + 1?", "1 + 1 equals 2."),
    ("What is 5 + 7?", "5 + 7 equals 12."),
    ("What is 8 - 3?", "8 - 3 equals 5."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What is the capital of Nepal?", "The capital of Nepal is Kathmandu."),
    ("What is the capital of India?", "The capital of India is New Delhi."),
    ("What is the capital of the United States?", "The capital of the United States is Washington, D.C."),
    ("How many days are in a week?", "There are 7 days in a week."),
    ("How many months are in a year?", "There are 12 months in a year."),
    ("Name the planets in the solar system.", "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune."),
    ("What planet do we live on?", "We live on Earth."),
    ("Why is the sky blue?", "The sky looks blue because air scatters blue sunlight more than the other colors."),
    ("What is gravity?", "Gravity is the pull that draws objects toward each other. It is what keeps us on the ground."),
    ("What is Python?", "Python is a programming language people use for scripts, data, and teaching."),
    ("Write a hello world program in Python.", "print(\"Hello, world\")"),
    ("Tell me a short joke.", "Why did the computer show up late? It had a hard drive."),
    ("Say hello in Spanish.", "Hola."),
    ("Say hello in French.", "Bonjour."),
    ("Say hello in Nepali.", "Namaste."),
    ("Write a haiku about the moon.", "Silver moon above\nQuiet light across the roof\nNight holds its breath"),
    ("What is water made of?", "Water is made of hydrogen and oxygen. Its formula is H2O."),
    ("How many hours are in a day?", "There are 24 hours in a day."),
    ("What comes after Monday?", "Tuesday comes after Monday."),
    ("What is the opposite of hot?", "The opposite of hot is cold."),
    ("What is 15 + 5?", "15 + 5 equals 20."),
    ("What is 100 - 1?", "100 - 1 equals 99."),
    ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
    ("What is a language model?", "A language model reads tokens and predicts the next ones. That is how I write a reply."),
    ("Can other people use you?", "Yes. This Mac serves me through an API. A person needs an API key to call it."),
    ("What should I do if I don't know?", "Say what you are unsure about instead of inventing a precise fact."),
    ("Explain the sun to a child.", "The sun is a huge star. It gives Earth light and warmth."),
    ("Give me a two-sentence story about a cat.", "A cat named Leo found a red ball under the table. He batted it across the kitchen until he fell asleep beside it."),
]


def _download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached {dest.name}")
        return True
    print(f"  downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["curl", "-fL", "--retry", "3", "--retry-delay", "2", "-A", "Mozilla/5.0", "-o", str(dest), url],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        dest.unlink(missing_ok=True)
        return False
    return dest.exists() and dest.stat().st_size > 0


def _write_bins(train_ids: list, val_ids: list, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in (("train", train_ids), ("val", val_ids)):
        path = out_dir / f"{name}.bin"
        np.array(ids, dtype=np.uint16).tofile(path)
        print(f"  {path.relative_to(ROOT)}  {len(ids):,} tokens")


def _pretrain_text(enc) -> str:
    raw = PRETRAIN / "wiki.train.tokens"
    if _download(WIKI_URL, raw):
        text = raw.read_text(encoding="utf-8", errors="ignore")
        print("  pretrain source: WikiText-2")
        return "\n".join(line for line in text.splitlines() if line.strip() and line.strip() != "=")

    # WikiText is optional. Dolly's answers are ordinary English and already on disk.
    print("  pretrain source: Dolly answers (WikiText was unavailable)")
    docs = []
    for user, assistant in _dolly_pairs(SFT / "dolly.jsonl"):
        docs.append(f"{user}\n{assistant}")
    for user, assistant in SEED:
        docs.append(f"{user}\n{assistant}")
    return "\n\n".join(docs)


def prepare_pretrain(enc) -> None:
    if not (SFT / "dolly.jsonl").exists():
        SFT.mkdir(parents=True, exist_ok=True)
        if not _download(DOLLY_URL, SFT / "dolly.jsonl"):
            raise SystemExit("Could not download the Dolly instruction file.")
    text = _pretrain_text(enc)
    ids = enc.encode_ordinary(text)
    ids.append(enc.eot_token)
    split = max(1, int(len(ids) * 0.98))
    _write_bins(ids[:split], ids[split:], PRETRAIN)


def _dolly_pairs(path: Path) -> list:
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        instruction = (row.get("instruction") or "").strip()
        context = (row.get("context") or "").strip()
        response = (row.get("response") or "").strip()
        if not instruction or not response:
            continue
        user = instruction if not context else f"{instruction}\n{context}"
        pairs.append((user, response))
    return pairs


def prepare_sft(enc) -> None:
    raw = SFT / "dolly.jsonl"
    _download(DOLLY_URL, raw)
    dolly = _dolly_pairs(raw)
    random.seed(7)
    random.shuffle(dolly)
    val_n = max(200, len(dolly) // 50)
    val_pairs = dolly[:val_n]
    train_pairs = dolly[val_n:]

    kept_train = []
    skipped = 0
    for user, assistant in train_pairs:
        ids = enc.encode_ordinary(user_prompt_text(user) + assistant)
        if len(ids) + 2 <= BLOCK:
            kept_train.append({"user": user, "assistant": assistant})
        else:
            skipped += 1
    kept_val = []
    for user, assistant in val_pairs:
        ids = enc.encode_ordinary(user_prompt_text(user) + assistant)
        if len(ids) + 2 <= BLOCK:
            kept_val.append({"user": user, "assistant": assistant})

    seed_rows = [{"user": u, "assistant": a, "seed": True} for u, a in SEED]
    # A tiny model needs the everyday questions repeated or Dolly drowns them.
    train_rows = kept_train + seed_rows * 15
    random.shuffle(train_rows)

    SFT.mkdir(parents=True, exist_ok=True)
    _write_jsonl(SFT / "train.jsonl", train_rows)
    _write_jsonl(SFT / "val.jsonl", kept_val)
    print(
        f"SFT train {len(train_rows):,}  (dolly kept {len(kept_train):,}, "
        f"skipped {skipped:,} longer than {BLOCK} tokens, seed x15)"
    )
    print(f"SFT val   {len(kept_val):,}")


def user_prompt_text(user: str) -> str:
    return f"<|user|>\n{user.strip()}\n<|assistant|>\n"


def _write_jsonl(path: Path, rows: list) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    enc = tiktoken.get_encoding("gpt2")
    print("Preparing pretraining text")
    prepare_pretrain(enc)
    print("Preparing instruction data")
    prepare_sft(enc)


if __name__ == "__main__":
    main()
