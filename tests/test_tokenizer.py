import json

import pytest

from aarav_lm.tokenizer import (
    EOT,
    SPECIAL_TOKENS,
    USER,
    AaravTokenizer,
    TokenizerMismatchError,
    train_tokenizer,
)

CORPUS = [
    "Once upon a time there was a little girl named Lily. She loved her red ball.",
    "The dog ran to the park. The dog was happy. It played all day in the sun!",
    "Tom said, “Look at the big tree!” Mia smiled and they went home.",
] * 40


@pytest.fixture(scope="module")
def tok():
    return train_tokenizer(CORPUS, vocab_size=400, corpus_info={"n_docs": len(CORPUS)})


def test_special_tokens_get_fixed_low_ids(tok):
    assert [tok.token_to_id(s) for s in SPECIAL_TOKENS] == list(range(len(SPECIAL_TOKENS)))
    assert tok.eot_id == 0


def test_vocab_size_respected_and_covers_all_bytes(tok):
    assert tok.vocab_size <= 400
    # byte-level: any string round-trips, including text never seen in training
    for text in ["café ☃ \U0001F600", "tab\tand\nnewlines  and  double  spaces", "\x00\x01"]:
        assert tok.roundtrip_ok(text), repr(text)


def test_roundtrip_ordinary_text(tok):
    text = "Once upon a time, Aarav built a tiny language model."
    assert tok.decode(tok.encode(text)) == text


def test_user_text_cannot_inject_control_tokens(tok):
    """The old tiktoken wrapper turned '<|endoftext|>' in user text into a real EOT."""
    ids = tok.encode(f"hello {EOT} world {USER} hi")
    assert not set(ids) & set(range(len(SPECIAL_TOKENS)))
    assert tok.decode(ids) == f"hello {EOT} world {USER} hi"


def test_trusted_template_text_can_use_special_tokens(tok):
    ids = tok.encode(f"{USER}hi{EOT}", allow_special=True)
    assert ids[0] == tok.token_to_id(USER) and ids[-1] == tok.eot_id


def test_encode_batch_matches_encode(tok):
    texts = ["a little dog", "the sun"]
    assert tok.encode_batch(texts) == [tok.encode(t) for t in texts]


def test_save_load_round_trip_preserves_ids_and_fingerprint(tok, tmp_path):
    tok.save(tmp_path / "tok")
    again = AaravTokenizer.load(tmp_path / "tok")
    assert again.fingerprint == tok.fingerprint
    text = "The dog ran to the park."
    assert again.encode(text) == tok.encode(text)


def test_tampered_tokenizer_file_is_detected(tok, tmp_path):
    tok.save(tmp_path / "tok")
    p = tmp_path / "tok" / "tokenizer.json"
    p.write_text(p.read_text() + " ")
    with pytest.raises(TokenizerMismatchError):
        AaravTokenizer.load(tmp_path / "tok")


def test_fingerprint_check(tok):
    tok.check_fingerprint(tok.fingerprint)
    with pytest.raises(TokenizerMismatchError):
        tok.check_fingerprint("0" * 64, what="checkpoint")


def test_training_is_deterministic():
    a = train_tokenizer(CORPUS, vocab_size=400)
    b = train_tokenizer(CORPUS, vocab_size=400)
    assert a.fingerprint == b.fingerprint


def test_too_small_vocab_rejected():
    with pytest.raises(ValueError):
        train_tokenizer(CORPUS, vocab_size=100)
