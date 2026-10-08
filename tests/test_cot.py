import random

import pytest

from procir_train.cot import keep_ratio, subsample_tokens, truncate_tokens


class FakeTok:
    def __init__(self):
        self.vocab = []

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for w in text.split():
            if w not in self.vocab:
                self.vocab.append(w)
            ids.append(self.vocab.index(w))
        return {"input_ids": ids}

    def decode(self, ids):
        return " ".join(self.vocab[i] for i in ids)


def test_keep_ratio_schedule():
    assert keep_ratio(0, 100) == 1.0
    assert keep_ratio(25, 100) == pytest.approx(0.5)
    assert keep_ratio(50, 100) == 0.0
    assert keep_ratio(90, 100) == 0.0


def test_subsample_keeps_ceil_ratio_tokens_in_order():
    text = "a b c d e f g h i j"
    out = subsample_tokens(text, FakeTok(), 0.35, random.Random(0)).split()
    assert len(out) == 4  # ceil(0.35 * 10)
    pos = [text.split().index(w) for w in out]
    assert pos == sorted(pos)


def test_subsample_edges():
    tok = FakeTok()
    assert subsample_tokens("a b c", tok, 1.0, random.Random(0)) == "a b c"
    assert subsample_tokens("a b c", tok, 0.0, random.Random(0)) == ""


def test_truncate_tokens():
    tok = FakeTok()
    assert truncate_tokens("a b c d", tok, 2) == "a b"
    assert truncate_tokens("a b", tok, 5) == "a b"


def test_keep_ratio_end_and_invalid_total():
    assert keep_ratio(100, 100) == 0.0
    with pytest.raises(ValueError, match="total_steps must be positive"):
        keep_ratio(0, 0)
    with pytest.raises(ValueError, match="total_steps must be positive"):
        keep_ratio(5, -3)


def test_subsample_deterministic_per_seed():
    text = " ".join(f"w{i}" for i in range(20))
    a = subsample_tokens(text, FakeTok(), 0.5, random.Random(7))
    b = subsample_tokens(text, FakeTok(), 0.5, random.Random(7))
    assert a == b
    outs = {subsample_tokens(text, FakeTok(), 0.5, random.Random(s)) for s in range(10)}
    assert len(outs) > 1


def test_subsample_empty_text_and_tiny_ratio():
    assert subsample_tokens("", FakeTok(), 0.5, random.Random(0)) == ""
    out = subsample_tokens("a b c d e", FakeTok(), 1e-9, random.Random(0))
    assert len(out.split()) == 1
