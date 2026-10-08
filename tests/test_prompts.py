import pytest
import torch
from PIL import Image

from procir.collators import IMAGE_MAX_PIXELS, IMAGE_MIN_PIXELS, BaseCollator
from procir_train.prompts import (EMB_TOKEN, caption_text, doc_text, ensure_emb_token, inject_think,
                                  multiturn_query_text, process_text, process_visual,
                                  singleturn_query_text)

pytestmark = pytest.mark.hf


def _imgs(n):
    return [Image.new("RGB", (200, 300), ((40 * i) % 255, 90, 140)) for i in range(n)]


def test_emb_token_fits_padded_vocab(processor):
    tid = ensure_emb_token(processor.tokenizer)
    assert tid == ensure_emb_token(processor.tokenizer)  # idempotent
    assert tid < 248320  # Qwen3.5-0.8B embedding rows, so no resize is needed


def test_doc_matches_upstream(processor):
    emb_id = ensure_emb_token(processor.tokenizer)
    imgs = _imgs(3)
    theirs = BaseCollator(processor, emb_id)._build_doc(imgs)
    ours = process_visual(processor, doc_text(processor, 3), imgs, IMAGE_MIN_PIXELS, IMAGE_MAX_PIXELS)
    assert torch.equal(ours["input_ids"], theirs["input_ids"])
    assert torch.equal(ours["pixel_values"], theirs["pixel_values"])


def test_multiturn_query_matches_upstream(processor):
    emb_id = ensure_emb_token(processor.tokenizer)
    imgs = _imgs(2)
    mod = "Make the sleeves shorter and add a back zipper."
    theirs = BaseCollator(processor, emb_id)._build_multiturn_query(imgs, mod)
    ours = process_visual(processor, multiturn_query_text(processor, 2, mod), imgs,
                          IMAGE_MIN_PIXELS, IMAGE_MAX_PIXELS)
    assert torch.equal(ours["input_ids"], theirs["input_ids"])


def test_inject_think_fills_only_the_requested_turn(processor):
    t = multiturn_query_text(processor, 2, "add a zipper")
    out = inject_think(t, 0, "red dress with pockets")
    assert out.count(f"<think>\nred dress with pockets\n</think>\n\n{EMB_TOKEN}") == 1
    assert out.count(f"<think>\n\n</think>\n\n{EMB_TOKEN}") == 1
    with pytest.raises(ValueError):
        inject_think(t, 2, "x")


def test_singleturn_and_caption_layouts(processor):
    s = singleturn_query_text(processor, 2, "make it red")
    assert s.count(EMB_TOKEN) == 1
    assert s.index("<|vision_start|>") < s.index("make it red")
    c = caption_text(processor, "a blue denim jacket")
    assert c.count(EMB_TOKEN) == 1 and "<|vision_start|>" not in c
    ids = process_text(processor, c)["input_ids"]
    assert ids.shape[0] == 1
    assert (ids == ensure_emb_token(processor.tokenizer)).sum() == 1


def test_inject_think_turns_are_assistant_turns(processor):
    t = multiturn_query_text(processor, 2, "add a zipper")
    out = inject_think(inject_think(t, 0, "T0"), 1, "T1")
    p0 = out.index("<think>\nT0\n</think>")
    p1 = out.index("<think>\nT1\n</think>")
    assert p0 < out.index("add a zipper") < p1
    assert out.count("<think>\n\n</think>") == 0


def test_inject_think_refuses_filled_turn_and_empty_content_is_noop(processor):
    t = multiturn_query_text(processor, 2, "add a zipper")
    once = inject_think(t, 0, "T0")
    with pytest.raises(ValueError):
        inject_think(once, 0, "T1")
    assert inject_think(t, 1, "") == t
