"""Prompt strings for ProCIR. Doc and two-turn query must stay token-identical to the
upstream eval collators (procir.collators.BaseCollator), so train and eval see one format."""
from procir.chat_utils import _ASSISTANT_PREFIX, _THINK_BLOCK, patch_think_tokens

EMB_TOKEN = "<emb_all>"


def ensure_emb_token(tokenizer):
    tid = tokenizer.convert_tokens_to_ids(EMB_TOKEN)
    if tid is None or tid == tokenizer.unk_token_id:
        tokenizer.add_tokens([EMB_TOKEN], special_tokens=True)
        tid = tokenizer.convert_tokens_to_ids(EMB_TOKEN)
    return tid


def _render(processor, messages):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return patch_think_tokens(text)


def _images(n):
    return [{"type": "image"} for _ in range(n)]


def _assistant():
    return {"role": "assistant", "content": [{"type": "text", "text": EMB_TOKEN}]}


def doc_text(processor, n_images):
    return _render(processor, [{"role": "user", "content": _images(n_images)}, _assistant()])


def multiturn_query_text(processor, n_images, mod_text):
    return _render(processor, [
        {"role": "user", "content": _images(n_images)},
        _assistant(),
        {"role": "user", "content": [{"type": "text", "text": mod_text}]},
        _assistant(),
    ])


def singleturn_query_text(processor, n_images, mod_text):
    content = _images(n_images) + [{"type": "text", "text": mod_text}]
    return _render(processor, [{"role": "user", "content": content}, _assistant()])


def caption_text(processor, caption):
    # Assumption 1: the text-only caption pass reuses the same chat frame.
    return _render(processor, [{"role": "user", "content": [{"type": "text", "text": caption}]},
                               _assistant()])


def inject_think(text, turn, content):
    """Fill the empty think block of assistant turn `turn` (0-based).

    Done on the rendered string because the Qwen3.5 chat template drops reasoning
    from assistant turns that are followed by another user turn."""
    start = -1
    for _ in range(turn + 1):
        start = text.find(_ASSISTANT_PREFIX, start + 1)
        if start < 0:
            raise ValueError(f"assistant turn {turn} not found")
    body = start + len(_ASSISTANT_PREFIX)
    if not text.startswith(_THINK_BLOCK, body):
        raise ValueError(f"assistant turn {turn} has no empty think block")
    filled = "<think>\n" + content + "\n</think>\n\n"
    return text[:body] + filled + text[body + len(_THINK_BLOCK):]


def process_visual(processor, text, images, min_pixels, max_pixels):
    return processor(text=[text], images=images, return_tensors="pt",
                     min_pixels=min_pixels, max_pixels=max_pixels)


def process_text(processor, text):
    return processor.tokenizer(text, return_tensors="pt")
