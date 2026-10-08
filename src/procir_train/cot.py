"""Chain-of-thought caption injection with progressive token dropping (paper Sec. 3.4)."""
import math


def keep_ratio(step, total_steps):
    """rho(t) = max(0, 1 - t / (0.5 T)): full caption at step 0, none from mid-training on."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    return max(0.0, 1.0 - step / (0.5 * total_steps))


def subsample_tokens(text, tokenizer, ratio, rng):
    """Keep ceil(ratio*n) tokens in order; approximate (subset is decoded, re-tokenized later); rng used only if 0 < ratio < 1."""
    if ratio <= 0.0:
        return ""
    if ratio >= 1.0:
        return text
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    k = math.ceil(ratio * len(ids))
    keep = sorted(rng.sample(range(len(ids)), k))
    return tokenizer.decode([ids[i] for i in keep])


def truncate_tokens(text, tokenizer, max_tokens):
    """Keep the first max_tokens tokens; approximate (decoded text is re-tokenized later)."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens])
