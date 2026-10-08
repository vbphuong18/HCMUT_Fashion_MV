"""GradCache (Gao et al., 2021): a large contrastive batch with chunk-sized activation memory.

1) encode every chunk without grad; 2) compute the full-batch loss on detached embeddings and
backprop to get d(loss)/d(embedding); 3) re-encode each chunk with grad and push those embedding
gradients through the network. encode_fn must be deterministic (no dropout) so 1) and 3) match.

Contract:
- both passes must see identical inputs and padding (the same chunk object is encoded twice);
- every chunk must return the same set of keys, each a [n, dim] tensor;
- keys the loss does not use (no gradient reaches them) are ignored."""
import torch


def gradcache_step(encode_fn, chunks, loss_fn):
    with torch.no_grad():
        cached = [encode_fn(c) for c in chunks]
    keys = list(cached[0])
    for i, c in enumerate(cached):
        if set(c) != set(keys):
            raise ValueError(f"chunk {i} returned keys {sorted(c)}, expected {sorted(keys)}")
        for k in keys:
            if c[k].shape[0] != c[keys[0]].shape[0]:
                raise ValueError(f"chunk {i}: key {k!r} has {c[k].shape[0]} rows, "
                                 f"but {keys[0]!r} has {c[keys[0]].shape[0]}")
    full = {k: torch.cat([c[k] for c in cached]).float().detach().requires_grad_(True) for k in keys}
    loss, parts = loss_fn(full)
    loss.backward()
    grads = {k: v.grad for k, v in full.items() if v.grad is not None}
    if not grads:
        raise ValueError("loss_fn produced no gradient for any embedding key")

    start = 0
    for chunk, cache in zip(chunks, cached):
        n = cache[keys[0]].shape[0]
        reps = encode_fn(chunk)
        surrogate = sum((reps[k].float() * g[start:start + n]).sum() for k, g in grads.items())
        surrogate.backward()
        start += n
    return loss.detach(), parts
