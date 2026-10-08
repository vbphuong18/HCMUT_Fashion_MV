import pytest
import torch
import torch.nn as nn

from procir_train.config import TrainConfig
from procir_train.gradcache import gradcache_step
from procir_train.losses import procir_loss, sym_info_nce


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.a, self.b = nn.Linear(6, 4), nn.Linear(6, 4)

    def encode(self, chunk):
        return {"q": self.a(chunk["x"]), "d": self.b(chunk["y"])}


def loss_fn(reps):
    loss = sym_info_nce(reps["q"], reps["d"], 0.07)
    return loss, {"l": loss.detach()}


def test_gradcache_matches_full_batch_backprop():
    torch.manual_seed(1)
    X, Y = torch.randn(8, 6), torch.randn(8, 6)
    m = Toy()
    full, _ = loss_fn(m.encode({"x": X, "y": Y}))
    full.backward()
    ref = {n: p.grad.clone() for n, p in m.named_parameters()}
    m.zero_grad()
    chunks = [{"x": X[i:i + 2], "y": Y[i:i + 2]} for i in range(0, 8, 2)]
    loss, parts = gradcache_step(m.encode, chunks, loss_fn)
    assert torch.allclose(loss, full.detach(), atol=1e-6)
    assert set(parts) == {"l"}
    for n, p in m.named_parameters():
        assert torch.allclose(p.grad, ref[n], atol=1e-6), n


# ---- shared-trunk model with the real ProCIR loss -------------------------------------------
SRC = ["a", "a", "b", "c", "d", "e", "f"]
TGT = ["p", "q", "q", "r", "s", "t", "u"]


class SharedToy(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.trunk = nn.Linear(6, 8)
        self.heads = nn.ModuleDict({k: nn.Linear(8, 4) for k in ["s", "q", "d", "t_src", "t_tgt"]})

    def encode(self, c):
        src = {"s": "x", "q": "x", "d": "y", "t_tgt": "z", "t_src": "w"}
        return {k: self.heads[k](torch.tanh(self.trunk(c[v]))) for k, v in src.items()}


def _batch():
    torch.manual_seed(1)
    return {k: torch.randn(7, 6) for k in "xyzw"}


def _chunks(batch, sizes):
    out, i = [], 0
    for n in sizes:
        out.append({k: v[i:i + n] for k, v in batch.items()})
        i += n
    return out


@pytest.mark.parametrize("cfg", [TrainConfig(), TrainConfig(align=False), TrainConfig(multi_turn=False)])
def test_gradcache_shared_trunk_with_procir_loss(cfg):
    batch = _batch()
    m = SharedToy()
    lf = lambda reps: procir_loss(reps, SRC, TGT, cfg)
    full, full_parts = lf(m.encode(batch))
    full.backward()
    ref = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True)
    loss, parts = gradcache_step(m.encode, _chunks(batch, [3, 3, 1]), lf)
    assert torch.allclose(loss, full.detach(), atol=1e-5)
    assert set(parts) == set(full_parts)
    for n, p in m.named_parameters():
        if n in ref:
            assert torch.allclose(p.grad, ref[n], atol=1e-5), n
        else:
            assert p.grad is None or p.grad.abs().sum() == 0, n


def test_gradcache_ignores_unused_keys():
    batch = _batch()
    m = SharedToy()
    cfg = TrainConfig(align=False)
    loss, parts = gradcache_step(m.encode, _chunks(batch, [4, 3]), lambda r: procir_loss(r, SRC, TGT, cfg))
    assert torch.isfinite(loss)
    assert m.trunk.weight.grad is not None


def test_gradcache_no_gradient_raises():
    m = SharedToy()
    with pytest.raises(ValueError):
        gradcache_step(m.encode, _chunks(_batch(), [4, 3]), lambda r: (torch.zeros((), requires_grad=True) * 1.0, {}))


def test_gradcache_chunk_key_mismatch_raises():
    chunks = [{"x": torch.randn(2, 6), "y": torch.randn(2, 6)}] * 2
    calls = iter([{"q": torch.randn(2, 4), "d": torch.randn(2, 4)}, {"q": torch.randn(2, 4)}])
    with pytest.raises(ValueError, match="keys"):
        gradcache_step(lambda c: next(calls), chunks, lambda r: (r["q"].sum(), {}))


def test_gradcache_row_count_mismatch_raises():
    chunks = [{"x": torch.randn(2, 6)}]
    with pytest.raises(ValueError, match="'d'"):
        gradcache_step(lambda c: {"q": torch.randn(2, 4), "d": torch.randn(3, 4)}, chunks, lambda r: (r["q"].sum(), {}))
