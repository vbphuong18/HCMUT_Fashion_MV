import torch
import torch.nn.functional as F

from procir_train.config import TrainConfig
from procir_train.losses import procir_loss, same_key_matrix, sym_info_nce


def test_matches_reference_formula():
    torch.manual_seed(0)
    x, y = torch.randn(5, 8), torch.randn(5, 8)
    s = F.normalize(x, dim=-1) @ F.normalize(y, dim=-1).T / 0.07
    ref = 0.5 * (-(s.diag() - s.logsumexp(1)).mean() - (s.diag() - s.logsumexp(0)).mean())
    assert torch.allclose(sym_info_nce(x, y, 0.07), ref, atol=1e-5)


def test_duplicate_targets_are_not_negatives():
    x = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    same = same_key_matrix(["a", "a", "b"], torch.device("cpu"))
    assert sym_info_nce(x, x.clone(), 0.07, same) < 1e-3
    assert sym_info_nce(x, x.clone(), 0.07) > 0.4


def test_procir_loss_parts_follow_switches():
    torch.manual_seed(0)
    reps = {k: torch.randn(4, 8) for k in ("q", "d", "s", "t_src", "t_tgt")}
    src, tgt = ["s0", "s1", "s2", "s3"], ["t0", "t1", "t2", "t3"]
    total, parts = procir_loss(reps, src, tgt, TrainConfig())
    assert set(parts) == {"l_cir", "l_doc", "l_src"}
    assert torch.allclose(total, parts["l_cir"] + 0.25 * parts["l_doc"] + 0.25 * parts["l_src"])
    assert set(procir_loss(reps, src, tgt, TrainConfig(multi_turn=False))[1]) == {"l_cir", "l_doc"}
    assert set(procir_loss(reps, src, tgt, TrainConfig(align=False))[1]) == {"l_cir"}


def _src_reps():
    s = torch.tensor([[1.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    reps = {k: torch.randn(4, 3) for k in ("q", "d", "t_tgt")}
    reps.update(s=s, t_src=s.clone())
    return reps


def test_l_src_masks_by_source_key():
    torch.manual_seed(0)
    reps = _src_reps()
    tgt = ["t0", "t1", "t2", "t3"]
    _, masked = procir_loss(reps, ["a", "a", "b", "c"], tgt, TrainConfig())
    _, unmasked = procir_loss(reps, ["a", "a2", "b", "c"], tgt, TrainConfig())
    assert masked["l_src"] < 1e-3
    # two tied rows out of four -> ln2 / 2 ~= 0.35 when the mask is absent
    assert unmasked["l_src"] > 0.3


def test_fully_masked_rows_keep_finite_gradients():
    torch.manual_seed(0)
    x = torch.randn(4, 8, requires_grad=True)
    y = torch.randn(4, 8)
    same = same_key_matrix(["a"] * 4, torch.device("cpu"))
    loss = sym_info_nce(x, y, 0.07, same)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(x.grad).all()
