"""SymInfoNCE and the ProCIR objective L = L_cir + lambda_d L_doc + lambda_s L_src."""
import torch
import torch.nn.functional as F


def same_key_matrix(keys, device):
    ids = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    t = torch.tensor([ids[k] for k in keys], device=device)
    return t.unsqueeze(0) == t.unsqueeze(1)


def sym_info_nce(x, y, tau, same=None):
    """Symmetric InfoNCE on L2-normalised rows; off-diagonal pairs flagged in `same` are not negatives."""
    x = F.normalize(x.float(), dim=-1)
    y = F.normalize(y.float(), dim=-1)
    logits = x @ y.T / tau
    if same is not None:
        off_diag = same & ~torch.eye(len(x), dtype=torch.bool, device=x.device)
        logits = logits.masked_fill(off_diag, float("-inf"))
    labels = torch.arange(len(x), device=x.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def procir_loss(reps, source_keys, target_keys, cfg):
    device = reps["q"].device
    same_t = same_key_matrix(target_keys, device)
    parts = {"l_cir": sym_info_nce(reps["q"], reps["d"], cfg.tau, same_t)}
    total = parts["l_cir"]
    if cfg.align:
        parts["l_doc"] = sym_info_nce(reps["d"], reps["t_tgt"], cfg.tau, same_t)
        total = total + cfg.lambda_doc * parts["l_doc"]
        if cfg.multi_turn:
            same_s = same_key_matrix(source_keys, device)
            parts["l_src"] = sym_info_nce(reps["s"], reps["t_src"], cfg.tau, same_s)
            total = total + cfg.lambda_src * parts["l_src"]
    return total, {k: v.detach() for k, v in parts.items()}
