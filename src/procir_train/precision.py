"""fp32 master weights for bf16 trainable parameters ("bfloat16 mixed precision", paper Table 7).

AdamW applied directly to bf16 weights loses most updates at lr ~1e-5 (bf16 keeps 8 mantissa
bits). The model keeps computing in bf16; each bf16 parameter gets an fp32 master that the
optimizer updates, and gradients are moved to the master in fp32 as soon as they are accumulated,
so GradCache's per-chunk backward passes sum in fp32. fp32 parameters (LoRA adapters, the
<emb_all> vector) are their own master, so the LoRA path is unchanged."""
import torch


class MasterWeights:
    def __init__(self, named_params):
        self.names, self.params, self.masters = [], [], []
        for name, p in named_params:
            if not p.requires_grad:
                continue
            if p.dtype == torch.float32:
                m = p
            else:
                m = p.detach().float().clone().requires_grad_(True)
                p.register_post_accumulate_grad_hook(self._mover(m))
            self.names.append(name)
            self.params.append(p)
            self.masters.append(m)

    @staticmethod
    def _mover(master):
        def hook(p):
            g = p.grad.float()
            master.grad = g if master.grad is None else master.grad.add_(g)
            p.grad = None
        return hook

    def copy_to_model(self):
        """Write the updated fp32 masters back into the (bf16) model parameters."""
        with torch.no_grad():
            for p, m in zip(self.params, self.masters):
                if m is not p:
                    p.copy_(m)

    def state_dict(self):
        return {n: m.detach().cpu().clone() for n, m in zip(self.names, self.masters)}

    def load_state_dict(self, state):
        with torch.no_grad():
            for n, m in zip(self.names, self.masters):
                m.copy_(state[n])
        self.copy_to_model()
