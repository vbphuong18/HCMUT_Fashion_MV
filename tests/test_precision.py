import torch
import torch.nn as nn

from procir_train.precision import MasterWeights


def _bf16_linear(seed=0):
    torch.manual_seed(seed)
    lin = nn.Linear(256, 256)
    with torch.no_grad():
        lin.weight.mul_(0.5)
    return lin.bfloat16()


def test_fp32_params_are_their_own_master():
    lin = nn.Linear(4, 4)
    mw = MasterWeights(lin.named_parameters())
    assert all(m is p for m, p in zip(mw.masters, lin.parameters()))


def test_small_updates_survive_on_bf16_params():
    """AdamW at the paper's lr=1e-5 is below bf16 resolution; the fp32 master must keep it."""
    model = _bf16_linear()
    ref = nn.Linear(256, 256)
    with torch.no_grad():
        ref.weight.copy_(model.weight.float())
        ref.bias.copy_(model.bias.float())
    mw = MasterWeights(model.named_parameters())
    opt = torch.optim.AdamW(mw.masters, lr=1e-5, weight_decay=0.01)
    ref_opt = torch.optim.AdamW(ref.parameters(), lr=1e-5, weight_decay=0.01)
    torch.manual_seed(1)
    for _ in range(50):
        x = torch.randn(8, 256)
        model(x.bfloat16()).float().pow(2).mean().backward()
        ref(x).pow(2).mean().backward()
        opt.step()
        ref_opt.step()
        mw.copy_to_model()
        opt.zero_grad(set_to_none=True)
        ref_opt.zero_grad(set_to_none=True)
    master_w = mw.masters[0]
    moved = (master_w.detach() != _bf16_linear().weight.float()).float().mean()
    assert moved > 0.99  # nearly every weight moved in the master copy
    assert torch.equal(model.weight, master_w.detach().to(torch.bfloat16))
    upd_master = master_w.detach() - _bf16_linear().weight.float()
    upd_ref = ref.weight.detach() - _bf16_linear().weight.float()
    assert torch.nn.functional.cosine_similarity(upd_master.flatten(), upd_ref.flatten(), dim=0) > 0.95


def test_grads_accumulate_in_fp32_across_backward_calls():
    model = _bf16_linear()
    mw = MasterWeights(model.named_parameters())
    xs = [torch.randn(4, 256) for _ in range(3)]
    for x in xs:  # like GradCache: one backward per chunk
        model(x.bfloat16()).float().sum().backward()
    assert model.weight.grad is None  # moved to the master, no bf16 copy kept
    expected = sum(x.sum(dim=0) for x in xs)  # d(sum W x)/dW row = sum of inputs
    assert mw.masters[0].grad.dtype == torch.float32
    assert torch.allclose(mw.masters[0].grad[0], expected.bfloat16().float(), rtol=2e-2, atol=2e-2)


def test_state_dict_roundtrip_restores_fp32_values():
    model = _bf16_linear()
    mw = MasterWeights(model.named_parameters())
    with torch.no_grad():
        mw.masters[0].add_(1e-6)  # below bf16 resolution
    state = mw.state_dict()
    assert state["weight"].dtype == torch.float32

    model2 = _bf16_linear()
    mw2 = MasterWeights(model2.named_parameters())
    mw2.load_state_dict(state)
    assert torch.equal(mw2.masters[0], mw.masters[0])
    assert torch.equal(model2.weight, mw.masters[0].detach().to(torch.bfloat16))
