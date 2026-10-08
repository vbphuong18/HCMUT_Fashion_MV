"""CPU end-to-end check of load_encoder on a tiny random Qwen3.5 (real architecture, real processor)."""
import pytest
import torch

from procir_train.config import TrainConfig

pytestmark = pytest.mark.hf


def _tiny_vlm(base_model):
    from transformers import AutoConfig, Qwen3_5ForConditionalGeneration

    config = AutoConfig.from_pretrained(base_model)
    t, v = config.text_config, config.vision_config
    t.hidden_size, t.intermediate_size = 64, 128
    t.num_hidden_layers = 4
    t.layer_types = t.layer_types[:4]
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 4, 2, 16
    t.linear_num_key_heads = t.linear_num_value_heads = 2
    t.linear_key_head_dim = t.linear_value_head_dim = 16
    t.rope_parameters["mrope_section"] = [1, 1, 0]  # sums to head_dim * partial_rotary_factor / 2
    t.mtp_num_hidden_layers = 0
    v.depth, v.hidden_size, v.num_heads, v.intermediate_size, v.out_hidden_size = 1, 32, 2, 64, 64
    torch.manual_seed(0)
    return Qwen3_5ForConditionalGeneration(config).to(torch.bfloat16), config


@pytest.fixture
def tiny_encoder(monkeypatch):
    from transformers import Qwen3_5ForConditionalGeneration

    from procir_train.encoder import load_encoder

    def build(**cfg_kw):
        cfg = TrainConfig(**cfg_kw)
        vlm, _ = _tiny_vlm(cfg.base_model)
        monkeypatch.setattr(Qwen3_5ForConditionalGeneration, "from_pretrained",
                            classmethod(lambda cls, *a, **k: vlm))
        enc, proc = load_encoder(cfg, torch.device("cpu"))
        return cfg, enc, proc

    return build


def _inputs(proc, cfg):
    from PIL import Image

    from procir_train.prompts import doc_text, multiturn_query_text, process_visual

    px = (32 * 32, 64 * 64)
    imgs = lambda n: [Image.new("RGB", (48, 64), (30 * i, 90, 160)) for i in range(n)]
    q = process_visual(proc, multiturn_query_text(proc, 2, "make it red"), imgs(2), *px)
    d0 = process_visual(proc, doc_text(proc, 1), imgs(1), *px)
    d1 = process_visual(proc, doc_text(proc, 2), imgs(2), *px)
    return q, d0, d1


def test_lora_load_hook_trainables_and_determinism(tiny_encoder):
    cfg, enc, proc = tiny_encoder()
    names = [n for n, p in enc.named_parameters() if p.requires_grad]
    assert "emb_vector" in names
    assert all(n == "emb_vector" or "lora_" in n for n in names)
    assert all(".visual." not in n for n in names)
    assert any("lora_" in n and "language_model" in n for n in names)
    lora_modules = [n for n, _ in enc.named_parameters() if "lora_A" in n]
    assert lora_modules and all("visual" not in n and "lm_head" not in n for n in lora_modules)

    enc.train()
    q, d0, d1 = _inputs(proc, cfg)
    assert q.get("mm_token_type_ids") is not None

    with torch.no_grad():
        s0, q0 = enc.encode_multiturn([q])
        a = enc.encode_single([d0, d1])
    s1, q1 = enc.encode_multiturn([q])
    b = enc.encode_single([d0, d1])
    assert torch.equal(q0, q1) and torch.equal(s0, s1) and torch.equal(a, b)
    assert not torch.allclose(s1, q1)

    (q1.float().pow(2).sum() + b.float().pow(2).sum()).backward()
    assert enc.emb_vector.grad is not None and enc.emb_vector.grad.abs().sum() > 0
    # lora_B starts at zero, so only lora_B has a nonzero gradient at init: proves LoRA is on the forward path.
    b_grads = [p.grad for n, p in enc.named_parameters() if "lora_B" in n and p.grad is not None]
    assert b_grads and any(g.abs().sum() > 0 for g in b_grads)


def test_full_finetune_keeps_embedding_table_frozen(tiny_encoder):
    _, enc, _ = tiny_encoder(lora=False)
    assert not enc.vlm.get_input_embeddings().weight.requires_grad
    trainable = [n for n, p in enc.named_parameters() if p.requires_grad]
    assert any(".language_model.layers." in n for n in trainable)
    assert all(".visual." not in n and "embed_tokens" not in n for n in trainable)
