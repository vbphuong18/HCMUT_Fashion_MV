import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def loaded():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from procir_train.config import TrainConfig
    from procir_train.encoder import load_encoder

    cfg = TrainConfig()
    enc, proc = load_encoder(cfg, torch.device("cuda"))
    enc.train()
    return cfg, enc, proc


def _imgs(n):
    return [Image.new("RGB", (224, 320), ((30 * i) % 255, 80, 160)) for i in range(n)]


def test_encode_shapes_and_gradients(loaded):
    from procir_train.prompts import (caption_text, doc_text, multiturn_query_text, process_text,
                                      process_visual)

    cfg, enc, proc = loaded
    px = (cfg.image_min_pixels, cfg.image_max_pixels)
    q = [process_visual(proc, multiturn_query_text(proc, 2, "make it red"), _imgs(2), *px)]
    d = [process_visual(proc, doc_text(proc, 3), _imgs(3), *px)]
    c = [process_text(proc, caption_text(proc, "a red dress"))]
    reps = enc.encode({"query": q, "doc": d, "tgt_caption": c, "src_caption": c}, cfg)
    assert reps["q"].shape == (1, 1024) and reps["d"].shape == (1, 1024)
    assert not torch.allclose(reps["s"], reps["q"])
    sum(v.float().pow(2).sum() for v in reps.values()).backward()
    lora_grads = [p.grad for n, p in enc.named_parameters() if "lora_" in n and p.grad is not None]
    assert lora_grads and any(g.abs().sum() > 0 for g in lora_grads)
    assert enc.emb_vector.grad is not None and enc.emb_vector.grad.abs().sum() > 0
    enc.zero_grad(set_to_none=True)


def test_only_lora_and_emb_vector_are_trainable(loaded):
    _, enc, _ = loaded
    trainable = [n for n, p in enc.named_parameters() if p.requires_grad]
    assert "emb_vector" in trainable
    assert all(n == "emb_vector" or "lora_" in n for n in trainable)
    assert all(".visual." not in n for n in trainable)


def test_full_finetune_keeps_embedding_table_frozen():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from procir_train.config import TrainConfig
    from procir_train.encoder import load_encoder

    enc, _ = load_encoder(TrainConfig(lora=False), torch.device("cuda"))
    assert not enc.vlm.get_input_embeddings().weight.requires_grad
    trainable = [n for n, p in enc.named_parameters() if p.requires_grad]
    assert any(".language_model.layers." in n for n in trainable)
    assert all(".visual." not in n and "embed_tokens" not in n for n in trainable)
    del enc
    torch.cuda.empty_cache()


def _two_docs(proc, cfg):
    from procir_train.prompts import doc_text, process_visual

    px = (cfg.image_min_pixels, cfg.image_max_pixels)
    d0 = process_visual(proc, doc_text(proc, 1), _imgs(1), *px)
    d1 = process_visual(proc, doc_text(proc, 3), _imgs(3), *px)
    return d0, d1


@pytest.mark.parametrize("padded", [False, True])
def test_no_grad_and_grad_passes_agree(loaded, padded):
    cfg, enc, proc = loaded
    d0, d1 = _two_docs(proc, cfg)
    chunk = [d0, d1] if padded else [d1]
    with torch.no_grad():
        a = enc.encode_single(chunk)
    b = enc.encode_single(chunk)
    cos = torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1)
    print("max abs diff", (a.float() - b.float()).abs().max().item())
    assert (cos > 0.9999).all()


def test_padding_invariance(loaded):
    cfg, enc, proc = loaded
    d0, d1 = _two_docs(proc, cfg)
    with torch.no_grad():
        alone = enc.encode_single([d1])[0]
        both = enc.encode_single([d0, d1])
        alone0 = enc.encode_single([d0])[0]
    padded = both[1]
    cos = torch.nn.functional.cosine_similarity(alone.float(), padded.float(), dim=0)
    print("max abs diff (unpadded row)", (alone.float() - padded.float()).abs().max().item())
    assert cos > 0.9999
    # d0 is the shorter sample, so this is the row that actually carries padding.
    cos0 = torch.nn.functional.cosine_similarity(alone0.float(), both[0].float(), dim=0)
    print("max abs diff (padded row)", (alone0.float() - both[0].float()).abs().max().item())
    assert cos0 > 0.9999
