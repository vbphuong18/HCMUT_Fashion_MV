"""CPU end-to-end run of procir_train.train.main(): real data pipeline, real Qwen3.5 processor,
real ProCIREncoder / GradCache / loss / dev eval / checkpointing, with a tiny toy VLM in place of
Qwen3.5-0.8B. Mirrors tests/test_train_gpu.py (smoke + deterministic resume) without a GPU."""
import json
import math
import shutil
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from PIL import Image

pytestmark = pytest.mark.hf

DIM = 16
LR = 5e-3
MAX_STEPS = 8
RESUME_FROM = 4
COLOURS = {"red": (220, 30, 30), "green": (30, 200, 30), "blue": (30, 30, 220), "yellow": (230, 220, 30),
           "black": (10, 10, 10), "white": (245, 245, 245), "purple": (140, 30, 160), "orange": (240, 140, 20)}


class ToyInner(nn.Module):
    """Stands in for Qwen3_5Model: frozen token table (hooked by ProCIREncoder for <emb_all>), a
    'vision tower' that projects pixel patches, merges them 2x2 and scatters them into the
    <|image_pad|> slots like the real model, then a trainable mixer and causal mean pooling."""

    def __init__(self, vocab, patch_dim, image_token_id, merge):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, DIM)
        self.patch = nn.Linear(patch_dim, DIM)
        self.mix = nn.Linear(DIM, DIM)
        self.image_token_id, self.merge = image_token_id, merge
        self.rope_deltas = None

    def forward(self, input_ids, attention_mask, pixel_values=None, image_grid_thw=None, **kwargs):
        assert set(kwargs) <= {"mm_token_type_ids"}, set(kwargs)
        assert self.rope_deltas is None  # the encoder resets it before each call
        x = self.embed_tokens(input_ids)
        if pixel_values is not None:
            assert int(image_grid_thw.prod(-1).sum()) == pixel_values.shape[0]
            feats = self.patch(pixel_values.float()).view(-1, self.merge, DIM).mean(dim=1)
            slots = input_ids == self.image_token_id
            assert int(slots.sum()) == feats.shape[0], (int(slots.sum()), feats.shape)
            x = x.masked_scatter(slots.unsqueeze(-1), feats.to(x.dtype))
        h = torch.tanh(self.mix(x))
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        h = (h * m).cumsum(dim=1) / m.cumsum(dim=1).clamp(min=1)
        self.rope_deltas = torch.zeros(1)  # mimics the real model's cached state
        return SimpleNamespace(last_hidden_state=h, rope_deltas=None)


class ToyVLM(nn.Module):
    def __init__(self, vocab, patch_dim, image_token_id, merge):
        super().__init__()
        self.model = ToyInner(vocab, patch_dim, image_token_id, merge)

    def get_input_embeddings(self):
        return self.model.embed_tokens


def _fake_load_encoder(processor, built):
    from procir_train.encoder import ProCIREncoder
    from procir_train.prompts import ensure_emb_token

    def load(cfg, device):
        emb_id = ensure_emb_token(processor.tokenizer)
        ip = processor.image_processor
        patch_dim = 3 * ip.temporal_patch_size * ip.patch_size ** 2
        with torch.random.fork_rng():
            torch.manual_seed(0)  # identical frozen weights in every run (only trainables are checkpointed)
            vlm = ToyVLM(len(processor.tokenizer) + 8, patch_dim, processor.image_token_id, ip.merge_size ** 2)
        assert emb_id < vlm.get_input_embeddings().num_embeddings
        vlm.get_input_embeddings().requires_grad_(False)
        emb_init = vlm.get_input_embeddings().weight[: len(processor.tokenizer) - 1].mean(dim=0)
        encoder = ProCIREncoder(vlm, emb_id, emb_init)
        built.append(device)
        return encoder.to("cpu"), processor

    return load


@pytest.fixture
def corpus(tmp_path):
    """Two datasets x 12 sources x 2 targets; every product has 1-3 views of its own colour and
    captions / modification texts that name that colour, which keeps the embeddings distinct."""
    names = list(COLOURS)
    rows, caps, products = [], [], {}
    for d, ds in enumerate(("deepfashion", "f200k")):
        for i in range(12):
            src = f"s{i}"
            products[(ds, src)] = names[(i + d) % len(names)]
            for j in range(2):
                tgt = f"t{i}_{j}"
                colour = names[(i + 3 * j + 2 * d + 1) % len(names)]
                products[(ds, tgt)] = colour
                rows.append({"source_id": src, "target_id": tgt, "dataset": ds,
                             "views_involved": ["front", "back"],
                             "modification_text_short": f"make it {colour}",
                             "modification_text_long": f"change the {products[(ds, src)]} garment to a {colour} one"})
    for n, ((ds, pid), colour) in enumerate(sorted(products.items())):
        d = tmp_path / "images" / ds / pid
        d.mkdir(parents=True)
        for v in range(1 + n % 3):
            Image.new("RGB", (32, 48), COLOURS[colour]).save(d / f"{v:02d}.jpg")
        caps.append({"product_id": pid, "dataset": ds, "short_caption": f"a {colour} garment",
                     "long_caption": f"a {colour} garment photographed from several views; the fabric is {colour}"})
    ann = tmp_path / "ann"
    ann.mkdir()
    (ann / "train_triplets.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (ann / "train_captions.jsonl").write_text("\n".join(json.dumps(c) for c in caps) + "\n", encoding="utf-8")
    return ann, tmp_path / "images"


@pytest.fixture
def run_main(tmp_path, corpus, processor, monkeypatch):
    import procir_train.train as train_mod

    ann, images = corpus
    cfg_path = tmp_path / "e2e.yaml"
    cfg_path.write_text(
        f"data_dir: {json.dumps(ann.as_posix())}\n"
        f"image_root: {json.dumps(images.as_posix())}\n"
        "datasets: [deepfashion, f200k]\n"
        "train_fraction: 1.0\n"
        "dev_product_fraction: 0.1\n"
        "max_views: 3\n"
        "batch_size: 4\n"
        "chunk_size: 2\n"
        "epochs: 1\n"
        f"max_steps: {MAX_STEPS}\n"
        f"lr: {LR!r}\n"
        "log_every: 1\n"
        f"eval_every: {RESUME_FROM}\n"
        f"save_every: {RESUME_FROM}\n"
        "dev_eval_max_queries: 0\n"
        "eval_batch_size: 4\n"
        "gradient_checkpointing: false\n",
        encoding="utf-8")

    devices = []
    monkeypatch.setattr(train_mod, "load_encoder", _fake_load_encoder(processor, devices))
    monkeypatch.setattr(train_mod, "write_env", lambda path: path.write_text("{}"))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a, **k: 0)

    def run(out, *overrides, resume=False):
        argv = ["--config", str(cfg_path), "--set", f"output_dir={out.as_posix()}", "num_workers=0", *overrides]
        train_mod.main(argv + (["--resume"] if resume else []))
        lines = [json.loads(l) for l in (out / "train_log.jsonl").read_text().splitlines() if l.strip()]
        return {l["step"]: l for l in lines if "loss" in l}, [l for l in lines if "dev" in l]

    run.devices = devices
    return run


def test_cpu_train_then_deterministic_resume(tmp_path, run_main):
    run1 = tmp_path / "run1"
    logs1, devs = run_main(run1)
    assert len(run_main.devices) == 1

    stats = json.loads((run1 / "data_stats.json").read_text())
    assert all(stats[ds]["dev"] > 0 and stats[ds]["train"] >= 8 for ds in ("deepfashion", "f200k"))

    assert sorted(logs1) == list(range(1, MAX_STEPS + 1))
    for rec in logs1.values():
        assert set(rec) >= {"loss", "l_cir", "l_doc", "l_src", "lr", "grad_norm"}
        assert all(torch.isfinite(torch.tensor(rec[k])) for k in ("loss", "l_cir", "l_doc", "l_src", "grad_norm"))
    for s in logs1:  # logged lr = the lr step s used: cosine over MAX_STEPS, no warmup
        assert logs1[s]["lr"] == pytest.approx(LR * 0.5 * (1 + math.cos(math.pi * (s - 1) / MAX_STEPS)))

    assert [d["step"] for d in devs] == [RESUME_FROM, MAX_STEPS]
    for d in devs:
        assert set(d["dev"]) == {"deepfashion", "f200k"}
        for res in d["dev"].values():
            assert {"R@1", "R@5", "R@10", "R@1_src_excluded", "n_queries", "n_gallery"} <= set(res)
            assert res["n_queries"] > 0 and 0.0 <= res["R@5"] <= 100.0

    ck = run1 / "ckpt"
    assert (ck / "latest.pt").exists()
    assert (ck / f"step_{RESUME_FROM:06d}.pt").exists() and (ck / f"step_{MAX_STEPS:06d}.pt").exists()
    final = torch.load(ck / "latest.pt", weights_only=False)
    assert final["step"] == MAX_STEPS
    assert {"emb_vector", "vlm.model.patch.weight", "vlm.model.mix.weight"} <= set(final["trainable"])
    assert "vlm.model.embed_tokens.weight" not in final["trainable"]
    mid = torch.load(ck / f"step_{RESUME_FROM:06d}.pt", weights_only=False)
    assert not torch.equal(mid["trainable"]["vlm.model.mix.weight"],
                           final["trainable"]["vlm.model.mix.weight"])  # trainables actually moved
    assert not torch.equal(mid["trainable"]["emb_vector"], final["trainable"]["emb_vector"])  # <emb_all> hook live

    with pytest.raises(ValueError, match="already exists"):  # no silent append to a finished run
        run_main(run1)

    run2 = tmp_path / "run2"
    (run2 / "ckpt").mkdir(parents=True)
    shutil.copyfile(ck / f"step_{RESUME_FROM:06d}.pt", run2 / "ckpt" / "latest.pt")
    logs2, devs2 = run_main(run2, "eval_every=0", resume=True)
    assert devs2 == []
    assert sorted(logs2) == list(range(RESUME_FROM + 1, MAX_STEPS + 1))
    for s in logs2:
        for k in ("loss", "l_cir", "l_doc", "l_src", "lr"):
            assert logs2[s][k] == pytest.approx(logs1[s][k], rel=1e-6, abs=1e-12), (s, k)
    resumed = torch.load(run2 / "ckpt" / "latest.pt", weights_only=False)
    assert resumed["step"] == MAX_STEPS
    for name, t in final["trainable"].items():
        torch.testing.assert_close(resumed["trainable"][name], t, rtol=1e-6, atol=1e-7, msg=name)


@pytest.mark.parametrize("overrides, parts", [
    (["align=false"], {"l_cir"}),
    (["multi_turn=false"], {"l_cir", "l_doc"}),
    (["cot=true"], {"l_cir", "l_doc", "l_src"}),
])
def test_cpu_variants_run(tmp_path, run_main, overrides, parts):
    out = tmp_path / "variant"
    logs, devs = run_main(out, "max_steps=2", "eval_every=2", "save_every=2", *overrides)
    assert sorted(logs) == [1, 2]
    for rec in logs.values():
        assert {k for k in rec if k.startswith("l_")} == parts
        assert torch.isfinite(torch.tensor(rec["loss"]))
    assert len(devs) == 1 and "R@5" in devs[0]["dev"]["deepfashion"]
    assert (out / "ckpt" / "latest.pt").exists()


def test_cpu_stop_after_steps_keeps_full_schedule(tmp_path, run_main):
    out = tmp_path / "pilot"
    logs, _ = run_main(out, "stop_after_steps=2", "eval_every=0")
    assert sorted(logs) == [1, 2]
    assert (out / "ckpt" / "step_000002.pt").exists() and (out / "ckpt" / "latest.pt").exists()
    # lr follows the MAX_STEPS-long cosine, not a 2-step one
    assert logs[2]["lr"] == pytest.approx(LR * 0.5 * (1 + math.cos(math.pi / MAX_STEPS)))
    assert torch.load(out / "ckpt" / "latest.pt", weights_only=False)["step"] == 2


def test_cpu_resume_refuses_changed_batch_size(tmp_path, run_main):
    out = tmp_path / "run"
    run_main(out, "stop_after_steps=2", "eval_every=0")
    with pytest.raises(ValueError, match="batch_size"):
        run_main(out, "batch_size=2", "eval_every=0", resume=True)
