import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import baselines.qwen3vl_emb_eval as qe
from eval_metrics import report
from eval_metrics.ranking import load_dump, rank_queries, save_dump

# ---- prompts ----


def test_messages_images_then_text_with_instruction():
    conv = qe.messages("Find the product", 3, "make it red")
    assert conv[0] == {"role": "system", "content": [{"type": "text", "text": "Find the product."}]}
    user = conv[1]
    assert user["role"] == "user"
    assert [c["type"] for c in user["content"]] == ["image"] * 3 + ["text"]
    assert user["content"][-1]["text"] == "make it red"


def test_doc_messages_are_images_only_with_default_instruction():
    conv = qe.messages(qe.DOC_INSTRUCTION, 2)
    assert conv[0]["content"][0]["text"] == "Represent the user's input."
    assert [c["type"] for c in conv[1]["content"]] == ["image", "image"]
    assert qe.messages("x", 0)[1]["content"] == [{"type": "text", "text": "NULL"}]
    assert qe.instruction_text("  Retrieve it!  ") == "Retrieve it!"


def test_items_joint_and_perview():
    products = {"d/a": ["a0", "a1"], "d/b": ["b0"]}
    queries = [{"source_images": ["a0", "a1"], "mod_short": "to b"}]
    g, q = qe.joint_items(products, list(products), queries, "Q.")
    assert [p for p, _ in g] == [["a0", "a1"], ["b0"]]
    assert q[0][0] == ["a0", "a1"] and q[0][1] == qe.messages("Q.", 2, "to b")
    g, g_own, q, q_own = qe.perview_items(products, list(products), queries, "Q.")
    assert [p for p, _ in g] == [["a0"], ["a1"], ["b0"]] and g_own == [0, 0, 1]
    assert [p for p, _ in q] == [["a0"], ["a1"]] and q_own == [0, 0]
    assert all(c == qe.messages("Q.", 1, "to b") for _, c in q)
    assert g[0][1] == qe.messages(qe.DOC_INSTRUCTION, 1)


class FakeProcessor:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, conv, add_generation_prompt, tokenize):
        assert add_generation_prompt is True and tokenize is False
        n = sum(c["type"] == "image" for c in conv[1]["content"])
        return f"{conv[0]['content'][0]['text']}|{n}"

    def __call__(self, **kw):
        self.calls.append(kw)
        return kw


def test_processor_inputs_flattens_images_in_order_and_caps_pixels():
    proc = FakeProcessor()
    convs = [qe.messages("I.", 2, "t"), qe.messages("I.", 1)]
    out = qe.processor_inputs(proc, convs, [["x0", "x1"], ["y0"]], (128 * 128, 512 * 512))
    assert out["text"] == ["I.|2", "I.|1"]
    assert out["images"] == ["x0", "x1", "y0"]
    assert out["padding"] is True and out["return_tensors"] == "pt"
    assert (out["min_pixels"], out["max_pixels"]) == (128 * 128, 512 * 512)


# ---- pooling and the model call ----


def test_pool_last_right_padding():
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float).view(2, 4, 3)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    out = qe.pool_last(hidden, mask)
    assert torch.equal(out, torch.stack([hidden[0, 2], hidden[1, 1]]))
    full = torch.ones(2, 4, dtype=torch.long)
    assert torch.equal(qe.pool_last(hidden, full), hidden[:, 3])


def test_runner_pools_last_token_without_cache():
    seen = {}

    def model(**kw):
        seen.update(kw)
        return types.SimpleNamespace(last_hidden_state=kw["input_ids"].float().unsqueeze(-1).repeat(1, 1, 2))

    run = qe.make_runner(model, "cpu")
    out = run({"input_ids": torch.tensor([[5, 6, 7], [8, 9, 0]]),
               "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]])})
    assert seen["use_cache"] is False
    assert out.tolist() == [[7.0, 7.0], [9.0, 9.0]]


# ---- scoring ----


def test_group_views_and_mean_pool_by_hand():
    emb = torch.tensor([[2.0, 0], [0, 1], [0, 3], [1, 1]])
    owner = [1, 0, 1, 1]  # product 0 has view 1; product 1 has views 0, 2, 3 (order kept)
    v, m = qe.group_views(emb, owner, 2)
    assert m.tolist() == [[True, False, False], [True, True, True]]
    assert torch.allclose(v[1], F.normalize(emb[[0, 2, 3]], dim=-1))
    mp = qe.mean_pool(v, m)
    assert torch.allclose(mp[0], torch.tensor([0.0, 1.0]))
    ref = F.normalize(F.normalize(emb[[0, 2, 3]], dim=-1).mean(0), dim=0)
    assert torch.allclose(mp[1], ref)
    with pytest.raises(ValueError, match="no views"):
        qe.group_views(emb, owner, 3)


def test_maxsim_by_hand():
    # query: views (1,0) and (0,1); gallery 0: view (0.6,0.8); gallery 1: views (1,0) and (-1,0)
    qv, qm = qe.group_views(torch.tensor([[1.0, 0], [0, 1]]), [0, 0], 1)
    gv, gm = qe.group_views(torch.tensor([[0.6, 0.8], [1.0, 0], [-1.0, 0]]), [0, 1, 1], 2)
    s = qe.maxsim_block(qv, qm, gv, gm)
    assert torch.allclose(s, torch.tensor([[0.8, 1.0]]))


def _random_views(rng, n, d, vmax):
    counts = rng.integers(1, vmax + 1, n)
    owner = np.repeat(np.arange(n), counts)
    rng.shuffle(owner)
    return torch.tensor(rng.normal(size=(len(owner), d)), dtype=torch.float), owner.tolist()


def test_maxsim_chunked_equals_brute_force():
    rng = np.random.default_rng(0)
    nq, ng, d = 23, 17, 6
    qemb, qown = _random_views(rng, nq, d, 5)
    gemb, gown = _random_views(rng, ng, d, 5)
    qn, gn = F.normalize(qemb, dim=-1), F.normalize(gemb, dim=-1)
    full = np.full((nq, ng), -np.inf, dtype=np.float32)
    for i, a in enumerate(qown):
        for j, b in enumerate(gown):
            full[a, b] = max(full[a, b], float(qn[i] @ gn[j]))
    gt, src = rng.integers(0, ng, nq), rng.integers(0, ng, nq)
    qv, qm = qe.group_views(qemb, qown, nq)
    gv, gm = qe.group_views(gemb, gown, ng)
    ranks, recall = qe.rank_scores(qe.maxsim_scorer(qv, qm, gv, gm), nq, gt, src, chunk=4)
    ref, ref_recall = qe.rank_scores(lambda i0, i1: full[i0:i1].copy(), nq, gt, src, chunk=nq)
    assert np.allclose(qe.maxsim_scorer(qv, qm, gv, gm)(0, nq), full, atol=1e-6)
    for k in ("rank_keep", "rank_excl", "rank_src", "top_idx"):
        assert np.array_equal(ranks[k], ref[k]), k
    assert recall == ref_recall
    for k in (1, 5, 10):  # Recall from top-k equals Recall from the ranks
        assert recall[f"R@{k}"] == round(100 * float((ranks["rank_keep"] <= k).mean()), 2)
        assert recall[f"R@{k}_src_excluded"] == round(100 * float((ranks["rank_excl"] <= k).mean()), 2)


def test_cosine_ranks_and_dump_equal_save_dump(tmp_path):
    rng = np.random.default_rng(1)
    q, g = rng.normal(size=(40, 8)).astype(np.float32), rng.normal(size=(25, 8)).astype(np.float32)
    gt, src = rng.integers(0, 25, 40), rng.integers(0, 25, 40)
    keys = [f"d/{i}" for i in range(25)]
    sids, tids = [f"s{i}" for i in range(40)], [f"t{i}" for i in range(40)]
    ranks, recall = qe.rank_scores(qe.cosine_scorer(q, g), 40, gt, src, chunk=7)
    ref = rank_queries(q, g, gt, src)
    assert all(np.array_equal(ranks[k], ref[k]) for k in ref)
    qe.write_dump(tmp_path / "a" / "d.npz", q, g, keys, sids, tids, gt, src, ranks, {"recall": recall})
    save_dump(tmp_path / "b" / "d.npz", q, g, keys, sids, tids, gt, src, meta={"recall": recall})
    a, b = load_dump(tmp_path / "a" / "d.npz"), load_dump(tmp_path / "b" / "d.npz")
    assert set(a) == set(b)
    for k in a:
        if k == "meta":
            assert a[k] == b[k]
        else:
            assert a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]), k


# ---- end to end with a fake encoder ----


@pytest.fixture
def val_data(tmp_path, make_images):
    """f200k val: a -> b ("to b"), c -> e ("to e"); products a, b, c, e with 2-3 views."""
    ann = tmp_path / "ann"
    ann.mkdir()
    rows = [{"source_id": "a", "target_id": "b", "dataset": "f200k", "views_involved": ["front"],
             "modification_text_short": "to b"},
            {"source_id": "c", "target_id": "e", "dataset": "f200k", "views_involved": ["front"],
             "modification_text_short": "to e"}]
    (ann / "val_triplets.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    for pid, n in (("a", 2), ("b", 3), ("c", 2), ("e", 3)):
        make_images(f"images/f200k/{pid}", n, size=(8, 8))
    return ann, tmp_path / "images"


AXES = {"a": 0, "b": 1, "c": 2, "e": 3}


def fake_run(calls=None):
    """build = identity, so run gets (convs, image paths). Products sit on their own axis (views
    differ slightly); query a->b lands on b, query c->e on its source c, then e."""
    def run(batch):
        convs, paths = batch
        if calls is not None:
            calls.append(len(convs))
        out = []
        for conv, ps in zip(convs, paths):
            v = torch.zeros(8)
            for p in ps:
                pid, view = Path(p).parent.name, int(Path(p).stem)
                text = next((c["text"] for c in conv[1]["content"] if c["type"] == "text"), None)
                if text is None:
                    v[AXES[pid]] += 1.0
                elif text == "to b":
                    v[AXES["b"]] += 1.0
                    v[AXES[pid]] += 0.1
                else:
                    v[AXES[pid]] += 1.0
                    v[AXES["e"]] += 0.5
                v[6 + view % 2] += 0.05
            out.append(v.to(torch.bfloat16))
        return torch.stack(out)

    return run


def _identity_build(convs, imgs):
    return convs, imgs


META = {"query_instruction": qe.QUERY_INSTRUCTION, "tool": "test"}


@pytest.mark.parametrize("mode", ["joint", "perview"])
def test_run_dataset_writes_report_readable_dumps(val_data, tmp_path, monkeypatch, mode):
    from procir_train.retrieval_eval import val_gallery_and_queries

    monkeypatch.setattr(qe, "load_images", lambda paths: list(paths))
    products, queries = val_gallery_and_queries(*val_data, "f200k")
    out = tmp_path / "out"
    res = qe.run_dataset(out, mode, "f200k", products, queries, _identity_build, fake_run(), 3, META)
    assert set(res) == set(qe.MODES[mode])
    for strategy in qe.MODES[mode]:
        d = load_dump(out / strategy / "f200k.npz")
        assert d["gallery_keys"].tolist() == ["f200k/a", "f200k/b", "f200k/c", "f200k/e"]
        assert d["rank_keep"].tolist() == [1, 2] and d["rank_excl"].tolist() == [1, 1]
        assert d["rank_src"].tolist() == [2, 1]
        assert d["meta"]["recall"]["R@1"] == 50.0 and d["meta"]["recall"]["R@1_src_excluded"] == 100.0
        assert d["meta"]["strategy"] == strategy and d["meta"]["query_instruction"] == qe.QUERY_INSTRUCTION
        assert "seconds_gallery" in d["meta"] and "seconds_queries" in d["meta"]
        dumps = report.load_model(out / strategy)
        report.check_recall(strategy, dumps)  # Recall from ranks == Recall in meta
        n_views = 10 if mode == "perview" else 4
        assert d["meta"]["n_gallery_inputs"] == n_views
    assert set(json.loads((out / qe.MODES[mode][0] / "f200k.json").read_text())) >= {"n_queries", "k_store"}


def test_embed_resumes_from_partial_and_full_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(qe, "load_images", lambda paths: list(paths))
    items = [([f"x/p{i}/{i}.jpg"], qe.messages("I.", 1)) for i in range(7)]
    axes = {f"p{i}": i for i in range(7)}

    def run(batch):
        _, paths = batch
        return torch.stack([F.one_hot(torch.tensor(axes[Path(p[0]).parent.name]), 8).float() for p in paths])

    ref, _ = qe.embed(items, _identity_build, run, 2, tmp_path / "ref.pt")

    calls = []

    def crashing(batch):
        if len(calls) == 2:
            raise RuntimeError("crash")
        calls.append(1)
        return run(batch)

    cache = tmp_path / "c" / "x.pt"
    with pytest.raises(RuntimeError, match="crash"):
        qe.embed(items, _identity_build, crashing, 2, cache, save_every=1)
    partial = cache.with_name("x.partial.pt")
    assert partial.exists() and torch.load(partial)["n_done"] == 4

    seen = []
    emb, _ = qe.embed(items, _identity_build, lambda b: (seen.append(len(b[1])), run(b))[1], 2, cache, save_every=1)
    assert seen == [2, 1] and torch.equal(emb, ref)  # only items 4..6 were encoded
    assert cache.exists() and not partial.exists()

    emb2, _ = qe.embed(items, _identity_build, lambda b: pytest.fail("should load the cache"), 2, cache)
    assert torch.equal(emb2, ref)
    with pytest.raises(ValueError, match="other inputs"):
        qe.embed(items[:3], _identity_build, run, 2, cache)


def test_main_skips_done_datasets_and_writes_eval(val_data, tmp_path, monkeypatch):
    ann, images = val_data
    out = tmp_path / "out"
    monkeypatch.setattr(qe, "load_images", lambda paths: list(paths))
    monkeypatch.setattr(qe, "load_model", lambda name, device: (None, None))
    monkeypatch.setattr(qe, "processor_inputs", lambda proc, convs, imgs, px: (convs, imgs))
    calls = []
    monkeypatch.setattr(qe, "make_runner", lambda model, device: fake_run(calls))
    argv = ["--data_dir", str(ann), "--image_root", str(images), "--out_dir", str(out),
            "--datasets", "f200k", "--batch_size", "2"]
    qe.main(argv)
    assert calls
    for s in ("joint", "meanpool", "maxsim"):
        ev = json.loads((out / s / "eval.json").read_text())
        assert ev["f200k"]["n_queries"] == 2 and ev["f200k"]["R@5"] == 100.0
        assert ev["f200k"]["R@1"] == 50.0 and ev["f200k"]["R@1_src_excluded"] == 100.0
    assert (out / "cache" / "joint_f200k_gallery.pt").exists()
    assert (out / "cache" / "perview_f200k_queries.pt").exists()

    monkeypatch.setattr(qe, "load_model", lambda name, device: pytest.fail("everything is done"))
    qe.main(argv)  # re-run: all dumps exist, the model is not loaded
    (out / "maxsim" / "f200k.npz").unlink()  # perview unfinished: it is redone from the cache
    monkeypatch.setattr(qe, "load_model", lambda name, device: (None, None))
    calls.clear()
    qe.main(argv)
    assert calls == [] and (out / "maxsim" / "f200k.npz").exists()


# ---- real processor (downloads the Qwen3-VL-Embedding-2B processor files, no weights) ----


@pytest.mark.hf
def test_real_processor_prompt_and_pixels():
    from PIL import Image
    from transformers import AutoProcessor

    try:
        proc = AutoProcessor.from_pretrained(qe.MODEL)
    except (OSError, ValueError) as e:
        pytest.skip(f"Qwen3-VL-Embedding processor unavailable: {e}")
    proc.tokenizer.padding_side = "right"
    q = qe.messages(qe.QUERY_INSTRUCTION, 2, "make it red")
    text = proc.apply_chat_template(q, add_generation_prompt=True, tokenize=False)
    assert text.startswith("<|im_start|>system\n" + qe.QUERY_INSTRUCTION + "<|im_end|>")
    assert text.count("<|vision_start|><|image_pad|><|vision_end|>") == 2
    assert text.index("<|vision_end|>") < text.index("make it red")
    assert text.endswith("<|im_start|>assistant\n")

    big, small = Image.new("RGB", (1024, 768)), Image.new("RGB", (60, 80))
    d = qe.messages(qe.DOC_INSTRUCTION, 1)
    inputs = qe.processor_inputs(proc, [q, d], [[big, small], [big]], (128 * 128, 512 * 512))
    grid = inputs["image_grid_thw"]
    pixels = (grid[:, 1] * grid[:, 2] * 16 * 16).tolist()
    assert len(pixels) == 3 and all(128 * 128 <= p <= 512 * 512 for p in pixels)
    assert "mm_token_type_ids" in inputs
    n_pad = (inputs["input_ids"] == proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")).sum(dim=1)
    per_conv = (grid.prod(dim=1) // 4).tolist()
    assert n_pad.tolist() == [per_conv[0] + per_conv[1], per_conv[2]]
    mask = inputs["attention_mask"]
    assert mask[0].all() and not mask[1].all() and mask[1, 0] == 1  # right padding
