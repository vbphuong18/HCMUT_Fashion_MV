import pytest
import torch

from procir_train.config import TrainConfig
from procir_train.dataset import StepBatchSampler, TripletTrainDataset, collate, split_chunks


def test_sampler_steps_epochs_and_resume():
    s = StepBatchSampler(n_records=10, batch_size=4, total_steps=5, seed=0)
    batches = list(s)
    assert len(batches) == 5 and s.steps_per_epoch == 2
    assert [b[0][1] for b in batches] == [0, 1, 2, 3, 4]
    first_epoch = [i for b in batches[:2] for i, _ in b]
    assert len(set(first_epoch)) == 8
    resumed = list(StepBatchSampler(10, 4, 5, seed=0, start_step=3))
    assert resumed == batches[3:]


def test_split_chunks():
    batch = {"query": list(range(8)), "target_key": list("abcdefgh")}
    chunks = split_chunks(batch, 4)
    assert [c["query"] for c in chunks] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert [c["target_key"] for c in chunks] == [list("abcd"), list("efgh")]


def test_sampler_len_never_negative():
    assert len(StepBatchSampler(10, 4, 5, seed=0, start_step=3)) == 2
    assert len(StepBatchSampler(10, 4, 5, seed=0, start_step=9)) == 0


@pytest.fixture
def tiny(make_images):
    s = make_images("deepfashion/A/1", 3)
    t = make_images("deepfashion/B/1", 2)
    rec = {"dataset": "deepfashion", "source_id": "A/1", "target_id": "B/1",
           "source_key": "deepfashion/A/1", "target_key": "deepfashion/B/1",
           "source_images": sorted(str(p) for p in s.iterdir()),
           "target_images": sorted(str(p) for p in t.iterdir()),
           "mod_short": "make it red", "mod_long": "make the whole dress red and add a back zipper"}
    caps = {"deepfashion/A/1": {"long": "a blue dress with a round neckline and short sleeves",
                                "short": "blue dress"},
            "deepfashion/B/1": {"long": "a red dress with a back zipper and long sleeves",
                                "short": "red dress"}}
    return [rec], caps


@pytest.mark.hf
def test_item_is_deterministic_and_has_variant_keys(tiny, processor):
    recs, caps = tiny
    ds = TripletTrainDataset(recs, caps, processor, TrainConfig(), total_steps=10)
    a, b = ds[(0, 3)], ds[(0, 3)]
    assert torch.equal(a["query"]["input_ids"], b["query"]["input_ids"])
    assert set(a) == {"query", "doc", "source_key", "target_key", "src_caption", "tgt_caption"}
    single = TripletTrainDataset(recs, caps, processor, TrainConfig(multi_turn=False), 10)[(0, 3)]
    assert "src_caption" not in single and "tgt_caption" in single
    no_align = TripletTrainDataset(recs, caps, processor, TrainConfig(align=False), 10)[(0, 3)]
    assert "tgt_caption" not in no_align


@pytest.mark.hf
def test_cot_injects_caption_early_and_vanishes_after_half(tiny, processor):
    recs, caps = tiny
    cot = TripletTrainDataset(recs, caps, processor, TrainConfig(cot=True), total_steps=10)
    plain = TripletTrainDataset(recs, caps, processor, TrainConfig(cot=False), total_steps=10)
    early = processor.tokenizer.decode(cot[(0, 0)]["doc"]["input_ids"][0])
    assert "back zipper and long sleeves" in early
    late_cot, late_plain = cot[(0, 5)], plain[(0, 5)]
    assert torch.equal(late_cot["doc"]["input_ids"], late_plain["doc"]["input_ids"])
    assert torch.equal(late_cot["query"]["input_ids"], late_plain["query"]["input_ids"])


@pytest.mark.hf
def test_collate_groups_by_key(tiny, processor):
    recs, caps = tiny
    ds = TripletTrainDataset(recs, caps, processor, TrainConfig(), total_steps=10)
    batch = collate([ds[(0, 0)], ds[(0, 1)]])
    assert len(batch["query"]) == 2 and batch["target_key"] == ["deepfashion/B/1"] * 2


def _dec(processor, x):
    return processor.tokenizer.decode(x["input_ids"][0])


@pytest.mark.hf
def test_mod_text_is_drawn_short_and_long(tiny, processor):
    recs, caps = tiny
    ds = TripletTrainDataset(recs, caps, processor, TrainConfig(cot=False), total_steps=40)
    texts = [_dec(processor, ds[(0, s)]["query"]) for s in range(30)]
    assert any("make it red" in t and "back zipper" not in t for t in texts)
    assert any("add a back zipper" in t for t in texts)


@pytest.mark.hf
def test_different_steps_give_different_samples(tiny, processor):
    recs, caps = tiny
    ds = TripletTrainDataset(recs, caps, processor, TrainConfig(cot=False), total_steps=40)
    ids = [ds[(0, s)]["query"]["input_ids"].tolist() for s in range(10)]
    assert len({str(i) for i in ids}) > 1


@pytest.mark.hf
def test_cot_puts_source_caption_in_first_think_of_query(tiny, processor):
    recs, caps = tiny
    ds = TripletTrainDataset(recs, caps, processor, TrainConfig(cot=True), total_steps=10)
    q = _dec(processor, ds[(0, 0)]["query"])
    a = q.index("<think>")
    b = q.index("</think>", a)
    assert "round neckline and short sleeves" in q[a:b]
    a2 = q.index("<think>", b)
    b2 = q.index("</think>", a2)
    assert q[a2 + len("<think>"):b2].strip() == ""


@pytest.mark.hf
def test_cot_does_not_change_other_random_draws(tiny, processor):
    recs, caps = tiny
    on = TripletTrainDataset(recs, caps, processor, TrainConfig(cot=True), total_steps=10)
    off = TripletTrainDataset(recs, caps, processor, TrainConfig(cot=False), total_steps=10)
    for step in range(5):
        a, b = on[(0, step)], off[(0, step)]
        for k in ("tgt_caption", "src_caption"):
            assert torch.equal(a[k]["input_ids"], b[k]["input_ids"])
        qa, qb = _dec(processor, a["query"]), _dec(processor, b["query"])
        assert ("make it red" in qa) == ("make it red" in qb)
        assert ("add a back zipper" in qa) == ("add a back zipper" in qb)


@pytest.mark.hf
def test_empty_texts_fall_back_to_other_variant(tiny, processor):
    recs, caps = tiny
    recs = [dict(recs[0], mod_short="")]
    caps = {k: dict(v, long="") for k, v in caps.items()}
    ds = TripletTrainDataset(recs, caps, processor, TrainConfig(cot=True), total_steps=10)
    for step in range(8):
        item = ds[(0, step)]
        assert "add a back zipper" in _dec(processor, item["query"])
        assert "blue dress" in _dec(processor, item["src_caption"])
        assert item["tgt_caption"]["input_ids"].shape[1] > 0
    assert "red dress" in _dec(processor, ds[(0, 0)]["doc"])
