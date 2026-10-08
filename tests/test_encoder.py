from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from procir_train.config import TrainConfig
from procir_train.encoder import ProCIREncoder, assert_no_dropout, language_linear_names, pad_batch


def test_pad_batch_right_pads_and_concatenates_pixels():
    a = {"input_ids": torch.tensor([[5, 6, 7]]), "pixel_values": torch.ones(4, 3),
         "image_grid_thw": torch.tensor([[1, 2, 2]]), "mm_token_type_ids": torch.tensor([[0, 1, 0]])}
    b = {"input_ids": torch.tensor([[8]]), "pixel_values": torch.zeros(2, 3),
         "image_grid_thw": torch.tensor([[1, 1, 2]]), "mm_token_type_ids": torch.tensor([[1]])}
    out = pad_batch([a, b], torch.device("cpu"))
    assert out["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert out["pixel_values"].shape == (6, 3) and out["pixel_values"].dtype == torch.bfloat16
    assert out["image_grid_thw"].shape == (2, 3)
    assert out["mm_token_type_ids"].tolist() == [[0, 1, 0], [1, 0, 0]]


def test_pad_batch_text_only():
    out = pad_batch([{"input_ids": torch.tensor([[1, 2]])}], torch.device("cpu"))
    assert set(out) == {"input_ids", "attention_mask"}


def test_language_linear_names_skip_vision_and_head():
    m = nn.Module()
    m.model = nn.Module()
    m.model.language_model = nn.ModuleDict({"q_proj": nn.Linear(4, 4), "mlp": nn.Sequential(nn.Linear(4, 4))})
    m.model.visual = nn.ModuleDict({"proj": nn.Linear(4, 4)})
    m.lm_head = nn.Linear(4, 4)
    assert language_linear_names(m) == ["model.language_model.q_proj", "model.language_model.mlp.0"]


def test_assert_no_dropout():
    assert_no_dropout(nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.0)))
    with pytest.raises(ValueError, match="dropout"):
        assert_no_dropout(nn.Sequential(nn.Dropout(0.1)))


class ToyInner(nn.Module):
    def __init__(self, embed):
        super().__init__()
        self.embed = embed
        self.rope_deltas = None

    def forward(self, input_ids, attention_mask, **kwargs):
        return SimpleNamespace(last_hidden_state=self.embed(input_ids).cumsum(dim=1))


class ToyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(10, 3)
        self.model = ToyInner(self.embed)

    def get_input_embeddings(self):
        return self.embed


def _enc():
    return ProCIREncoder(ToyVLM(), emb_token_id=7, emb_init=torch.ones(3))


def test_emb_vector_replaces_token_and_gets_gradient():
    enc = _enc()
    out = enc.vlm.embed(torch.tensor([[1, 7, 2]]))
    assert torch.equal(out[0, 1], torch.ones(3))
    assert torch.equal(out[0, 0], enc.vlm.embed.weight[1])
    out.sum().backward()
    assert torch.equal(enc.emb_vector.grad, torch.ones(3))


def test_multiturn_reads_first_and_last_emb_token():
    enc = _enc()
    E = enc.vlm.embed.weight.detach()
    v = enc.emb_vector.detach()
    s, q = enc.encode_multiturn([{"input_ids": torch.tensor([[1, 7, 2, 7, 3]])}])
    assert torch.allclose(s[0], E[1] + v)
    assert torch.allclose(q[0], E[1] + v + E[2] + v)
    with pytest.raises(ValueError):
        enc.encode_single([{"input_ids": torch.tensor([[1, 2]])}])


def test_encode_returns_keys_for_variant():
    enc = _enc()
    x = [{"input_ids": torch.tensor([[1, 7, 2, 7]])}]
    chunk = {"query": x, "doc": x, "src_caption": x, "tgt_caption": x}
    assert set(enc.encode(chunk, TrainConfig())) == {"s", "q", "d", "t_src", "t_tgt"}
    assert set(enc.encode(chunk, TrainConfig(multi_turn=False))) == {"q", "d", "t_tgt"}
    assert set(enc.encode(chunk, TrainConfig(align=False))) == {"s", "q", "d"}
