"""ProCIR encoder: Qwen3.5 hidden state at <emb_all>, LoRA on the language model, and a trainable
input vector for <emb_all> (a forward hook on the embedding layer, so the 248k-row matrix stays frozen)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .prompts import EMB_TOKEN, ensure_emb_token


def pad_batch(inputs_list, device):
    ids = [x["input_ids"].squeeze(0) for x in inputs_list]
    length = max(len(t) for t in ids)
    input_ids = torch.zeros((len(ids), length), dtype=ids[0].dtype)
    attention_mask = torch.zeros((len(ids), length), dtype=torch.long)
    for i, t in enumerate(ids):
        input_ids[i, :len(t)] = t
        attention_mask[i, :len(t)] = 1
    out = {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}
    if "pixel_values" in inputs_list[0]:
        out["pixel_values"] = torch.cat([x["pixel_values"] for x in inputs_list]).to(device, torch.bfloat16)
        out["image_grid_thw"] = torch.cat([x["image_grid_thw"] for x in inputs_list]).to(device)
    if inputs_list[0].get("mm_token_type_ids") is not None:
        first = inputs_list[0]["mm_token_type_ids"]
        mm = torch.zeros((len(ids), length), dtype=first.dtype)
        for i, x in enumerate(inputs_list):
            t = x["mm_token_type_ids"].squeeze(0)
            mm[i, :len(t)] = t
        out["mm_token_type_ids"] = mm.to(device)
    return out


def language_linear_names(model):
    names = [n for n, m in model.named_modules()
             if isinstance(m, nn.Linear) and ".language_model." in f".{n}." and "lm_head" not in n]
    if not names:
        raise ValueError("no nn.Linear found under language_model")
    return names


def assert_no_dropout(model):
    for name, m in model.named_modules():
        if isinstance(m, nn.Dropout) and m.p > 0:
            raise ValueError(f"dropout p={m.p} at {name}: GradCache needs a deterministic forward")


class ProCIREncoder(nn.Module):
    def __init__(self, vlm, emb_token_id, emb_init):
        super().__init__()
        self.vlm = vlm
        self.emb_token_id = emb_token_id
        self.emb_vector = nn.Parameter(emb_init.detach().float().clone())
        self._hook = vlm.get_input_embeddings().register_forward_hook(self._override_emb)
        object.__setattr__(self, "peft_model", None)  # not a submodule: avoids duplicate parameters

    def _override_emb(self, module, inputs, output):
        mask = (inputs[0] == self.emb_token_id).unsqueeze(-1)
        return torch.where(mask, self.emb_vector.to(output.dtype), output)

    @property
    def device(self):
        return self.emb_vector.device

    def _hidden(self, inputs_list):
        batch = pad_batch(inputs_list, self.device)
        inner = self.vlm.model
        inner.rope_deltas = None
        out = inner(**batch)
        inner.rope_deltas = None
        return out.last_hidden_state, batch["input_ids"]

    def _positions(self, input_ids):
        return [(row == self.emb_token_id).nonzero(as_tuple=True)[0].tolist() for row in input_ids]

    def encode_single(self, inputs_list):
        hidden, ids = self._hidden(inputs_list)
        rows = []
        for i, pos in enumerate(self._positions(ids)):
            if not pos:
                raise ValueError(f"no {EMB_TOKEN} in sample {i}")
            rows.append(hidden[i, pos[-1]])
        return torch.stack(rows)

    def encode_multiturn(self, inputs_list):
        hidden, ids = self._hidden(inputs_list)
        s, q = [], []
        for i, pos in enumerate(self._positions(ids)):
            if len(pos) < 2:
                raise ValueError(f"expected two {EMB_TOKEN} in sample {i}, found {len(pos)}")
            s.append(hidden[i, pos[0]])
            q.append(hidden[i, pos[-1]])
        return torch.stack(s), torch.stack(q)

    def encode(self, chunk, cfg):
        reps = {}
        if cfg.multi_turn:
            reps["s"], reps["q"] = self.encode_multiturn(chunk["query"])
        else:
            reps["q"] = self.encode_single(chunk["query"])
        reps["d"] = self.encode_single(chunk["doc"])
        if cfg.align:
            reps["t_tgt"] = self.encode_single(chunk["tgt_caption"])
            if cfg.multi_turn:
                reps["t_src"] = self.encode_single(chunk["src_caption"])
        return reps


def load_encoder(cfg, device):
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(cfg.base_model)
    processor.tokenizer.padding_side = "right"
    emb_id = ensure_emb_token(processor.tokenizer)

    vlm = Qwen3_5ForConditionalGeneration.from_pretrained(cfg.base_model, dtype=torch.bfloat16)
    weight = vlm.get_input_embeddings().weight
    if emb_id >= weight.shape[0]:
        raise ValueError(f"{EMB_TOKEN} id {emb_id} exceeds embedding rows {weight.shape[0]}")
    text_cfg = getattr(vlm.config, "text_config", vlm.config)
    if getattr(text_cfg, "attention_dropout", 0.0):
        raise ValueError("attention_dropout must be 0 for GradCache")
    vlm.config.use_cache = False
    text_cfg.use_cache = False
    vlm.requires_grad_(False)

    # Assumption 3: initialise <emb_all> at the mean of the real vocabulary rows.
    if emb_id != len(processor.tokenizer) - 1:
        raise ValueError(f"{EMB_TOKEN} must be the newest token: id {emb_id}, tokenizer size {len(processor.tokenizer)}")
    emb_init = weight[:emb_id].float().mean(dim=0)
    encoder = ProCIREncoder(vlm, emb_id, emb_init)

    if cfg.lora:
        from peft import LoraConfig, get_peft_model

        lora_cfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0, bias="none",
                              target_modules=language_linear_names(vlm))
        # get_peft_model injects LoRA layers into `vlm` in place; encoder.vlm.model now carries them.
        object.__setattr__(encoder, "peft_model", get_peft_model(vlm, lora_cfg))
    else:
        vlm.model.language_model.requires_grad_(True)
        # Keep the 248k x 1024 table (tied to lm_head) frozen: emb_vector already learns <emb_all>,
        # and AdamW state for the full table would cost ~4 GB and skew the LoRA/full pilot.
        vlm.get_input_embeddings().requires_grad_(False)

    if cfg.gradient_checkpointing:
        # gradient_checkpointing_enable adds its own make_inputs_require_grads embedding hook; harmless
        # alongside the <emb_all> override hook.
        vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert_no_dropout(vlm)
    return encoder.to(device), processor
