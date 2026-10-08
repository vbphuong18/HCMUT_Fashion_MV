"""Step-aware training dataset, deterministic batch sampler, and chunking for GradCache.

Every random choice has its own named stream random.Random(f"{seed}-{step}-{idx}-{name}") with
name in {mod, cot-tgt, cot-src, cap-tgt, cap-src}, drawn inside the worker. A sample therefore
depends only on (seed, step, idx): resume replays identical batches, toggling CoT does not shift
other draws, and the CoT keep-ratio follows the global step without sharing state across
DataLoader workers."""
import random

from PIL import Image
from torch.utils.data import Dataset, Sampler

from .cot import keep_ratio, subsample_tokens, truncate_tokens
from .prompts import (caption_text, doc_text, inject_think, multiturn_query_text, process_text,
                      process_visual, singleturn_query_text)


def _pick(rng, first, second):
    """Choose first/second with probability 1/2, falling back to the other if the choice is empty."""
    a, b = (first, second) if rng.random() < 0.5 else (second, first)
    return a or b or ""


def load_images(paths):
    imgs = []
    for p in paths:
        with Image.open(p) as im:
            imgs.append(im.convert("RGB"))
    return imgs


class TripletTrainDataset(Dataset):
    def __init__(self, records, captions, processor, cfg, total_steps):
        self.records = records
        self.captions = captions
        self.processor = processor
        self.cfg = cfg
        self.total_steps = total_steps

    def __len__(self):
        return len(self.records)

    def _caption_inputs(self, key, rng):
        cap = self.captions[key]
        text = _pick(rng, cap.get("long"), cap.get("short"))
        text = truncate_tokens(text, self.processor.tokenizer, self.cfg.max_caption_tokens)
        return process_text(self.processor, caption_text(self.processor, text))

    def _think(self, key, ratio, rng):
        cap = self.captions[key]
        long_cap = truncate_tokens(cap.get("long") or cap.get("short") or "", self.processor.tokenizer,
                                   self.cfg.max_caption_tokens)
        return subsample_tokens(long_cap, self.processor.tokenizer, ratio, rng)

    def __getitem__(self, item):
        idx, step = item
        cfg, proc = self.cfg, self.processor
        r = self.records[idx]
        base = f"{cfg.seed}-{step}-{idx}"  # independent streams: toggling CoT/align never shifts other draws
        rng = lambda name: random.Random(f"{base}-{name}")  # noqa: E731
        src_imgs, tgt_imgs = load_images(r["source_images"]), load_images(r["target_images"])

        mod = _pick(rng("mod"), r["mod_short"], r["mod_long"])
        if cfg.multi_turn:
            q_text = multiturn_query_text(proc, len(src_imgs), mod)
        else:
            q_text = singleturn_query_text(proc, len(src_imgs), mod)
        d_text = doc_text(proc, len(tgt_imgs))

        if cfg.cot:
            ratio = keep_ratio(step, self.total_steps)
            d_text = inject_think(d_text, 0, self._think(r["target_key"], ratio, rng("cot-tgt")))
            if cfg.multi_turn:  # Assumption 2: single-turn queries have no perception turn
                q_text = inject_think(q_text, 0, self._think(r["source_key"], ratio, rng("cot-src")))

        px = (cfg.image_min_pixels, cfg.image_max_pixels)
        out = {
            "query": process_visual(proc, q_text, src_imgs, *px),
            "doc": process_visual(proc, d_text, tgt_imgs, *px),
            "source_key": r["source_key"],
            "target_key": r["target_key"],
        }
        if cfg.align:
            out["tgt_caption"] = self._caption_inputs(r["target_key"], rng("cap-tgt"))
            if cfg.multi_turn:
                out["src_caption"] = self._caption_inputs(r["source_key"], rng("cap-src"))
        return out


class StepBatchSampler(Sampler):
    """Yields [(idx, step), ...] per step; one fresh permutation per epoch; drops the last partial batch."""

    def __init__(self, n_records, batch_size, total_steps, seed, start_step=0):
        if n_records < batch_size:
            raise ValueError(f"need at least {batch_size} records, got {n_records}")
        self.n, self.bs, self.total, self.seed, self.start = n_records, batch_size, total_steps, seed, start_step
        self.steps_per_epoch = n_records // batch_size
        self._epoch, self._perm = None, None

    def _perm_for(self, epoch):
        if epoch != self._epoch:
            perm = list(range(self.n))
            random.Random(f"{self.seed}-epoch-{epoch}").shuffle(perm)
            self._epoch, self._perm = epoch, perm
        return self._perm

    def __iter__(self):
        for step in range(self.start, self.total):
            epoch, b = divmod(step, self.steps_per_epoch)
            perm = self._perm_for(epoch)
            yield [(i, step) for i in perm[b * self.bs:(b + 1) * self.bs]]

    def __len__(self):
        return max(0, self.total - self.start)


def collate(samples):
    return {k: [s[k] for s in samples] for k in samples[0]}


def split_chunks(batch, chunk_size):
    n = len(batch["query"])
    return [{k: v[i:i + chunk_size] for k, v in batch.items()} for i in range(0, n, chunk_size)]
