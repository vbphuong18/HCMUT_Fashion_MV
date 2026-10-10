"""Which detection does the Fashion200K release crop to? Check labelled images, then count unlabelled ones."""
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src/data")
import crop_f200k as c  # noqa: E402

EXPECTED = {"dresses": {"dress"}, "jackets": {"outerwear"}, "pants": {"pants", "shorts"},
            "skirts": {"skirt"}, "tops": {"top"}}


def lines(z, prefix):
    for name in z.namelist():
        if name.startswith(prefix) and name.endswith(".txt"):
            for line in io.TextIOWrapper(z.open(name), encoding="utf-8"):
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) > 1:
                    yield name, parts


split_products = {"train": set(), "val": set()}
for split in split_products:
    for l in open(f"data/FashionMV_hf/data/data/{split}_triplets.jsonl", encoding="utf-8"):
        t = json.loads(l)
        if t["dataset"] == "f200k":
            split_products[split].update([str(t["source_id"]), str(t["target_id"])])

with zipfile.ZipFile(c.ZIP) as z:
    label = {}
    label_split = Counter()
    for name, p in lines(z, "fashion-200k/labels/"):
        label[Path(p[0]).stem] = float(p[1])
        label_split["test" if "_test_" in name else "train"] += 1
    print("label lines:", dict(label_split))
    labelled = Counter()      # class of the label-score detection vs category
    exact = Counter()
    rank = Counter()          # is the label-score detection the top-scored one?
    cat_rank = Counter()      # for labelled images: is the best detection of the category class the labelled one?
    unl = {"train": Counter(), "val": Counter()}
    images = {"train": Counter(), "val": Counter()}
    for _, (path, *ds) in lines(z, "fashion-200k/detection/"):
        p = Path(path)
        cat, pid, stem = p.parts[1], p.parent.name, p.stem
        parsed = [d.split("_") for d in ds]
        cls = lambda d: "_".join(d[:-5])
        splits = [s for s, ids in split_products.items() if pid in ids]
        for s in splits:
            images[s]["with label" if stem in label else "without label"] += 1
        if stem in label:
            chosen = min(parsed, key=lambda d: abs(float(d[-5]) - label[stem]))
            exact[abs(float(chosen[-5]) - label[stem]) < 0.005] += 1
            labelled[cls(chosen) in EXPECTED.get(cat, set())] += 1
            rank[parsed.index(chosen) == 0] += 1
            same_class = [d for d in parsed if cls(d) in EXPECTED.get(cat, set())]
            cat_rank[bool(same_class) and max(same_class, key=lambda d: float(d[-5])) is chosen] += 1
        else:
            for s in splits:
                top_ok = cls(parsed[0]) in EXPECTED.get(cat, set())
                has = any(cls(d) in EXPECTED.get(cat, set()) for d in parsed)
                unl[s]["top detection is the category class" if top_ok else
                       ("category class exists further down" if has else "no detection of the category class")] += 1

print("labelled images: label score found among detections:", dict(exact))
print("labelled images: labelled box is of the product's category class:", dict(labelled))
print("labelled images: labelled box is the top-scored detection:", dict(rank))
print("labelled images: labelled box == best-scored detection of the category class:", dict(cat_rank))
for s in ("train", "val"):
    print(f"FashionMV {s} f200k images in detection file:", dict(images[s]), "| unlabelled:", dict(unl[s]))
