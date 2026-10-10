"""For labelled Fashion200K images: which detection class does each category (and sub-category) crop to?"""
import io
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src/data")
import crop_f200k as c  # noqa: E402


def lines(z, prefix):
    for name in z.namelist():
        if name.startswith(prefix) and name.endswith(".txt"):
            for line in io.TextIOWrapper(z.open(name), encoding="utf-8"):
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) > 1:
                    yield parts


with zipfile.ZipFile(c.ZIP) as z:
    label = {Path(p[0]).stem: float(p[1]) for p in lines(z, "fashion-200k/labels/")}
    by_cat = defaultdict(Counter)
    by_sub = defaultdict(Counter)
    rule = Counter()
    for path, *ds in lines(z, "fashion-200k/detection/"):
        p = Path(path)
        if p.stem not in label:
            continue
        parsed = [d.split("_") for d in ds]
        cls = lambda d: "_".join(d[:-5])
        chosen = min(parsed, key=lambda d: abs(float(d[-5]) - label[p.stem]))
        by_cat[p.parts[1]][cls(chosen)] += 1
        by_sub[(p.parts[1], p.parts[2])][cls(chosen)] += 1
        # candidate rule: best-scored detection among the classes the category uses
        allowed = {"dresses": {"dress"}, "jackets": {"outerwear"}, "pants": {"pants", "shorts"},
                   "skirts": {"skirt"}, "tops": {"top"}}[p.parts[1]]
        cand = [d for d in parsed if cls(d) in allowed]
        rule[bool(cand) and max(cand, key=lambda d: float(d[-5])) is chosen] += 1
for cat, cnt in sorted(by_cat.items()):
    print(cat, cnt.most_common(4))
print("sub-categories whose main class is not the category's:")
for (cat, sub), cnt in sorted(by_sub.items()):
    top = cnt.most_common(1)[0][0]
    if top != by_cat[cat].most_common(1)[0][0]:
        print("  ", cat, sub, cnt.most_common(3))
print("rule 'best-scored detection of the category's classes' reproduces the labelled box:", dict(rule))
