"""Re-crop the Fashion200K images whose box changed with the category-class rule; list them for a patch zip."""
import io
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

sys.path.insert(0, "src/data")
import crop_f200k as c  # noqa: E402

SRC, DST = Path("data/images_official_hr/f200k_source"), Path("data/images_official_hr/f200k")
MIRROR = Path("data/images/f200k")

new = c.load_boxes(c.ZIP)
old = {}
with zipfile.ZipFile(c.ZIP) as z:  # the previous rule for unlabelled images: the top-scored detection
    label = {Path(p[0]).stem for p in c._lines(z, "fashion-200k/labels/")}
    for path, *dets in c._lines(z, "fashion-200k/detection/"):
        stem = Path(path).stem
        if stem not in label:
            old[stem] = tuple(min(1.0, max(0.0, float(v))) for v in dets[0].split("_")[-4:])
changed_stems = {s for s, b in old.items() if new[s] != b}
print("unlabelled images:", len(old), "| box changed:", len(changed_stems))


def one(pid):
    out = []
    for f in sorted((SRC / pid).iterdir()):
        if f.stem not in changed_stems or f.suffix.lower() not in c.IMAGE_EXTS:
            continue
        if c._is_released(f, MIRROR / pid):
            continue
        dst = DST / pid / f.name
        with Image.open(f) as im:
            tmp = dst.with_name(dst.name + ".part")
            im.convert("RGB").crop(c.pixel_box(new[f.stem], im.size)).save(tmp, "JPEG", quality=95)
        tmp.replace(dst)
        out.append(f"{pid}/{f.name}")
    return out


products = sorted(p.name for p in SRC.iterdir() if p.is_dir())
changed = []
with ThreadPoolExecutor(8) as ex:
    for n, files in enumerate(ex.map(one, products), 1):
        changed += files
        if n % 20000 == 0:
            print(n, "products,", len(changed), "files re-cropped", flush=True)
Path("data/f200k_recrop_changed.json").write_text(json.dumps(changed), encoding="utf-8")
print("re-cropped", len(changed), "files in", len({f.split('/')[0] for f in changed}), "products")
