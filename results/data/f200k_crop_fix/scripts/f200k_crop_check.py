"""Fashion200K val crops, before and after the category-class fix: photo with both boxes next to the new crop."""
import html
import json
import random
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, "src/data")
import crop_f200k as c  # noqa: E402

SRC, DST = Path("data/images_official_hr/f200k_source"), Path("data/images_official_hr/f200k")
OUT = Path("data/f200k_crop_check")
N, SEED, H = 100, 42, 230


def main():
    val = set()
    for l in open("data/FashionMV_hf/data/data/val_triplets.jsonl", encoding="utf-8"):
        t = json.loads(l)
        if t["dataset"] == "f200k":
            val.update([str(t["source_id"]), str(t["target_id"])])
    changed = set(json.loads(Path("data/f200k_recrop_changed.json").read_text(encoding="utf-8")))
    new = c.load_boxes(c.ZIP)
    old, cat, cls_old, cls_new = {}, {}, {}, {}
    with zipfile.ZipFile(c.ZIP) as z:
        for path, *dets in c._lines(z, "fashion-200k/detection/"):
            p = Path(path)
            if p.parent.name not in val:
                continue
            parsed = [d.split("_") for d in dets]
            old[p.stem] = tuple(min(1.0, max(0.0, float(v))) for v in parsed[0][-4:])
            cat[p.stem] = p.parts[1]
            cls_old[p.stem] = "_".join(parsed[0][:-5])
            cls_new[p.stem] = next(("_".join(d[:-5]) for d in parsed
                                    if tuple(min(1.0, max(0.0, float(v))) for v in d[-4:]) == new[p.stem]), "?")

    used, fixed = [], []
    for pid in sorted(val):
        for f in sorted((DST / pid).iterdir())[:5]:  # the 5 views evaluation uses
            used.append(f)
            if f"{pid}/{f.name}" in changed:
                fixed.append(f)
    stats = {"val_products": len(val), "val_images_used_in_eval": len(used),
             "val_images_whose_crop_was_wrong_and_is_now_fixed": len(fixed),
             "share": round(len(fixed) / len(used), 3),
             "val_products_with_at_least_one_fixed_image": len({f.parent.name for f in fixed}),
             "all_f200k_images_recropped_train_and_val": len(changed)}
    print(json.dumps(stats, indent=1))

    rng = random.Random(SEED)
    sample = rng.sample(fixed, N)
    OUT.mkdir(parents=True, exist_ok=True)
    for old_file in list(OUT.glob("sheet_*.jpg")):
        old_file.unlink()
    (OUT / "fixed").mkdir(exist_ok=True)
    page = ["<meta charset='utf-8'><title>Fashion200K crop fix</title>",
            "<style>body{font:13px sans-serif;margin:16px}figure{display:inline-block;margin:6px;vertical-align:top;"
            "border:1px solid #ccc;padding:4px}img{height:230px}figcaption{font-size:11px}</style>",
            "<h1>Fashion200K val: 100 ảnh ngẫu nhiên trong số ảnh bị cắt nhầm</h1>",
            "<p>Trái: ảnh gốc. <b style='color:#06c'>Khung xanh</b> = hộp cũ (điểm cao nhất, sai món đồ); "
            "<b style='color:#d00'>khung đỏ</b> = hộp mới (đúng loại sản phẩm). Phải: ảnh sau khi cắt lại.</p>",
            f"<pre>{html.escape(json.dumps(stats, indent=1))}</pre>"]
    tiles = []
    for i, f in enumerate(sample):
        src = next((SRC / f.parent.name).glob(f.stem + ".*"))
        with Image.open(src) as im:
            im = im.convert("RGB")
            d = ImageDraw.Draw(im)
            w = max(3, im.width // 120)
            d.rectangle(c.pixel_box(old[f.stem], im.size), outline=(0, 102, 204), width=w)
            d.rectangle(c.pixel_box(new[f.stem], im.size), outline=(221, 0, 0), width=w)
            left = im.resize((max(1, round(im.width * H / im.height)), H))
        with Image.open(f) as cr:
            cr = cr.convert("RGB")
            right = cr.resize((max(1, round(cr.width * H / cr.height)), H))
        tile = Image.new("RGB", (left.width + right.width + 8, H), "white")
        tile.paste(left, (0, 0))
        tile.paste(right, (left.width + 8, 0))
        name = f"{i:03d}_{f.stem}.jpg"
        tile.save(OUT / "fixed" / name, quality=88)
        tiles.append(tile)
        cap = f"{f.stem} | sản phẩm '{cat[f.stem]}' | hộp cũ: {cls_old[f.stem]} → hộp mới: {cls_new[f.stem]}"
        page.append(f"<figure><img src='fixed/{name}'><figcaption>{html.escape(cap)}</figcaption></figure>")
    for s in range(0, len(tiles), 20):
        chunk = tiles[s:s + 20]
        cw = max(t.width for t in chunk) + 10
        sheet = Image.new("RGB", (cw * 5, (H + 10) * ((len(chunk) + 4) // 5)), "white")
        for j, t in enumerate(chunk):
            sheet.paste(t, ((j % 5) * cw, (j // 5) * (H + 10)))
        sheet.save(OUT / f"sheet_fixed_{s // 20 + 1}.jpg", quality=85)
    (OUT / "index.html").write_text("\n".join(page), encoding="utf-8")
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print("written", OUT / "index.html")


if __name__ == "__main__":
    main()
