"""Build images/deepfashion/<source_id>/<files> from official DeepFashion In-shop images.

Official layout:  <src>/WOMEN/Tees_Tanks/id_00006581/12_1_front.jpg
evaluate.py wants: <out>/WOMEN/Tees_Tanks/id_00006581/12/12_1_front.jpg
Only products referenced by val_triplets.jsonl (dataset == deepfashion) are copied.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repository root
TRIPLETS = ROOT / "data" / "FashionMV_hf" / "data" / "data" / "val_triplets.jsonl"
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="folder holding WOMEN/ and MEN/ (official images)")
ap.add_argument("--out", required=True, help="output folder, e.g. data/images_official_hr/deepfashion")
args = ap.parse_args()

wanted = set()
with open(TRIPLETS, encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        if item["dataset"] == "deepfashion":
            wanted.add(str(item["source_id"]))
            wanted.add(str(item["target_id"]))
print(f"deepfashion ids needed: {len(wanted)}")

n_files = 0
missing = []
for sid in sorted(wanted):
    prod_dir, nn = sid.rsplit("/", 1)
    src_dir = os.path.join(args.src, *prod_dir.split("/"))
    files = []
    if os.path.isdir(src_dir):
        files = [x for x in os.listdir(src_dir)
                 if x.startswith(nn + "_") and Path(x).suffix.lower() in EXTS]
    if not files:
        missing.append(sid)
        continue
    dst_dir = os.path.join(args.out, *sid.split("/"))
    os.makedirs(dst_dir, exist_ok=True)
    for x in files:
        dst = os.path.join(dst_dir, x)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(src_dir, x), dst)
        n_files += 1

print(f"ids covered: {len(wanted) - len(missing)} / {len(wanted)}")
print(f"images in output: {n_files}")
print(f"missing ids: {len(missing)}")
if missing:
    print("sample missing:", missing[:10])
