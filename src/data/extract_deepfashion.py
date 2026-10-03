import glob
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]  # repository root
# parquet shards downloaded from HuggingFace (Marqo/deepfashion-inshop)
SRC_GLOB = str(ROOT / "data" / "raw" / "deepfashion_parquet" / "data" / "*.parquet")
OUT_ROOT = str(ROOT / "data" / "images" / "deepfashion")
TRIPLETS = str(ROOT / "data" / "FashionMV_hf" / "data" / "data" / "val_triplets.jsonl")

os.makedirs(OUT_ROOT, exist_ok=True)

wanted_sids = set()
with open(TRIPLETS, encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        if item["dataset"] == "deepfashion":
            wanted_sids.add(str(item["source_id"]))
            wanted_sids.add(str(item["target_id"]))
print(f"val_triplets deepfashion ids needed: {len(wanted_sids)}")

prefix_map = {sid.replace("/", "_"): sid for sid in wanted_sids}

n_written = 0
sids_found = set()
for shard in sorted(glob.glob(SRC_GLOB)):
    table = pq.read_table(shard, columns=["image", "item_ID"])
    item_ids = table.column("item_ID").to_pylist()
    images = table.column("image").to_pylist()
    for item_id, img in zip(item_ids, images):
        prefix = item_id.rsplit("_", 2)[0]
        sid = prefix_map.get(prefix)
        if sid is None:
            continue
        pdir = os.path.join(OUT_ROOT, sid)
        os.makedirs(pdir, exist_ok=True)
        out_path = os.path.join(pdir, f"{item_id}.jpg")
        if not os.path.exists(out_path):
            with open(out_path, "wb") as fo:
                fo.write(img["bytes"])
            n_written += 1
        sids_found.add(sid)
    print(f"  processed {os.path.basename(shard)}, written so far: {n_written}")

missing = wanted_sids - sids_found
print(f"\nWritten images: {n_written}")
print(f"Ids covered: {len(sids_found)} / {len(wanted_sids)}")
print(f"Missing ids: {len(missing)}")
if missing:
    print("Sample missing:", list(missing)[:10])
