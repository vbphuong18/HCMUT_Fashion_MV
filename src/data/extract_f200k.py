import glob
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]  # repository root
# parquet shards downloaded from HuggingFace (Marqo/fashion200k)
SRC_GLOB = str(ROOT / "data" / "raw" / "fashion200k_parquet" / "data" / "*.parquet")
OUT_ROOT = str(ROOT / "data" / "images" / "f200k")
TRIPLETS = str(ROOT / "data" / "FashionMV_hf" / "data" / "data" / "val_triplets.jsonl")

os.makedirs(OUT_ROOT, exist_ok=True)

wanted_pids = set()
with open(TRIPLETS, encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        if item["dataset"] == "f200k":
            wanted_pids.add(str(item["source_id"]))
            wanted_pids.add(str(item["target_id"]))
print(f"val_triplets f200k product ids needed: {len(wanted_pids)}")

n_written = 0
pids_found = set()
for shard in sorted(glob.glob(SRC_GLOB)):
    table = pq.read_table(shard, columns=["image", "item_ID"])
    item_ids = table.column("item_ID").to_pylist()
    images = table.column("image").to_pylist()
    for item_id, img in zip(item_ids, images):
        pid = item_id.rsplit("_", 1)[0]
        if pid not in wanted_pids:
            continue
        pdir = os.path.join(OUT_ROOT, pid)
        os.makedirs(pdir, exist_ok=True)
        out_path = os.path.join(pdir, f"{item_id}.jpg")
        if not os.path.exists(out_path):
            with open(out_path, "wb") as fo:
                fo.write(img["bytes"])
            n_written += 1
        pids_found.add(pid)
    print(f"  processed {os.path.basename(shard)}, written so far: {n_written}")

missing = wanted_pids - pids_found
print(f"\nWritten images: {n_written}")
print(f"Product ids covered: {len(pids_found)} / {len(wanted_pids)}")
print(f"Missing product ids: {len(missing)}")
if missing:
    print("Sample missing:", list(missing)[:10])
