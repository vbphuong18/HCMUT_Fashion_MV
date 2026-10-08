"""Shared logic: copy images from HuggingFace parquet shards into images/<dataset>/<product_id>/."""
import glob
import json
import os

import pyarrow.parquet as pq


def wanted_ids(triplets_path, dataset):
    ids = set()
    with open(triplets_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["dataset"] == dataset:
                ids.add(str(item["source_id"]))
                ids.add(str(item["target_id"]))
    return ids


def extract(src_dir, out_root, id_of):
    """id_of(item_ID) returns the FashionMV product id to file the image under, or None to skip."""
    shards = sorted(glob.glob(os.path.join(src_dir, "*.parquet")))
    if not shards:
        raise FileNotFoundError(f"no .parquet shards in {src_dir}")
    os.makedirs(out_root, exist_ok=True)
    n_written, found = 0, set()
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=256, columns=["image", "item_ID"]):
            for item_id, img in zip(batch.column("item_ID").to_pylist(), batch.column("image").to_pylist()):
                pid = id_of(item_id)
                if pid is None:
                    continue
                pdir = os.path.join(out_root, pid)
                os.makedirs(pdir, exist_ok=True)
                out_path = os.path.join(pdir, f"{item_id}.jpg")
                if not os.path.exists(out_path):
                    tmp_path = out_path + ".tmp"
                    with open(tmp_path, "wb") as fo:
                        fo.write(img["bytes"])
                    os.replace(tmp_path, out_path)
                    n_written += 1
                found.add(pid)
        print(f"  processed {os.path.basename(shard)}, written so far: {n_written}", flush=True)
    return n_written, found
