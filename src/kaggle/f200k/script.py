import glob
import json
import os
import subprocess
import sys


def run(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


WORK = "/kaggle/working"
REPO = f"{WORK}/FashionMV"
DATA_DIR = f"{WORK}/hf_data/data"
MODEL_DIR = f"{WORK}/hf_model"
IMAGE_ROOT = f"{WORK}/images"
F200K_PARQUET_DIR = f"{WORK}/f200k_parquet"

# ---- fix dependency versions (huggingface_hub<2.0 required by transformers) ----
run(f"{sys.executable} -m pip install -q -U 'huggingface_hub==1.33.0' 'transformers>=5.0' pyarrow")

# ---- get the eval code ----
run(f"git clone -q https://github.com/yuandaxia2001/FashionMV.git {REPO}")

from huggingface_hub import snapshot_download

print("Downloading annotations...", flush=True)
snapshot_download("yuandaxia/FashionMV", repo_type="dataset", local_dir=f"{WORK}/hf_data")
print("Downloading checkpoint...", flush=True)
snapshot_download("yuandaxia/ProCIR", local_dir=MODEL_DIR)
print("Downloading Fashion200K image source (Marqo/fashion200k)...", flush=True)
snapshot_download("Marqo/fashion200k", repo_type="dataset", local_dir=F200K_PARQUET_DIR,
                   allow_patterns=["data/*.parquet"])

# ---- build images/f200k/<product_id>/<item_id>.jpg from parquet ----
import pyarrow.parquet as pq

os.makedirs(f"{IMAGE_ROOT}/f200k", exist_ok=True)

wanted_pids = set()
with open(f"{DATA_DIR}/val_triplets.jsonl", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        if item["dataset"] == "f200k":
            wanted_pids.add(str(item["source_id"]))
            wanted_pids.add(str(item["target_id"]))
print(f"val_triplets f200k product ids needed: {len(wanted_pids)}", flush=True)

n_written = 0
pids_found = set()
for shard in sorted(glob.glob(f"{F200K_PARQUET_DIR}/data/*.parquet")):
    table = pq.read_table(shard, columns=["image", "item_ID"])
    item_ids = table.column("item_ID").to_pylist()
    images = table.column("image").to_pylist()
    for item_id, img in zip(item_ids, images):
        pid = item_id.rsplit("_", 1)[0]
        if pid not in wanted_pids:
            continue
        pdir = os.path.join(IMAGE_ROOT, "f200k", pid)
        os.makedirs(pdir, exist_ok=True)
        out_path = os.path.join(pdir, f"{item_id}.jpg")
        if not os.path.exists(out_path):
            with open(out_path, "wb") as fo:
                fo.write(img["bytes"])
            n_written += 1
        pids_found.add(pid)
    print(f"  processed {os.path.basename(shard)}, written so far: {n_written}", flush=True)

missing = wanted_pids - pids_found
print(f"Written images: {n_written}, product ids covered: {len(pids_found)}/{len(wanted_pids)} "
      f"(missing {len(missing)} - source Marqo/fashion200k is not a 100% mirror; "
      f"evaluate.py will skip triplets with missing source/target images)", flush=True)

# ---- run evaluation on GPU ----
os.chdir(REPO)
run(
    f"{sys.executable} evaluate.py "
    f"--model_path {MODEL_DIR} "
    f"--image_root {IMAGE_ROOT} "
    f"--data_dir {DATA_DIR} "
    f"--datasets f200k "
    f"--output_dir {WORK}/results "
    f"--batch_size 32"
)

print("DONE", flush=True)
