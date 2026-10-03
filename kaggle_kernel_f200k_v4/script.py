import glob
import json
import os
import random
import subprocess
import sys
import traceback

SAMPLE_SIZE = 1500  # reduced further to bound both time and memory risk


def run(cmd, check=True):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check)


WORK = "/kaggle/working"
REPO = f"{WORK}/FashionMV"
DATA_DIR = f"{WORK}/hf_data/data"
SAMPLE_DIR = f"{WORK}/hf_data_sample"
MODEL_DIR = f"{WORK}/hf_model"
IMAGE_ROOT = f"{WORK}/images"
F200K_PARQUET_DIR = f"{WORK}/f200k_parquet"

run(f"{sys.executable} -m pip install -q -U 'huggingface_hub==1.33.0' 'transformers>=5.0' pyarrow")
run(f"{sys.executable} -m pip install -q causal-conv1d flash-linear-attention", check=False)
run(f"git clone -q https://github.com/yuandaxia2001/FashionMV.git {REPO}")

from huggingface_hub import snapshot_download

print("Downloading annotations...", flush=True)
snapshot_download("yuandaxia/FashionMV", repo_type="dataset", local_dir=f"{WORK}/hf_data")
print("Downloading checkpoint...", flush=True)
snapshot_download("yuandaxia/ProCIR", local_dir=MODEL_DIR)
print("Downloading Fashion200K image source (Marqo/fashion200k)...", flush=True)
snapshot_download("Marqo/fashion200k", repo_type="dataset", local_dir=F200K_PARQUET_DIR,
                   allow_patterns=["data/*.parquet"])

all_lines = []
with open(f"{DATA_DIR}/val_triplets.jsonl", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        if item["dataset"] == "f200k":
            all_lines.append(line)

wanted_pids = set()
for line in all_lines:
    item = json.loads(line)
    wanted_pids.add(str(item["source_id"]))
    wanted_pids.add(str(item["target_id"]))
print(f"val_triplets f200k product ids needed: {len(wanted_pids)}", flush=True)

import pyarrow.parquet as pq

os.makedirs(f"{IMAGE_ROOT}/f200k", exist_ok=True)

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

print(f"Written images: {n_written}, product ids covered: {len(pids_found)}/{len(wanted_pids)}", flush=True)

usable = [
    line for line in all_lines
    if json.loads(line)["source_id"] in pids_found and json.loads(line)["target_id"] in pids_found
]
print(f"Usable f200k triplets (both images present): {len(usable)} / {len(all_lines)}", flush=True)

random.seed(42)
sampled = random.sample(usable, min(SAMPLE_SIZE, len(usable)))
os.makedirs(SAMPLE_DIR, exist_ok=True)
with open(f"{SAMPLE_DIR}/val_triplets.jsonl", "w", encoding="utf-8") as f:
    f.writelines(sampled)
print(f"Subsampled f200k triplets for eval: {len(sampled)}", flush=True)

os.chdir(REPO)
try:
    run(
        f"{sys.executable} evaluate.py "
        f"--model_path {MODEL_DIR} "
        f"--image_root {IMAGE_ROOT} "
        f"--data_dir {SAMPLE_DIR} "
        f"--datasets f200k "
        f"--output_dir {WORK}/results "
        f"--batch_size 8"
    )
except subprocess.CalledProcessError:
    print("evaluate.py failed - see stderr above for the real traceback", flush=True)
    raise

print("DONE", flush=True)
