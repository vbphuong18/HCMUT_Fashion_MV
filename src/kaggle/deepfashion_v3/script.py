import glob
import json
import os
import random
import subprocess
import sys

SAMPLE_SIZE = 1500  # bound runtime; full val set is 5188 triplets


def run(cmd, check=True):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check)


WORK = "/kaggle/working"
REPO = f"{WORK}/FashionMV"
DATA_DIR = f"{WORK}/hf_data/data"
SAMPLE_DIR = f"{WORK}/hf_data_sample"
MODEL_DIR = f"{WORK}/hf_model"
IMAGE_ROOT = f"{WORK}/images"
DF_PARQUET_DIR = f"{WORK}/deepfashion_parquet"

# ---- fix dependency versions (huggingface_hub<2.0 required by transformers) ----
run(f"{sys.executable} -m pip install -q -U 'huggingface_hub==1.33.0' 'transformers>=5.0' pyarrow")

# ---- best-effort: install optimized CUDA kernels (much faster than reference fallback) ----
run(f"{sys.executable} -m pip install -q causal-conv1d flash-linear-attention", check=False)

# ---- get the eval code ----
run(f"git clone -q https://github.com/yuandaxia2001/FashionMV.git {REPO}")

from huggingface_hub import snapshot_download

print("Downloading annotations...", flush=True)
snapshot_download("yuandaxia/FashionMV", repo_type="dataset", local_dir=f"{WORK}/hf_data")
print("Downloading checkpoint...", flush=True)
snapshot_download("yuandaxia/ProCIR", local_dir=MODEL_DIR)
print("Downloading DeepFashion image source (Marqo/deepfashion-inshop)...", flush=True)
snapshot_download("Marqo/deepfashion-inshop", repo_type="dataset", local_dir=DF_PARQUET_DIR,
                   allow_patterns=["data/*.parquet"])

# ---- subsample val_triplets.jsonl to bound runtime ----
os.makedirs(SAMPLE_DIR, exist_ok=True)
all_lines = []
with open(f"{DATA_DIR}/val_triplets.jsonl", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        if item["dataset"] == "deepfashion":
            all_lines.append(line)

random.seed(42)
sampled = random.sample(all_lines, min(SAMPLE_SIZE, len(all_lines)))
with open(f"{SAMPLE_DIR}/val_triplets.jsonl", "w", encoding="utf-8") as f:
    f.writelines(sampled)
print(f"Subsampled deepfashion triplets: {len(sampled)} / {len(all_lines)}", flush=True)

# ---- build images/deepfashion/<source_id>/<item_id>.jpg from parquet (only for sampled ids) ----
import pyarrow.parquet as pq

os.makedirs(f"{IMAGE_ROOT}/deepfashion", exist_ok=True)

wanted_sids = set()
for line in sampled:
    item = json.loads(line)
    wanted_sids.add(str(item["source_id"]))
    wanted_sids.add(str(item["target_id"]))
print(f"Sampled deepfashion ids needed: {len(wanted_sids)}", flush=True)

prefix_map = {sid.replace("/", "_"): sid for sid in wanted_sids}

n_written = 0
sids_found = set()
for shard in sorted(glob.glob(f"{DF_PARQUET_DIR}/data/*.parquet")):
    table = pq.read_table(shard, columns=["image", "item_ID"])
    item_ids = table.column("item_ID").to_pylist()
    images = table.column("image").to_pylist()
    for item_id, img in zip(item_ids, images):
        prefix = item_id.rsplit("_", 2)[0]
        sid = prefix_map.get(prefix)
        if sid is None:
            continue
        pdir = os.path.join(IMAGE_ROOT, "deepfashion", sid)
        os.makedirs(pdir, exist_ok=True)
        out_path = os.path.join(pdir, f"{item_id}.jpg")
        if not os.path.exists(out_path):
            with open(out_path, "wb") as fo:
                fo.write(img["bytes"])
            n_written += 1
        sids_found.add(sid)

print(f"Written images: {n_written}, ids covered: {len(sids_found)}/{len(wanted_sids)}", flush=True)

# ---- run evaluation on GPU ----
os.chdir(REPO)
run(
    f"{sys.executable} evaluate.py "
    f"--model_path {MODEL_DIR} "
    f"--image_root {IMAGE_ROOT} "
    f"--data_dir {SAMPLE_DIR} "
    f"--datasets deepfashion "
    f"--output_dir {WORK}/results "
    f"--batch_size 48"
)

print("DONE", flush=True)
