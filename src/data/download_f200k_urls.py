"""Download original Fashion200K images listed in image_urls.txt for the val IDs that
the HF mirror (data/images/f200k) does not cover.

Output: <out>/<id>/<id>_<k>.jpeg  (first MAX_VIEWS images per id, as procir/datasets.py reads)
Resumable: existing valid files are skipped. Stops early if the server starts refusing.
"""
import argparse
import io
import json
import os
import time
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]  # repository root
ZIP = ROOT / "data" / "raw" / "fashion-200k-20261003T154803Z-1-001.zip"
TRIPLETS = ROOT / "data" / "FashionMV_hf" / "data" / "data" / "val_triplets.jsonl"
HF_DIR = ROOT / "data" / "images" / "f200k"
MAX_VIEWS = 5

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "data" / "images_f200k_url" / "f200k"))
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--delay", type=float, default=0.25, help="seconds each worker waits after a request")
ap.add_argument("--all-ids", action="store_true", help="also fetch ids already covered by the HF mirror")
ap.add_argument("--limit", type=int, default=0, help="only the first N ids (testing)")
args = ap.parse_args()

# val ids that need images
ids = set()
with open(TRIPLETS, encoding="utf-8") as f:
    for line in f:
        it = json.loads(line)
        if it["dataset"] == "f200k":
            ids.update([str(it["source_id"]), str(it["target_id"])])
if not args.all_ids:
    ids = {i for i in ids if not (HF_DIR / i).is_dir() or not any((HF_DIR / i).iterdir())}
ids = sorted(ids)
if args.limit:
    ids = ids[: args.limit]
print(f"ids to fetch: {len(ids)}", flush=True)

# id -> urls (file name -> url)
urls = defaultdict(dict)
with zipfile.ZipFile(ZIP) as z, z.open("fashion-200k/image_urls.txt") as fh:
    wanted = set(ids)
    for raw in io.TextIOWrapper(fh, encoding="utf-8"):
        path, url = raw.rstrip("\n").split("\t")
        parts = path.split("/")
        if parts[-2] in wanted:
            urls[parts[-2]][parts[-1]] = url

jobs = []
for i in ids:
    for name in sorted(urls.get(i, {}))[:MAX_VIEWS]:
        jobs.append((i, name, urls[i][name]))
print(f"images to fetch: {len(jobs)} (ids without url: {sum(1 for i in ids if i not in urls)})", flush=True)

os.makedirs(args.out, exist_ok=True)
stats = {"ok": 0, "skip": 0, "fail": 0, "consec_fail": 0}
failed = []


def valid(p):
    try:
        with Image.open(p) as im:
            im.verify()
        return True
    except Exception:
        return False


def fetch(job):
    i, name, url = job
    if stats["consec_fail"] >= 50:
        return
    dest = os.path.join(args.out, i, name)
    if os.path.exists(dest) and valid(dest):
        stats["skip"] += 1
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    err = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            data = urllib.request.urlopen(req, timeout=20).read()
            Image.open(io.BytesIO(data)).verify()
            with open(dest + ".part", "wb") as fo:
                fo.write(data)
            os.replace(dest + ".part", dest)
            stats["ok"] += 1
            stats["consec_fail"] = 0
            time.sleep(args.delay)
            return
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:80]}"
            time.sleep(1 + attempt * 2)
    stats["fail"] += 1
    stats["consec_fail"] += 1
    failed.append((i, name, url, err))


t0 = time.time()
with ThreadPoolExecutor(args.workers) as ex:
    for n, _ in enumerate(ex.map(fetch, jobs), 1):
        if n % 1000 == 0:
            print(f"{n}/{len(jobs)} ok={stats['ok']} skip={stats['skip']} fail={stats['fail']} {time.time()-t0:.0f}s", flush=True)
        if stats["consec_fail"] >= 50:
            print("STOP: 50 consecutive failures, server may be refusing. Re-run later to resume.", flush=True)
            break

print(f"DONE ok={stats['ok']} skip={stats['skip']} fail={stats['fail']} time={time.time()-t0:.0f}s", flush=True)
with open(os.path.join(os.path.dirname(args.out), "f200k_failed.json"), "w", encoding="utf-8") as f:
    json.dump(failed, f, indent=1)
for x in failed[:10]:
    print("FAIL", x, flush=True)
