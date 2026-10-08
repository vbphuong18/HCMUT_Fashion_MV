"""Download original Fashion200K images (fashion-200k/image_urls.txt) for the FashionMV products.

Output: <out>/<id>/<id>_<k>.jpeg, the first MAX_VIEWS images per product (as procir/datasets.py reads).
Resumable: views are matched by file stem (<id>_<k>), so a view already present under another
extension (the .jpg files of the earlier HF mirror) is not fetched again; only missing views are. Stops early when
the server keeps refusing; re-run later to continue. Failures are listed in <out>/../f200k_failed.json.

  python src/data/download_f200k_urls.py                       # train + val products
  python src/data/download_f200k_urls.py --triplets data/FashionMV_hf/data/data/train_triplets.jsonl
"""
import argparse
import io
import json
import os
import threading
import time
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]  # repository root
ZIP = ROOT / "data" / "raw" / "fashion-200k-20261003T154803Z-1-001.zip"
DATA = ROOT / "data" / "FashionMV_hf" / "data" / "data"
OUT = ROOT / "data" / "images_official_hr" / "f200k"
MAX_VIEWS = 5
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def wanted_ids(triplet_paths):
    ids = set()
    for path in triplet_paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                it = json.loads(line)
                if it["dataset"] == "f200k":
                    ids.update([str(it["source_id"]), str(it["target_id"])])
    return ids


def _url_lines(path):
    """Lines of image_urls.txt, read from the Fashion200K zip or from the extracted text file."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z, z.open("fashion-200k/image_urls.txt") as fh:
            yield from io.TextIOWrapper(fh, encoding="utf-8")
    else:
        with open(path, encoding="utf-8") as fh:
            yield from fh


def load_urls(path, ids):
    """{product_id: {file_name: url}} for the wanted products; `path` is the zip or image_urls.txt."""
    urls = defaultdict(dict)
    for raw in _url_lines(path):
        if not raw.strip():
            continue
        rel, url = raw.rstrip("\r\n").split("\t")
        parts = rel.split("/")
        if parts[-2] in ids:
            urls[parts[-2]][parts[-1]] = url
    return dict(urls)


def _stems_in(folder):
    if not folder.is_dir():
        return set()
    return {p.stem for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS}


def _wanted_names(urls_of_product, max_views=MAX_VIEWS):
    return sorted(urls_of_product)[:max_views]


def build_jobs(ids, urls, out, max_views=MAX_VIEWS):
    """(id, file_name, url) still to fetch: views whose stem is not on disk yet."""
    out = Path(out)
    jobs = []
    for i in sorted(ids):
        have = _stems_in(out / i)
        jobs.extend((i, n, urls[i][n]) for n in _wanted_names(urls.get(i, {}), max_views)
                    if Path(n).stem not in have)
    return jobs


def http_get(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def download(jobs, out, fetch=http_get, workers=4, delay=0.25, retries=3, max_consec_fail=50,
             progress_every=1000):
    out = Path(out)
    stats = {"ok": 0, "fail": 0, "consec_fail": 0, "stopped": False}
    failed, lock = [], threading.Lock()

    def one(job):
        i, name, url = job
        with lock:
            if stats["consec_fail"] >= max_consec_fail:
                stats["stopped"] = True
                return
        dest = out / i / name
        err = None
        for attempt in range(retries):
            try:
                data = fetch(url)
                Image.open(io.BytesIO(data)).verify()  # rejects HTML error pages served with 200
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name(dest.name + ".part")
                tmp.write_bytes(data)
                os.replace(tmp, dest)
                with lock:
                    stats["ok"] += 1
                    stats["consec_fail"] = 0
                time.sleep(delay)
                return
            except Exception as e:  # noqa: BLE001 - every failure is recorded and retried
                err = f"{type(e).__name__}: {str(e)[:80]}"
                if attempt + 1 < retries:
                    time.sleep(1 + attempt * 2)
        with lock:
            stats["fail"] += 1
            stats["consec_fail"] += 1
            failed.append((i, name, url, err))

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for n, _ in enumerate(ex.map(one, jobs), 1):
            if progress_every and n % progress_every == 0:
                print(f"{n}/{len(jobs)} ok={stats['ok']} fail={stats['fail']} {time.time() - t0:.0f}s", flush=True)
    return stats, failed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--triplets", action="append",
                    help="repeatable; default: train_triplets.jsonl and val_triplets.jsonl")
    ap.add_argument("--zip", "--urls", dest="zip", default=str(ZIP),
                    help="Fashion200K zip holding fashion-200k/image_urls.txt, or that text file itself")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--delay", type=float, default=0.25, help="seconds each worker waits after a request")
    ap.add_argument("--limit", type=int, default=0, help="only the first N products (testing)")
    args = ap.parse_args(argv)

    triplets = args.triplets or [DATA / "train_triplets.jsonl", DATA / "val_triplets.jsonl"]
    ids = sorted(wanted_ids(triplets))
    if args.limit:
        ids = ids[: args.limit]
    urls = load_urls(args.zip, set(ids))
    jobs = build_jobs(ids, urls, args.out)
    no_url = [i for i in ids if i not in urls]
    print(f"products: {len(ids)} (without url: {len(no_url)}); images to fetch: {len(jobs)}", flush=True)

    stats, failed = download(jobs, args.out, workers=args.workers, delay=args.delay)
    if stats["stopped"]:
        print("STOP: too many consecutive failures, the server may be refusing. Re-run later to resume.", flush=True)
    print(f"DONE ok={stats['ok']} fail={stats['fail']}", flush=True)
    report = Path(args.out).parent / "f200k_failed.json"
    report.write_text(json.dumps(failed, indent=1), encoding="utf-8")
    complete = sum(1 for i in ids if i in urls and not build_jobs([i], urls, args.out))
    print(f"complete products: {complete}/{len(ids)}; failures listed in {report}", flush=True)


if __name__ == "__main__":
    main()
