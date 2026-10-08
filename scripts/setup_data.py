"""Rebuild the training data on a fresh machine (after cloning the repository).

Result, as the configs expect it:
  data/FashionMV_hf/data/data/{train,val}_{triplets,captions}.jsonl
  data/images_official_hr/{deepfashion,f200k,fashiongen_train,fashiongen_val}/<product_id>/<views>

Stages (`all` runs them in this order; each one can be re-run and resumes where it stopped):
  annotations  FashionMV jsonl files from Hugging Face (yuandaxia/FashionMV).
  deepfashion  MMLab DeepFashion In-shop img_highres -> one folder per colorway (FashionMV product).
               --deepfashion: the In-shop img_highres zip (the part with MEN/ and WOMEN/) or an
               extracted folder. The zip is encrypted: put the MMLab password in DEEPFASHION_PASSWORD.
  f200k        Fashion200K: full images from image_urls.txt (train + val products) into f200k_source/,
               then cropped to the released garment boxes into f200k/ (src/data/crop_f200k.py).
               --f200k-urls: the Fashion200K zip from the xthan/fashion-200k Google Drive (it holds
               image_urls.txt plus the detection/ and labels/ files that give the crop boxes).
  fashiongen   fashiongen_256_256_{train,validation}.h5 from Kaggle (bothin/fashiongen-validation; needs
               Kaggle API credentials), then the upstream tools/prepare_fashiongen.py.
  verify       Triplets whose source and target both have images, per dataset and split; writes
               data/data_status.json and exits 1 if anything is missing.

  python scripts/setup_data.py all --deepfashion /path/img_highres.zip --f200k-urls /path/fashion-200k.zip
  python scripts/setup_data.py f200k --f200k-urls /path/fashion-200k.zip --workers 8
  python scripts/setup_data.py verify
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "data")]

STAGES = ("annotations", "deepfashion", "f200k", "fashiongen", "verify")
SOURCES = ("deepfashion", "f200k", "fashiongen")
SPLIT_NAME = {("train", "fashiongen"): "fashiongen_train", ("val", "fashiongen"): "fashiongen_val"}
FASHIONGEN_KAGGLE = "bothin/fashiongen-validation"
FASHIONGEN_FILES = {"fashiongen_train": "fashiongen_256_256_train.h5",
                    "fashiongen_val": "fashiongen_256_256_validation.h5"}


def _run(cmd):
    print("+", " ".join(str(c) for c in cmd if c), flush=True)
    env = dict(os.environ, PYTHONUTF8="1")
    subprocess.run([str(c) for c in cmd], check=True, env=env)


def image_root(data_root):
    return Path(data_root) / "images_official_hr"


def triplet_files(ann):
    return [Path(ann) / "train_triplets.jsonl", Path(ann) / "val_triplets.jsonl"]


# ---- annotations ----

def stage_annotations(data_root):
    from huggingface_hub import snapshot_download

    target = Path(data_root) / "FashionMV_hf" / "data"
    snapshot_download("yuandaxia/FashionMV", repo_type="dataset", local_dir=str(target), allow_patterns=["*.jsonl"])
    hits = sorted(target.rglob("train_triplets.jsonl"))
    if not hits:
        raise SystemExit(f"no train_triplets.jsonl under {target}")
    print(f"annotations in {hits[0].parent}", flush=True)
    return hits[0].parent


# ---- DeepFashion In-shop high-res ----

def find_inshop_root(path):
    """The folder that holds MEN/ and WOMEN/: `path` itself or a child such as img_highres/."""
    path = Path(path)
    for cand in [path, *sorted(p for p in path.glob("*") if p.is_dir()),
                 *sorted(p for p in path.glob("*/*") if p.is_dir())]:
        if (cand / "MEN").is_dir() and (cand / "WOMEN").is_dir():
            return cand
    raise FileNotFoundError(f"no folder with MEN/ and WOMEN/ under {path}")


def extract_inshop(zip_path, dest, password=None, use_unzip=None, run=_run):
    """Extract only img_highres/{MEN,WOMEN} (the In-shop images). `unzip` is far faster than Python's
    ZipCrypto for the encrypted MMLab zip; -n skips files already extracted, so re-runs resume."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if use_unzip is None:
        use_unzip = shutil.which("unzip") is not None
    if use_unzip:
        run(["unzip", "-q", "-n", *(["-P", password] if password else []), zip_path,
             "img_highres/MEN/*", "img_highres/WOMEN/*", "-d", dest])
    else:
        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.namelist() if m.startswith(("img_highres/MEN/", "img_highres/WOMEN/"))]
            for m in members:
                if not (dest / m).exists():
                    zf.extract(m, dest, pwd=password.encode() if password else None)
    return find_inshop_root(dest)


def stage_deepfashion(src, data_root, ann, password=None):
    from parquet_extract import wanted_ids
    from regroup_deepfashion_highres import regroup

    src = Path(src)
    if src.is_file():
        password = password or os.environ.get("DEEPFASHION_PASSWORD")
        root = extract_inshop(src, Path(data_root) / "extracted" / "inshop_highres", password)
    else:
        root = find_inshop_root(src)
    wanted = set()
    for t in triplet_files(ann):
        wanted |= wanted_ids(t, "deepfashion")
    n, found = regroup(str(root), str(image_root(data_root) / "deepfashion"), wanted)
    print(f"deepfashion: {n} images written, {len(found)}/{len(wanted)} products", flush=True)


# ---- Fashion200K ----

def stage_f200k(urls, data_root, ann, workers=8, delay=0.1):
    """Full images from the URLs into f200k_source/, then the release's garment crops into f200k/."""
    import crop_f200k
    import download_f200k_urls

    if not zipfile.is_zipfile(urls):
        raise SystemExit("--f200k-urls must be the Fashion200K zip: its detection/ and labels/ files give the "
                         "crop boxes of the released images")
    source, out = image_root(data_root) / "f200k_source", image_root(data_root) / "f200k"
    argv = ["--urls", str(urls), "--out", str(source), "--workers", str(workers), "--delay", str(delay)]
    for t in triplet_files(ann):
        argv += ["--triplets", str(t)]
    download_f200k_urls.main(argv)
    crop_f200k.main(["--src", str(source), "--dst", str(out), "--zip", str(urls)])


# ---- FashionGen ----

def _unzip_file(zip_path, dest):
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)


def stage_fashiongen(data_root, run=_run, unzip=_unzip_file):
    raw = Path(data_root) / "raw" / "fashiongen"
    raw.mkdir(parents=True, exist_ok=True)
    for out_name, fname in FASHIONGEN_FILES.items():
        h5 = raw / fname
        if not h5.exists():
            run([sys.executable, "-m", "kaggle", "datasets", "download", FASHIONGEN_KAGGLE, "-f", fname, "-p", raw])
            archive = raw / f"{fname}.zip"
            if archive.exists():  # Kaggle serves single large files zipped
                unzip(archive, raw)
                archive.unlink()
        run([sys.executable, ROOT / "external" / "FashionMV" / "tools" / "prepare_fashiongen.py",
             "--h5_path", h5, "--output_dir", image_root(data_root) / out_name])


# ---- verify ----

def verify(data_root, ann, datasets=SOURCES):
    from procir_train.data import attach_images, load_triplets

    status, ok = {}, True
    for split in ("train", "val"):
        status[split] = {}
        for src in datasets:
            name = SPLIT_NAME.get((split, src), src)
            t = load_triplets(Path(ann) / f"{split}_triplets.jsonl", [name])
            usable = len(attach_images(t, str(image_root(data_root)), 5))
            status[split][name] = {"usable": usable, "total": len(t)}
            ok &= usable == len(t)
            print(f"{split:5s} {name:17s} {usable:7d} / {len(t):7d} triplets", flush=True)
    Path(data_root, "data_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("all triplets have images" if ok else "INCOMPLETE: re-run the stage of the missing dataset", flush=True)
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+", choices=("all", *STAGES))
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--deepfashion", help="In-shop img_highres zip or extracted folder")
    ap.add_argument("--f200k-urls", help="Fashion200K zip (with fashion-200k/image_urls.txt) or image_urls.txt")
    ap.add_argument("--workers", type=int, default=8, help="Fashion200K download threads")
    ap.add_argument("--delay", type=float, default=0.1, help="seconds each download thread waits per request")
    ap.add_argument("--datasets", nargs="+", choices=SOURCES, default=list(SOURCES), help="for verify")
    args = ap.parse_args(argv)

    stages = STAGES if "all" in args.stages else [s for s in STAGES if s in args.stages]
    data_root = Path(args.data_root)
    ann = data_root / "FashionMV_hf" / "data" / "data"
    missing = [f"--{k}" for s, k in (("deepfashion", "deepfashion"), ("f200k", "f200k-urls"))
               if s in stages and not getattr(args, k.replace("-", "_"))]
    if missing:
        ap.error(f"{', '.join(missing)} required for the requested stages")

    if "annotations" in stages:
        ann = stage_annotations(data_root)
    if "deepfashion" in stages:
        stage_deepfashion(args.deepfashion, data_root, ann)
    if "f200k" in stages:
        stage_f200k(args.f200k_urls, data_root, ann, args.workers, args.delay)
    if "fashiongen" in stages:
        stage_fashiongen(data_root)
    if "verify" in stages:
        return 0 if verify(data_root, ann, args.datasets) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
