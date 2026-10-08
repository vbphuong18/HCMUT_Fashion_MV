"""Server side of scripts/pack_data.py: download the procir_*.zip files from a shared Google Drive
folder, extract them into the repository root and check that every triplet has its images.

  python scripts/fetch_data.py --gdrive-folder "https://drive.google.com/drive/folders/<id>"
  python scripts/fetch_data.py --zips-dir /path/with/zips      # zips already on the server

Re-runs are cheap: gdown skips files already downloaded and every extracted zip leaves a marker in
data/.fetched/. The Drive folder must be shared as "Anyone with the link".
"""
import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def download_folder(url, dest, run=_run):
    Path(dest).mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "gdown", "--folder", url, "-O", dest, "--continue"])


def extract_all(zips_dir, root):
    """Extract every procir_*.zip under zips_dir into root; returns the zips extracted now."""
    root = Path(root)
    markers = root / "data" / ".fetched"
    markers.mkdir(parents=True, exist_ok=True)
    done = []
    for z in sorted(Path(zips_dir).rglob("procir_*.zip")):
        marker = markers / (z.name + ".done")
        if marker.exists():
            continue
        print(f"extracting {z.name}", flush=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(root)
        marker.write_text("ok\n", encoding="utf-8")
        done.append(z.name)
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gdrive-folder", help="shared Google Drive folder holding procir_*.zip")
    src.add_argument("--zips-dir", help="folder that already holds procir_*.zip")
    ap.add_argument("--download-dir", default=str(ROOT / "data" / "_downloads"))
    ap.add_argument("--delete-zips", action="store_true", help="remove each zip folder after extraction")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args(argv)

    zips_dir = Path(args.zips_dir) if args.zips_dir else Path(args.download_dir)
    if args.gdrive_folder:
        download_folder(args.gdrive_folder, zips_dir)
    extracted = extract_all(zips_dir, ROOT)
    print(f"extracted {len(extracted)} zip(s): {extracted}", flush=True)
    if args.delete_zips and args.gdrive_folder:
        for z in zips_dir.rglob("procir_*.zip"):
            z.unlink()
    if args.no_verify:
        return 0
    sys.path.insert(0, str(ROOT / "scripts"))
    import setup_data

    return setup_data.main(["verify"])


if __name__ == "__main__":
    sys.exit(main())
