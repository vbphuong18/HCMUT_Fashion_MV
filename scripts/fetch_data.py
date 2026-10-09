"""Server side of scripts/pack_data.py: download the procir_*.zip files from a shared Google Drive
folder, extract them into the repository root and check that every triplet has its images.

  python scripts/fetch_data.py --gdrive-folder "https://drive.google.com/drive/folders/<id>"
  python scripts/fetch_data.py --zips-dir /path/with/zips      # zips already on the server

From Drive the zips are fetched one at a time: download, extract, delete, then the next one, so the
disk needs the extracted data plus a single zip (~4 GB), not every zip and its extracted copy.
Re-runs are cheap: every extracted zip leaves a marker in data/.fetched/ and is not downloaded again.
The Drive folder must be shared as "Anyone with the link".
"""
import argparse
import fnmatch
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE_MARGIN = 1 << 30  # keep 1 GiB free after extracting


def list_drive_zips(url):
    """[(file id, name)] of the files in a shared Drive folder, without downloading."""
    import gdown

    files = gdown.download_folder(url=url, skip_download=True, quiet=True)
    return [(f.id, Path(f.path).name) for f in files]


def _download(fid, out):
    import gdown

    gdown.download(id=fid, output=str(out), quiet=False)
    if not zipfile.is_zipfile(out):
        raise SystemExit(f"download of {out.name} failed or is incomplete")


def _extract(z, root):
    root = Path(root)
    with zipfile.ZipFile(z) as zf:
        need = sum(i.file_size for i in zf.infolist())
        free = shutil.disk_usage(root).free
        if free < need + SPACE_MARGIN:
            raise SystemExit(f"No space for {Path(z).name}: needs {need / 2**30:.1f} GiB + 1 GiB margin, "
                             f"{free / 2**30:.1f} GiB free on {root}")
        print(f"extracting {Path(z).name} ({need / 2**30:.1f} GiB)", flush=True)
        zf.extractall(root)
    (root / "data" / ".fetched" / (Path(z).name + ".done")).write_text("ok\n", encoding="utf-8")


def _markers(root):
    markers = Path(root) / "data" / ".fetched"
    markers.mkdir(parents=True, exist_ok=True)
    return markers


def extract_all(zips_dir, root):
    """Extract every procir_*.zip under zips_dir into root; returns the zips extracted now."""
    markers = _markers(root)
    done = []
    for z in sorted(Path(zips_dir).rglob("procir_*.zip")):
        if (markers / (z.name + ".done")).exists():
            continue
        _extract(z, root)
        done.append(z.name)
    return done


def fetch_drive(url, download_dir, root, list_files=list_drive_zips, download=_download):
    """Download, extract and delete the Drive zips one by one; returns the zips extracted now.

    A zip already in download_dir is reused when complete and fetched again when partial."""
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    markers = _markers(root)
    done = []
    for fid, name in list_files(url):
        if not fnmatch.fnmatch(name, "procir_*.zip") or (markers / (name + ".done")).exists():
            continue
        z = download_dir / name
        if not zipfile.is_zipfile(z):
            z.unlink(missing_ok=True)
            download(fid, z)
        _extract(z, root)
        z.unlink()
        done.append(name)
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gdrive-folder", help="shared Google Drive folder holding procir_*.zip")
    src.add_argument("--zips-dir", help="folder that already holds procir_*.zip")
    ap.add_argument("--download-dir", default=str(ROOT / "data" / "_downloads"))
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args(argv)

    if args.gdrive_folder:
        extracted = fetch_drive(args.gdrive_folder, args.download_dir, ROOT)
    else:
        extracted = extract_all(args.zips_dir, ROOT)
    print(f"extracted {len(extracted)} zip(s): {extracted}", flush=True)
    if args.no_verify:
        return 0
    sys.path.insert(0, str(ROOT / "scripts"))
    import setup_data

    return setup_data.main(["verify"])


if __name__ == "__main__":
    sys.exit(main())
