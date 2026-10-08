"""Extract Fashion200K images (Marqo/fashion200k parquet) for the products of a FashionMV triplet file."""
import argparse
from pathlib import Path

from parquet_extract import extract, wanted_ids

ROOT = Path(__file__).resolve().parents[2]  # repository root
DEFAULT_SRC = ROOT / "data" / "raw" / "fashion200k_parquet" / "data"
DEFAULT_OUT = ROOT / "data" / "images" / "f200k"
DEFAULT_TRIPLETS = ROOT / "data" / "FashionMV_hf" / "data" / "data" / "val_triplets.jsonl"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--triplets", default=str(DEFAULT_TRIPLETS))
    args = ap.parse_args(argv)

    wanted = wanted_ids(args.triplets, "f200k")
    print(f"f200k product ids needed: {len(wanted)}")

    def id_of(item_id):
        pid = item_id.rsplit("_", 1)[0]
        return pid if pid in wanted else None

    n_written, found = extract(args.src, args.out, id_of)
    missing = wanted - found
    print(f"\nWritten images: {n_written}")
    print(f"Product ids covered: {len(found)} / {len(wanted)}")
    print(f"Missing product ids: {len(missing)}")
    if missing:
        print("Sample missing:", sorted(missing)[:10])


if __name__ == "__main__":
    main()
