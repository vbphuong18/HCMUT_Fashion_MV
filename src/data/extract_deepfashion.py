"""Extract DeepFashion In-shop images (Marqo/deepfashion-inshop parquet) for the products of a FashionMV triplet file."""
import argparse
from pathlib import Path

from parquet_extract import extract, wanted_ids

ROOT = Path(__file__).resolve().parents[2]  # repository root
DEFAULT_SRC = ROOT / "data" / "raw" / "deepfashion_parquet" / "data"
DEFAULT_OUT = ROOT / "data" / "images" / "deepfashion"
DEFAULT_TRIPLETS = ROOT / "data" / "FashionMV_hf" / "data" / "data" / "val_triplets.jsonl"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--triplets", default=str(DEFAULT_TRIPLETS))
    args = ap.parse_args(argv)

    wanted = wanted_ids(args.triplets, "deepfashion")
    print(f"deepfashion ids needed: {len(wanted)}")
    # item_ID = product id with "/" -> "_" plus two suffixes (image index, view name)
    prefix_map = {sid.replace("/", "_"): sid for sid in wanted}

    def id_of(item_id):
        return prefix_map.get(item_id.rsplit("_", 2)[0])

    n_written, found = extract(args.src, args.out, id_of)
    missing = wanted - found
    print(f"\nWritten images: {n_written}")
    print(f"Ids covered: {len(found)} / {len(wanted)}")
    print(f"Missing ids: {len(missing)}")
    if missing:
        print("Sample missing:", sorted(missing)[:10])


if __name__ == "__main__":
    main()
