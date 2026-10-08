import io
import json
import sys
import zipfile

import pytest
from PIL import Image

from conftest import ROOT

sys.path.insert(0, str(ROOT / "src" / "data"))


def _jpeg():
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (200, 10, 10)).save(buf, "JPEG")
    return buf.getvalue()


def _triplets(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _url_zip(path, entries):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("fashion-200k/image_urls.txt",
                   "".join(f"women/dresses/x/{pid}/{name}\thttp://cdn/{name}\n" for pid, name in entries))
    return path


def test_wanted_ids_reads_f200k_from_several_triplet_files(tmp_path):
    import download_f200k_urls as d

    a = _triplets(tmp_path / "train.jsonl", [{"dataset": "f200k", "source_id": 1, "target_id": 2},
                                             {"dataset": "deepfashion", "source_id": "W/x", "target_id": "W/y"}])
    b = _triplets(tmp_path / "val.jsonl", [{"dataset": "f200k", "source_id": "3", "target_id": "1"}])
    assert d.wanted_ids([a, b]) == {"1", "2", "3"}


def test_load_urls_keeps_only_wanted_products(tmp_path):
    import download_f200k_urls as d

    z = _url_zip(tmp_path / "u.zip", [("1", "1_0.jpeg"), ("1", "1_1.jpeg"), ("9", "9_0.jpeg")])
    assert d.load_urls(z, {"1"}) == {"1": {"1_0.jpeg": "http://cdn/1_0.jpeg", "1_1.jpeg": "http://cdn/1_1.jpeg"}}


def test_jobs_skip_complete_products_and_finish_partial_ones(tmp_path):
    import download_f200k_urls as d

    urls = {"A": {"A_0.jpeg": "u", "A_1.jpeg": "u"},
            "B": {"B_0.jpeg": "u", "B_1.jpeg": "u", "B_2.jpeg": "u"},
            "C": {f"C_{k}.jpeg": "u" for k in range(7)}}
    out = tmp_path / "f200k"
    (out / "A").mkdir(parents=True)
    for n in ("A_0.jpg", "A_1.jpg"):  # complete, under another extension (earlier mirror)
        (out / "A" / n).write_bytes(_jpeg())
    (out / "B").mkdir()
    (out / "B" / "B_0.jpeg").write_bytes(_jpeg())  # interrupted download
    jobs = d.build_jobs(["A", "B", "C"], urls, out, max_views=5)
    assert [(i, n) for i, n, _ in jobs] == [("B", "B_1.jpeg"), ("B", "B_2.jpeg")] + \
        [("C", f"C_{k}.jpeg") for k in range(5)]


def test_download_writes_valid_images_and_records_failures(tmp_path):
    import download_f200k_urls as d

    def fetch(url):
        if "bad" in url:
            raise OSError("403 Forbidden")
        return _jpeg()

    jobs = [("A", "A_0.jpeg", "http://ok"), ("A", "A_1.jpeg", "http://bad")]
    stats, failed = d.download(jobs, tmp_path, fetch=fetch, workers=2, delay=0, retries=1)
    assert stats["ok"] == 1 and stats["fail"] == 1
    assert (tmp_path / "A" / "A_0.jpeg").exists() and not (tmp_path / "A" / "A_1.jpeg").exists()
    assert failed[0][:2] == ("A", "A_1.jpeg") and "403" in failed[0][3]
    assert not list(tmp_path.rglob("*.part"))


def test_download_rejects_non_image_payloads(tmp_path):
    import download_f200k_urls as d

    stats, _ = d.download([("A", "A_0.jpeg", "http://html")], tmp_path, fetch=lambda u: b"<html>",
                          workers=1, delay=0, retries=1)
    assert stats["fail"] == 1 and not (tmp_path / "A" / "A_0.jpeg").exists()


def test_download_stops_after_consecutive_failures(tmp_path):
    import download_f200k_urls as d

    calls = []

    def fetch(url):
        calls.append(url)
        raise OSError("refused")

    jobs = [("A", f"A_{k}.jpeg", f"http://x/{k}") for k in range(50)]
    stats, _ = d.download(jobs, tmp_path, fetch=fetch, workers=1, delay=0, retries=1, max_consec_fail=5)
    assert stats["stopped"] and len(calls) == 5


def test_jobs_match_existing_views_by_stem_not_extension(tmp_path):
    """Mirror files are <id>_<k>.jpg, url files <id>_<k>.jpeg: the same view must not be fetched twice."""
    import download_f200k_urls as d

    urls = {"P": {f"P_{k}.jpeg": "u" for k in range(5)}}
    (tmp_path / "P").mkdir()
    for k in (0, 1, 3):
        (tmp_path / "P" / f"P_{k}.jpg").write_bytes(_jpeg())
    jobs = d.build_jobs(["P"], urls, tmp_path, max_views=5)
    assert [n for _, n, _ in jobs] == ["P_2.jpeg", "P_4.jpeg"]


def test_load_urls_also_reads_a_plain_image_urls_txt(tmp_path):
    import download_f200k_urls as d

    txt = tmp_path / "image_urls.txt"
    txt.write_text("women/dresses/x/1/1_0.jpeg\thttp://cdn/1_0.jpeg\nwomen/dresses/x/9/9_0.jpeg\thttp://cdn/9_0.jpeg\n",
                   encoding="utf-8")
    assert d.load_urls(txt, {"1"}) == {"1": {"1_0.jpeg": "http://cdn/1_0.jpeg"}}
