import sys
import zipfile

import pytest
from PIL import Image

from conftest import ROOT

sys.path.insert(0, str(ROOT / "src" / "data"))


def _zip(path, lines, labels=()):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("fashion-200k/detection/women_dress_detection.txt", "".join(l + "\n" for l in lines))
        z.writestr("fashion-200k/labels/dress_train_detect_all.txt", "".join(l + "\n" for l in labels))
    return path


def test_load_boxes_uses_the_detection_whose_score_the_label_file_records(tmp_path):
    """The release crops to the box scored in labels/*_detect_all.txt, not always the top-scored one."""
    import crop_f200k as c

    z = _zip(tmp_path / "f.zip", [
        "women/dresses/x/1/1_0.jpeg\tdress_-1.78_0.23_0.74_0.20_0.86\tskirt_-4.86_0.35_0.74_0.54_0.86",
        "women/dresses/x/2/2_0.jpeg\tskirt_-1.00_0.10_0.20_0.30_0.40\tdress_-4.57_0.02_0.98_0.00_1.00",
        "women/dresses/x/1/1_1.jpeg\tpants_-12.26_-0.00_0.41_0.00_1.02"],
        labels=["women/dresses/x/1/1_0.jpeg\t-1.780000\tgreen dress",
                "women/dresses/x/2/2_0.jpeg\t-4.570000\tblue dress"])
    boxes = c.load_boxes(z)
    assert boxes["1_0"] == (0.23, 0.74, 0.20, 0.86)
    assert boxes["2_0"] == (0.02, 0.98, 0.0, 1.0)      # second detection: its score is the labelled one
    assert boxes["1_1"] == (0.0, 0.41, 0.0, 1.0)       # no label line: top box, clamped to [0, 1]


def test_pixel_box_matches_the_released_crop():
    import crop_f200k as c

    # 91352269_0: the released (mirror) crop sits at (72, 84) with size 380x456 inside 520x650
    assert c.pixel_box((0.14, 0.87, 0.13, 0.83), (520, 650)) == (73, 84, 452, 540)


def test_crop_product_crops_full_images_and_keeps_released_crops(tmp_path):
    import crop_f200k as c

    src, dst, mirror = tmp_path / "src" / "7", tmp_path / "dst" / "7", tmp_path / "mirror" / "7"
    for d in (src, mirror):
        d.mkdir(parents=True)
    Image.new("RGB", (100, 200), (10, 20, 30)).save(src / "7_0.jpeg")          # full image from the url
    Image.new("RGB", (40, 50), (1, 2, 3)).save(mirror / "7_1.jpg")              # released crop
    (src / "7_1.jpg").write_bytes((mirror / "7_1.jpg").read_bytes())
    Image.new("RGB", (100, 200)).save(src / "7_2.jpeg")                         # no detection line
    boxes = {"7_0": (0.1, 0.6, 0.25, 0.75), "7_1": (0.0, 1.0, 0.0, 1.0)}
    counts = c.crop_product(src, dst, boxes, mirror)
    assert counts == {"cropped": 1, "released": 1, "no_box": 1}
    with Image.open(dst / "7_0.jpeg") as im:
        assert im.size == (50, 100)
    assert (dst / "7_1.jpg").read_bytes() == (mirror / "7_1.jpg").read_bytes()
    with Image.open(dst / "7_2.jpeg") as im:
        assert im.size == (100, 200)  # kept whole when no box is known


def test_crop_product_is_resumable(tmp_path):
    import crop_f200k as c

    src, dst = tmp_path / "src" / "7", tmp_path / "dst" / "7"
    src.mkdir(parents=True)
    Image.new("RGB", (100, 200)).save(src / "7_0.jpeg")
    boxes = {"7_0": (0.0, 0.5, 0.0, 0.5)}
    c.crop_product(src, dst, boxes, None)
    assert c.crop_product(src, dst, boxes, None) == {"cropped": 0, "released": 0, "no_box": 0}


@pytest.mark.parametrize("box", [(0.5, 0.5, 0.0, 1.0), (0.9, 0.1, 0.0, 1.0)])
def test_degenerate_box_keeps_at_least_one_pixel(box):
    import crop_f200k as c

    left, top, right, bottom = c.pixel_box(box, (100, 100))
    assert right > left and bottom > top
