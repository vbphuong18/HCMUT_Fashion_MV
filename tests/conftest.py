import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "FashionMV"))

BASE_MODEL = "Qwen/Qwen3.5-0.8B"


@pytest.fixture(scope="session")
def processor():
    from transformers import AutoProcessor

    from procir_train.prompts import ensure_emb_token

    try:
        proc = AutoProcessor.from_pretrained(BASE_MODEL)
    except (OSError, ValueError) as e:
        pytest.skip(f"Qwen3.5 processor unavailable: {e}")
    proc.tokenizer.padding_side = "right"
    ensure_emb_token(proc.tokenizer)
    return proc


@pytest.fixture
def make_images(tmp_path):
    from PIL import Image

    def _make(rel_dir, n, size=(64, 96)):
        d = tmp_path / rel_dir
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            Image.new("RGB", size, ((40 * i) % 255, 100, 150)).save(d / f"{i:02d}.jpg")
        return d

    return _make
