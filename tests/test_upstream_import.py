import subprocess

from conftest import ROOT


def test_upstream_commit_is_pinned():
    head = subprocess.run(
        ["git", "-C", str(ROOT / "external" / "FashionMV"), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert head == "1c2f05cf6b4e166160e8f8e210f536e8cc8a08a7"


def test_upstream_think_patch():
    from procir.chat_utils import patch_think_tokens

    out = patch_think_tokens("<|im_start|>assistant\nX")
    assert out == "<|im_start|>assistant\n<think>\n\n</think>\n\nX"
