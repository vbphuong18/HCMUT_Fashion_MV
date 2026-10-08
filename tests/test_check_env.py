import importlib.util
import json
import sys

import pytest
import torch

from conftest import ROOT


def _check_env():
    spec = importlib.util.spec_from_file_location("check_env", ROOT / "scripts" / "check_env.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def t4_without_kernels(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: (7, 5))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "Tesla T4")
    monkeypatch.setitem(sys.modules, "fla", None)  # import fla -> ImportError
    monkeypatch.setitem(sys.modules, "causal_conv1d", None)


def test_strict_mode_rejects_t4(t4_without_kernels):
    with pytest.raises(SystemExit, match="sm75"):
        _check_env().main()


def test_allow_slow_reports_instead_of_failing(t4_without_kernels, tmp_path):
    out = tmp_path / "env.json"
    _check_env().main(str(out), allow_slow=True)
    info = json.loads(out.read_text(encoding="utf-8"))
    assert info["capability"] == "sm75"
    assert info["kernels"] == {"fla": False, "causal_conv1d": False}
    assert len(info["warnings"]) == 2


def test_no_cuda_fails_even_when_slow_is_allowed(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="CUDA is not available"):
        _check_env().main(allow_slow=True)
