import os
import glob
import importlib
import pytest

def test_utils_module_importable():
    # ensure few_shot.utils can be imported
    spec = importlib.util.find_spec("few_shot.utils")
    assert spec is not None, "few_shot.utils not found"
    mod = importlib.import_module("few_shot.utils")
    assert hasattr(mod, "__file__"), "few_shot.utils imported but missing __file__"

def test_pretrained_weights_present():
    assert os.path.exists("pretrain/model_weights_abnormal300.pth"), "pretrain/model_weights_abnormal300.pth missing"

def test_mat_files_exist():
    mats = sorted(glob.glob("*.mat") + glob.glob("datasets/*.mat"))
    assert len(mats) > 0, "No .mat dataset files found in repo root or datasets/"

def test_mat_file_loadable():
    # try loading one .mat file to ensure readable format (requires scipy)
    mats = sorted(glob.glob("*.mat") + glob.glob("datasets/*.mat"))
    assert mats, "No .mat files to test"
    try:
        import scipy.io as scio
    except Exception as e:
        pytest.skip(f"scipy not installed: {e}")
    data = scio.loadmat(mats[0])
    assert isinstance(data, dict) and len(data) > 0, f"Failed to load .mat or empty: {mats[0]}"