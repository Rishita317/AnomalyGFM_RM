import pytest

def test_torch_importable():
    try:
        import torch
    except Exception as e:
        pytest.skip(f"torch not available: {e}")
    assert hasattr(torch, "__version__")

def test_optional_dgl_import_or_skip():
    try:
        import dgl  # optional heavy dep
    except Exception as e:
        pytest.skip(f"dgl not available: {e}")
    assert hasattr(dgl, "__version__")