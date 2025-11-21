"""Very small `glorot` initializer stub used by tests.

The real `glorot` performs parameter initialization; tests only import it
and call it in a few places during module import. This stub is a no-op that
accepts a tensor-like object.
"""
def glorot(tensor):
    """No-op glorot initializer used for tests when `torch_geometric` is not
    installed.
    """
    # Try to detect PyTorch tensors and leave them unchanged.
    try:
        import torch
        if isinstance(tensor, torch.Tensor):
            # mimic an in-place initialization if needed (no-op here)
            return
    except Exception:
        pass
    return
