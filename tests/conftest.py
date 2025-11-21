# tests/conftest.py
import os

# DGL on macOS tries to load GraphBolt (native .dylib) which usually isn't present.
# This env var tells DGL to skip loading GraphBolt at import time.
os.environ.setdefault("DGL_SKIP_GRAPHBOLT", "1")
