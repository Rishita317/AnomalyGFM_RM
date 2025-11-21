# tests/conftest.py
import os

# stop DGL from trying to load the missing GraphBolt .dylib on macOS
os.environ.setdefault("DGL_SKIP_GRAPHBOLT", "1")

