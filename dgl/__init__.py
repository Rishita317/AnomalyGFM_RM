"""Lightweight local stub of a tiny subset of the DGL API used by tests.

This is NOT the real DGL library. It only implements a minimal `graph`
constructor and a tiny Graph object with `num_nodes()` so the test-suite can
run in environments where installing full DGL (and GraphBolt) is infeasible.

Keep this file small and explicitly documented so it can be removed when real
DGL is available in CI or developer machines.
"""
__all__ = ["__version__", "graph"]

__version__ = "0.0-stub"

class Graph:
    def __init__(self, edges):
        # edges: tuple (src_list, dst_list) or (src, dst)
        try:
            src, dst = edges
        except Exception:
            raise TypeError("graph expects a tuple (src_list, dst_list)")
        # accept lists/tuples/iterables
        self.src = list(src)
        self.dst = list(dst)

    def num_nodes(self):
        if not self.src and not self.dst:
            return 0
        # number of nodes is max index + 1
        max_idx = max(self.src + self.dst)
        return int(max_idx) + 1

def graph(edges):
    """Construct a small Graph object.

    Mirrors the small-surface API used by the tests (dgl.graph(...)).
    """
    return Graph(edges)
