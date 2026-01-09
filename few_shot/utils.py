# few_shot/utils.py
# Safe to import on macOS without GraphBolt: never import dgl at module import time.

import os
os.environ.setdefault("DGL_SKIP_GRAPHBOLT", "1")  # belt & suspenders; fine to keep

import random
from collections import Counter

import numpy as np
import networkx as nx
import scipy.io as sio
import scipy.sparse as sp
import torch

# ----------------------------
# Generic utils (no DGL)
# ----------------------------

def sparse_to_tuple(sparse_mx, insert_batch: bool = False):
    """Convert scipy.sparse matrix (or list of them) to (coords, values, shape).
    Set insert_batch=True to insert a fake batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).T
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).T
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
        return sparse_mx
    else:
        return to_tuple(sparse_mx)


def preprocess_features(features: sp.spmatrix):
    """Row-normalize feature matrix and return (dense_features, sparse_tuple)."""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def min_max_normalize(tensor: torch.Tensor) -> torch.Tensor:
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val + 1e-12)


def preprocess_features_tensor(features: torch.Tensor) -> torch.Tensor:
    rowsum = torch.sum(features)
    r_inv = torch.pow(rowsum, -1)
    r_inv[torch.isinf(r_inv)] = 0.0
    return features * r_inv


def normalize_adj(adj: sp.spmatrix) -> sp.coo_matrix:
    """Symmetrically normalize adjacency matrix D^{-1/2} A D^{-1/2}."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).T.dot(d_mat_inv_sqrt).tocoo()


def dense_to_one_hot(labels_dense: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def harmonic_mean_fusion(score1, score2, epsilon=1e-6):
    return 2 * score1 * score2 / (score1 + score2 + epsilon)


def min_fusion(score1, score2):
    return np.minimum(score1, score2)


def max_fusion(score1, score2):
    return np.maximum(score1, score2)


def z_score_normalize(vector):
    mean = np.mean(vector)
    std = np.std(vector) + 1e-12
    return (vector - mean) / std


def min_max(vector):
    min_val = np.min(vector)
    max_val = np.max(vector)
    return (vector - min_val) / (max_val - min_val + 1e-12)


def load_mat(dataset_train: str, dataset_test: str, train_rate: float = 0.3, val_rate: float = 0.1):
    """Load .mat datasets from data/<name>.mat (as your current code expects)."""
    data_train = sio.loadmat(f"data/{dataset_train}.mat")
    data_test  = sio.loadmat(f"data/{dataset_test}.mat")

    label_train = data_train['Label'] if ('Label' in data_train) else data_train['gnd']
    attr_train  = data_train['Attributes'] if ('Attributes' in data_train) else data_train['X']
    network_train = data_train['Network'] if ('Network' in data_train) else data_train['A']

    label_test = data_test['Label'] if ('Label' in data_test) else data_test['gnd']
    attr_test  = data_test['Attributes'] if ('Attributes' in data_test) else data_test['X']
    network_test = data_test['Network'] if ('Network' in data_test) else data_test['A']

    adj_train  = sp.csr_matrix(network_train)
    feat_train = sp.lil_matrix(attr_train)

    adj_test  = sp.csr_matrix(network_test)
    feat_test = sp.lil_matrix(attr_test)

    all_labels      = np.squeeze(np.array(label_train))
    ano_labels_test = np.squeeze(np.array(label_test))

    num_node_train = adj_train.shape[0]
    num_train = int(num_node_train * train_rate)
    num_val   = int(num_node_train * val_rate)

    all_idx_train = list(range(num_node_train))
    random.shuffle(all_idx_train)

    idx_train = all_idx_train[:num_train]
    idx_val   = all_idx_train[num_train : num_train + num_val]

    labels_train = all_labels[idx_train]
    labels_val   = all_labels[idx_val]

    print('Test', Counter(np.squeeze(ano_labels_test)))

    return adj_train, adj_test, feat_train, feat_test, labels_train, labels_val, ano_labels_test, idx_train, idx_val


# ----------------------------
# DGL-dependent helpers (lazy)
# ----------------------------

def _require_dgl():
    """Import DGL lazily and return it, raising a helpful error if not available."""
    try:
        import dgl  # noqa: WPS433
        return dgl
    except Exception as e:
        raise ImportError(
            "DGL is required for this function but failed to import. "
            "On macOS, set DGL_SKIP_GRAPHBOLT=1 before importing DGL, or run on Linux."
        ) from e


def adj_to_dgl_graph(adj: sp.spmatrix):
    """Convert a scipy.sparse adjacency matrix to a DGLGraph (without importing DGL at module import)."""
    dgl = _require_dgl()
    # networkx API changed; try old then new
    try:
        nx_graph = nx.from_scipy_sparse_matrix(adj)
    except Exception:
        nx_graph = nx.from_scipy_sparse_array(adj)
    if hasattr(dgl, "from_networkx"):
        return dgl.from_networkx(nx_graph)
    return dgl.DGLGraph(nx_graph)  # very old DGL fallback


def generate_rwr_subgraph(dgl_graph, subgraph_size: int):
    """Generate subgraph with Random Walk w/ Restart (legacy contrib API if present)."""
    dgl = _require_dgl()
    all_idx = list(range(int(dgl_graph.number_of_nodes())))
    reduced_size = max(1, subgraph_size - 1)

    # legacy API path (as in your original code)
    if hasattr(dgl, "contrib") and hasattr(dgl.contrib, "sampling") and \
       hasattr(dgl.contrib.sampling, "random_walk_with_restart"):
        traces = dgl.contrib.sampling.random_walk_with_restart(
            dgl_graph, all_idx, restart_prob=1, max_nodes_per_seed=subgraph_size * 3
        )
        subv = []
        for i, trace in enumerate(traces):
            nodes = torch.unique(torch.cat(trace), sorted=False).tolist()
            retry_time = 0
            while len(nodes) < reduced_size:
                cur_trace = dgl.contrib.sampling.random_walk_with_restart(
                    dgl_graph, [i], restart_prob=0.9, max_nodes_per_seed=subgraph_size * 5
                )
                nodes = torch.unique(torch.cat(cur_trace[0]), sorted=False).tolist()
                retry_time += 1
                if (len(nodes) <= 2) and (retry_time > 10):
                    nodes = nodes * reduced_size
                    break
            nodes = nodes[: reduced_size * 3]
            nodes.append(i)
            subv.append(nodes)
        return subv

    # If contrib sampling is missing (newer DGL), raise a clear message.
    raise NotImplementedError(
        "generate_rwr_subgraph relies on dgl.contrib.sampling.random_walk_with_restart "
        "(removed in newer DGL). Use an environment with older DGL (Linux wheel) or "
        "replace this with the modern sampling API."
    )


# ----------------------------
# Plotting (no DGL)
# ----------------------------

# Delay heavy plotting imports until plotting functions are used. Importing
# matplotlib at module import time pulls in fontTools which currently emits a
# DeprecationWarning; if pytest is run with warnings-as-errors that breaks test
# collection. Import plotting libraries lazily in the functions below.


def draw_pdf(message_normal, message_abnormal, message_real_abnormal, dataset: str, epoch: int):
    # Import plotting libraries lazily to avoid import-time side-effects
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.mlab as mlab  # deprecated but retained for parity
    from matplotlib.backends.backend_pdf import PdfPages  # noqa: F401

    message_all = [np.squeeze(message_normal), np.squeeze(message_abnormal), np.squeeze(message_real_abnormal)]
    mu_0, sigma_0 = float(np.mean(message_all[0])), float(np.std(message_all[0]))
    mu_1, sigma_1 = float(np.mean(message_all[1])), float(np.std(message_all[1]))
    mu_2, sigma_2 = float(np.mean(message_all[2])), float(np.std(message_all[2]))

    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['figure.figsize'] = (8.5, 7.5)

    n, bins, patches = plt.hist(message_all, bins=30, density=True, label=['Normal', 'Outlier', 'Abnormal'])
    y_0 = mlab.normpdf(bins, mu_0, sigma_0)  # type: ignore[attr-defined]
    y_1 = mlab.normpdf(bins, mu_1, sigma_1)  # type: ignore[attr-defined]
    y_2 = mlab.normpdf(bins, mu_2, sigma_2)  # type: ignore[attr-defined]

    plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=2.5)
    plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=2.5)
    plt.plot(bins, y_2, color='green', linestyle='--', linewidth=2.5)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.legend(loc='upper left', fontsize=12)
    os.makedirs(f'fig/{dataset}', exist_ok=True)
    plt.savefig(f'fig/{dataset}/{dataset}_{epoch}.svg')
    plt.close()


def draw_pdf_methods(method: str, message_normal, message_abnormal, message_real_abnormal, dataset: str, epoch: int):
    # Import plotting libraries lazily to avoid import-time side-effects
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.mlab as mlab  # deprecated but retained for parity
    from matplotlib.backends.backend_pdf import PdfPages  # noqa: F401

    message_all = [np.squeeze(message_normal), np.squeeze(message_abnormal), np.squeeze(message_real_abnormal)]
    mu_0, sigma_0 = float(np.mean(message_all[0])), float(np.std(message_all[0]))
    mu_1, sigma_1 = float(np.mean(message_all[1])), float(np.std(message_all[1]))
    mu_2, sigma_2 = float(np.mean(message_all[2])), float(np.std(message_all[2]))

    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['figure.figsize'] = (8.5, 7.5)

    n, bins, patches = plt.hist(message_all, bins=30, density=True, label=['Normal', 'Outlier', 'Abnormal'])
    y_0 = mlab.normpdf(bins, mu_0, sigma_0)  # type: ignore[attr-defined]
    y_1 = mlab.normpdf(bins, mu_1, sigma_1)  # type: ignore[attr-defined]
    y_2 = mlab.normpdf(bins, mu_2, sigma_2)  # type: ignore[attr-defined]

    plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=2.5)
    plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=2.5)
    plt.plot(bins, y_2, color='green', linestyle='--', linewidth=2.5)
    plt.ylim(0, 8)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    os.makedirs(f'fig/{method}/{dataset}2', exist_ok=True)
    plt.savefig(f'fig/{method}/{dataset}2/{dataset}_{epoch}.svg')
    plt.close()
