# few_shot/test_dgl.py
import pytest

# Import DGL safely. If it's not importable (e.g., GraphBolt dylib missing on macOS),
# skip the entire module cleanly during collection.
try:
    import dgl  # noqa: F401
except Exception as e:
    pytest.skip(f"Skipping DGL tests on this platform/env: {e}", allow_module_level=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot

def test_dgl_importable():
    import dgl  # local import inside test for safety
    assert dgl is not None

def test_create_simple_graph():
    import dgl
    g = dgl.graph(([0, 1, 2], [1, 2, 0]))
    assert g.num_nodes() == 3

# ---- Below are lightweight class definitions (no heavy work at import time) ----

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super().__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_ft))
        else:
            self.register_parameter('bias', None)

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out = out + self.bias
        return self.act(out)

class AvgReadout(nn.Module):
    def forward(self, seq):  # [B, N, D]
        return torch.mean(seq, 1)

class MaxReadout(nn.Module):
    def forward(self, seq):
        return torch.max(seq, 1).values

class MinReadout(nn.Module):
    def forward(self, seq):
        return torch.min(seq, 1).values

class WSReadout(nn.Module):
    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, seq.size(-1))
        out = torch.mul(seq, sim)
        return torch.sum(out, 1)

class SimplePrompt(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.global_emb = nn.Parameter(torch.Tensor(1, input_size))
        self.a = nn.Linear(input_size, input_size)
        self.act = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.global_emb)
        self.a.reset_parameters()

    def forward(self, x):
        return x + self.act(self.a(x))

class Model_fine_tuning(nn.Module):
    def __init__(self, n_in_1, n_in_2, n_h, activation, negsamp_round, readout):
        super().__init__()
        self.fc_map = nn.Linear(n_in_1, n_in_2, bias=False)
        self.gcn1 = GCN(n_in_2, n_h, activation)
        self.gcn2 = GCN(n_h, n_h, activation)
        self.gcn3 = GCN(n_h, n_h, activation)
        self.fc1 = nn.Linear(n_h, 1, bias=False)
        self.fc2 = nn.Linear(n_h, 1, bias=False)
        self.fc_normal_prompt = nn.Linear(n_h, n_h, bias=False)
        self.fc_abnormal_prompt = nn.Linear(n_h, n_h, bias=False)
        self.prompt = SimplePrompt(300)
        self.act = nn.ReLU()

        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()
        else:
            self.read = AvgReadout()

    def forward(self, seq1, adj, raw_adj, normal_prompt, abnormal_prompt, sparse=False):
        h_1 = self.gcn1(seq1, adj, sparse)
        emb = self.gcn2(h_1, adj, sparse)

        normal_prompt = self.act(self.fc_normal_prompt(normal_prompt))
        abnormal_prompt = self.act(self.fc_abnormal_prompt(abnormal_prompt))

        # residual feature
        raw_adj = raw_adj * (1 - torch.eye(raw_adj.size(0), device=raw_adj.device))
        col_normalized = raw_adj.sum(1, keepdim=True)
        adj_normalized = raw_adj / torch.clamp(col_normalized, min=1e-12)

        emb_neighbors = torch.bmm(torch.unsqueeze(adj_normalized, 0), emb)
        emb_residual = emb - emb_neighbors

        logit = self.fc1(emb)
        logit_residual = self.fc2(emb_residual)

        return logit, logit_residual, emb, emb_residual, normal_prompt, abnormal_prompt, emb_neighbors
