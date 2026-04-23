from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mean_field import mean_field_one_step


@dataclass
class STESPModelConfig:
    num_nodes: int
    feature_dim: int
    gcn_hidden: int = 64
    tcn_hidden: int = 64
    graph_layers: int = 2
    tcn_layers: int = 3
    kernel_size: int = 3
    dropout: float = 0.2
    beta: float = 0.6
    gamma: float = 0.3
    dt: float = 1.0

    def asdict(self):
        return asdict(self)


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    b, n, _ = adj.shape
    eye = torch.eye(n, device=adj.device, dtype=adj.dtype).unsqueeze(0).expand(b, -1, -1)
    a_hat = adj + eye
    deg = a_hat.sum(-1).clamp_min(1e-8)
    inv_sqrt = deg.pow(-0.5)
    return inv_sqrt.unsqueeze(-1) * a_hat * inv_sqrt.unsqueeze(-2)


class EvolvingGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.state_dim = in_dim * out_dim
        self.gru = nn.GRUCell(in_dim, self.state_dim)
        self.h0 = nn.Parameter(torch.empty(self.state_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.h0.view(in_dim, out_dim))

    def forward(self, x_seq: torch.Tensor, adj_seq: torch.Tensor) -> torch.Tensor:
        b, t, n, _ = x_seq.shape
        h = self.h0.unsqueeze(0).expand(b, -1).contiguous()
        outs = []
        for step in range(t):
            xt = x_seq[:, step]
            summary = xt.mean(dim=1)
            h = self.gru(summary, h)
            w = h.view(b, self.in_dim, self.out_dim)
            support = torch.einsum('bnf,bfo->bno', xt, w)
            norm_adj = normalize_adjacency(adj_seq[:, step])
            yt = torch.bmm(norm_adj, support) + self.bias.view(1, 1, -1)
            outs.append(F.relu(yt))
        return torch.stack(outs, dim=1)


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    @staticmethod
    def _trim(x: torch.Tensor, trim: int) -> torch.Tensor:
        return x[..., :-trim] if trim > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self._trim(self.conv1(x), self.conv1.padding[0])
        x = self.dropout(F.relu(x))
        x = self._trim(self.conv2(x), self.conv2.padding[0])
        x = self.dropout(F.relu(x))
        x = x + res[..., -x.size(-1):]
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return x


class TemporalConvEncoder(nn.Module):
    def __init__(self, channels: int, layers: int, kernel_size: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList([TemporalBlock(channels, kernel_size, 2 ** i, dropout) for i in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class STESPModel(nn.Module):
    def __init__(self, cfg: STESPModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.feature_dim, cfg.gcn_hidden)
        self.graph_layers = nn.ModuleList([EvolvingGraphConv(cfg.gcn_hidden, cfg.gcn_hidden) for _ in range(cfg.graph_layers)])
        self.temporal = TemporalConvEncoder(cfg.gcn_hidden, cfg.tcn_layers, cfg.kernel_size, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)
        self.context_mlp = nn.Sequential(
            nn.Linear(2, cfg.gcn_hidden // 2),
            nn.ReLU(),
            nn.Linear(cfg.gcn_hidden // 2, cfg.gcn_hidden // 2),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.gcn_hidden + cfg.gcn_hidden // 2, cfg.tcn_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.tcn_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, adj_seq: torch.Tensor, x_seq: torch.Tensor, current_i: torch.Tensor, current_r: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x_seq)
        for layer in self.graph_layers:
            h = self.dropout(layer(h, adj_seq))
        b, t, n, c = h.shape
        temp_in = h.permute(0, 2, 3, 1).reshape(b * n, c, t)
        temp_out = self.temporal(temp_in)[:, :, -1].view(b, n, c)
        ctx = self.context_mlp(torch.stack([current_i, current_r], dim=-1))
        z = torch.cat([temp_out, ctx], dim=-1)
        return self.head(z).squeeze(-1)

    def predict_next(self, adj_seq, x_seq, current_i, current_r, population, avg_degree):
        y_prob = self.forward(adj_seq, x_seq, current_i, current_r)
        _, next_i, next_r = mean_field_one_step(y_prob, current_i, current_r, avg_degree, self.cfg.beta, self.cfg.gamma, self.cfg.dt)
        pred_counts = next_i * population
        return {'y_prob': y_prob, 'next_i_ratio': next_i, 'next_r_ratio': next_r, 'pred_counts': pred_counts}
