from __future__ import annotations

import csv
import json
import os
import urllib.request
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import STESPConfig
from .mean_field import compute_recovered_ratio_from_cases, compute_target_infection_probability

RAW_URLS = {
    'spain-covid.txt': 'https://raw.githubusercontent.com/Xiefeng69/EpiGNN/main/data/spain-covid.txt',
    'spain-adj.txt': 'https://raw.githubusercontent.com/Xiefeng69/EpiGNN/main/data/spain-adj.txt',
    'spain-label.csv': 'https://raw.githubusercontent.com/Xiefeng69/EpiGNN/main/data/spain-label.csv',
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def maybe_download_raw(data_dir: str) -> None:
    ensure_dir(data_dir)
    for name, url in RAW_URLS.items():
        dst = os.path.join(data_dir, name)
        if os.path.exists(dst):
            continue
        try:
            urllib.request.urlretrieve(url, dst)
        except Exception:
            pass


def load_labels(path: str) -> List[str]:
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.reader(f):
            if row:
                labels.append(str(row[0]))
    return labels


def load_raw(cfg: STESPConfig) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    cases = np.loadtxt(os.path.join(cfg.data_dir, cfg.cases_file), delimiter=',').astype(np.float32)
    adj = np.loadtxt(os.path.join(cfg.data_dir, cfg.adj_file), delimiter=',').astype(np.float32)
    labels = load_labels(os.path.join(cfg.data_dir, cfg.label_file))
    if cases.ndim != 2:
        raise ValueError('Cases must be [T, N].')
    if adj.shape[0] != adj.shape[1] or adj.shape[0] != cases.shape[1]:
        raise ValueError('Adjacency shape mismatch.')
    if len(labels) != cases.shape[1]:
        raise ValueError('Label count mismatch.')
    return cases, adj, labels


def select_regions(cases: np.ndarray, adj: np.ndarray, labels: List[str], cfg: STESPConfig):
    if cfg.select_mode == 'explicit_indices_file' and cfg.explicit_indices_file:
        idx = np.loadtxt(cfg.explicit_indices_file, dtype=int).reshape(-1)
    else:
        idx = np.argsort(-cases.sum(axis=0))[: cfg.num_regions]
        idx = np.sort(idx)
    if len(idx) != cfg.num_regions:
        raise ValueError(f'Expected {cfg.num_regions} regions, got {len(idx)}')
    return cases[:, idx], adj[np.ix_(idx, idx)], [labels[i] for i in idx], idx.astype(int)


def derive_population(cases: np.ndarray, cfg: STESPConfig) -> np.ndarray:
    if cfg.population_file:
        pop = np.loadtxt(cfg.population_file, delimiter=',').astype(np.float32).reshape(-1)
        if len(pop) != cases.shape[1]:
            raise ValueError('Population length mismatch.')
        return np.maximum(pop, cases.max(axis=0) + 1.0)
    peak = cases.max(axis=0)
    pop = np.maximum(np.ceil(peak * cfg.population_scale), cfg.min_population).astype(np.float32)
    return np.maximum(pop, peak + 1.0)


def zscore_per_timestep(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return ((x - mu) / (std + 1e-8)).astype(np.float32)


def compute_case_features(cases: np.ndarray, population: np.ndarray) -> Dict[str, np.ndarray]:
    i_ratio = np.clip(cases / population[None, :], 0.0, 1.0).astype(np.float32)
    growth = np.zeros_like(i_ratio, dtype=np.float32)
    growth[1:] = (cases[1:] - cases[:-1]) / np.maximum(cases[:-1], 1.0)
    growth = np.clip(growth, -5.0, 5.0).astype(np.float32)
    log_cases = np.log1p(cases).astype(np.float32)
    return {
        'i_ratio': i_ratio,
        'growth': growth,
        'z_i_ratio': zscore_per_timestep(i_ratio),
        'z_growth': zscore_per_timestep(growth),
        'z_log_cases': zscore_per_timestep(log_cases),
    }


def generate_activity_potentials(n: int, u: float, rng: np.random.Generator) -> np.ndarray:
    eps = 1e-6
    return np.power(rng.uniform(eps, 1.0, size=n), 1.0 / max(u - 1.0, eps)).astype(np.float32)


def data_informed_attributes(case_feats: Dict[str, np.ndarray], activity: np.ndarray, cfg: STESPConfig, rng: np.random.Generator):
    T, N = case_feats['i_ratio'].shape
    r_seq = np.zeros((T, N), dtype=np.float32)
    phi_seq = np.zeros((T, N), dtype=np.float32)
    for t in range(T):
        noise_r = rng.normal(0.0, cfg.sigma_r, size=N).astype(np.float32)
        noise_phi = rng.normal(0.0, cfg.sigma_phi, size=N).astype(np.float32)
        r_t = cfg.mu_r + cfg.r_case_alpha * case_feats['z_i_ratio'][t] + cfg.r_growth_alpha * np.maximum(case_feats['z_growth'][t], 0.0) + 0.10 * activity + noise_r
        phi_t = cfg.mu_phi + cfg.phi_case_alpha * np.maximum(case_feats['z_i_ratio'][t], 0.0) + cfg.phi_growth_alpha * np.maximum(case_feats['z_growth'][t], 0.0) + noise_phi
        r_seq[t] = np.clip(r_t, 0.0, 1.0)
        phi_seq[t] = np.clip(phi_t, 0.0, None)
    return r_seq, phi_seq


def build_snapshot_adjacency(base_adj: np.ndarray, activity: np.ndarray, cases_t: np.ndarray, temperature: float, m: int, rng: np.random.Generator) -> np.ndarray:
    n = base_adj.shape[0]
    A = np.zeros((n, n), dtype=np.float32)
    case_score = cases_t + 1e-6
    case_score = case_score / np.maximum(case_score.sum(), 1e-8)
    for i in range(n):
        if rng.random() > float(activity[i]):
            continue
        neighbors = np.where(base_adj[i] > 0)[0]
        neighbors = neighbors[neighbors != i]
        if len(neighbors) == 0:
            neighbors = np.array([j for j in range(n) if j != i], dtype=int)
        if len(neighbors) == 0:
            continue
        weights = case_score[neighbors] ** max(float(temperature), 1e-6)
        weights = weights / np.maximum(weights.sum(), 1e-8)
        k = min(int(m), len(neighbors))
        chosen = rng.choice(neighbors, size=k, replace=False, p=weights)
        A[i, chosen] = 1.0
    return A


def build_transmission_matrix(A: np.ndarray, r_t: np.ndarray, phi_t: np.ndarray) -> np.ndarray:
    V = (r_t[:, None] * np.exp(-phi_t[None, :])).astype(np.float32)
    V *= A
    np.fill_diagonal(V, 0.0)
    return V


def build_dynamic_sequences(base_adj: np.ndarray, case_feats: Dict[str, np.ndarray], cfg: STESPConfig):
    rng = np.random.default_rng(cfg.seed)
    T, N = case_feats['i_ratio'].shape
    activity = generate_activity_potentials(N, cfg.activity_index_u, rng)
    r_seq, phi_seq = data_informed_attributes(case_feats, activity, cfg, rng)
    A_seq = np.zeros((T, N, N), dtype=np.float32)
    X_seq = np.zeros((T, N, N + 5), dtype=np.float32)
    for t in range(T):
        A_t = build_snapshot_adjacency(base_adj, activity, case_feats['i_ratio'][t], cfg.edge_temperature, cfg.connections_per_active_node, rng)
        V_t = build_transmission_matrix(A_t, r_seq[t], phi_seq[t])
        incoming_vec = V_t.T.copy()
        node_context = np.stack([
            case_feats['i_ratio'][t],
            case_feats['growth'][t],
            np.full((N,), float(activity.mean()), dtype=np.float32),
            r_seq[t],
            phi_seq[t],
        ], axis=-1).astype(np.float32)
        X_seq[t] = np.concatenate([incoming_vec, node_context], axis=-1).astype(np.float32)
        A_seq[t] = A_t
    attrs = {'activity': activity.astype(np.float32), 'interaction_r': r_seq.astype(np.float32), 'mitigation_phi': phi_seq.astype(np.float32)}
    return A_seq, X_seq, attrs


def split_boundaries(T: int, cfg: STESPConfig):
    return int(round(T * cfg.train_ratio)), int(round(T * (cfg.train_ratio + cfg.val_ratio)))


class STESPDataset(Dataset):
    def __init__(self, npz_path: str):
        d = np.load(npz_path)
        self.adj_seq = d['adj_seq'].astype(np.float32)
        self.x_seq = d['x_seq'].astype(np.float32)
        self.current_i = d['current_i'].astype(np.float32)
        self.current_r = d['current_r'].astype(np.float32)
        self.target_i = d['target_i'].astype(np.float32)
        self.target_counts = d['target_counts'].astype(np.float32)
        self.target_y = d['target_y'].astype(np.float32)
        self.target_time = d['target_time'].astype(np.int64)

    def __len__(self):
        return len(self.target_time)

    def __getitem__(self, idx):
        return {
            'adj_seq': torch.from_numpy(self.adj_seq[idx]),
            'x_seq': torch.from_numpy(self.x_seq[idx]),
            'current_i': torch.from_numpy(self.current_i[idx]),
            'current_r': torch.from_numpy(self.current_r[idx]),
            'target_i': torch.from_numpy(self.target_i[idx]),
            'target_counts': torch.from_numpy(self.target_counts[idx]),
            'target_y': torch.from_numpy(self.target_y[idx]),
            'target_time': torch.tensor(int(self.target_time[idx]), dtype=torch.long),
        }


def collate_batch(batch):
    return {k: torch.stack([item[k] for item in batch], dim=0) for k in batch[0].keys()}


def build_processed_dataset(cfg: STESPConfig):
    ensure_dir(cfg.processed_dir)
    maybe_download_raw(cfg.data_dir)
    cases_raw, adj_raw, labels_raw = load_raw(cfg)
    cases, base_adj, labels, region_idx = select_regions(cases_raw, adj_raw, labels_raw, cfg)
    if cases.shape[0] != 122:
        raise ValueError(f'Expected 122 days, got {cases.shape[0]}')
    if cfg.horizon != 1:
        raise ValueError('This implementation keeps horizon=1 because the paper uses one-step SIR coupling.')
    population = derive_population(cases, cfg)
    avg_degree = np.full((cfg.num_regions,), cfg.avg_degree_value, dtype=np.float32)
    case_feats = compute_case_features(cases, population)
    recovered = compute_recovered_ratio_from_cases(cases, cfg.gamma, cfg.dt, population)
    A_seq, X_seq, _ = build_dynamic_sequences(base_adj, case_feats, cfg)
    T = cases.shape[0]
    train_end, val_end = split_boundaries(T, cfg)
    buffers = {split: {k: [] for k in ['adj_seq', 'x_seq', 'current_i', 'current_r', 'target_i', 'target_counts', 'target_y', 'target_time']} for split in ['train', 'val', 'test']}
    for target_idx in range(cfg.window, T):
        hist_start, hist_end = target_idx - cfg.window, target_idx
        current_day = hist_end - 1
        target_day = target_idx
        split = 'train' if target_day < train_end else 'val' if target_day < val_end else 'test'
        current_i = case_feats['i_ratio'][current_day]
        current_r = recovered[current_day]
        next_i = case_feats['i_ratio'][target_day]
        target_y = compute_target_infection_probability(current_i, current_r, next_i, cfg.beta, avg_degree, cfg.gamma, cfg.dt)
        buffers[split]['adj_seq'].append(A_seq[hist_start:hist_end])
        buffers[split]['x_seq'].append(X_seq[hist_start:hist_end])
        buffers[split]['current_i'].append(current_i)
        buffers[split]['current_r'].append(current_r)
        buffers[split]['target_i'].append(next_i)
        buffers[split]['target_counts'].append(cases[target_day].astype(np.float32))
        buffers[split]['target_y'].append(target_y)
        buffers[split]['target_time'].append(np.array(target_day, dtype=np.int64))

    out_paths = {}
    for split, bundle in buffers.items():
        dst = os.path.join(cfg.processed_dir, f'{split}.npz')
        np.savez_compressed(
            dst,
            adj_seq=np.asarray(bundle['adj_seq'], dtype=np.float32),
            x_seq=np.asarray(bundle['x_seq'], dtype=np.float32),
            current_i=np.asarray(bundle['current_i'], dtype=np.float32),
            current_r=np.asarray(bundle['current_r'], dtype=np.float32),
            target_i=np.asarray(bundle['target_i'], dtype=np.float32),
            target_counts=np.asarray(bundle['target_counts'], dtype=np.float32),
            target_y=np.asarray(bundle['target_y'], dtype=np.float32),
            target_time=np.asarray(bundle['target_time'], dtype=np.int64),
        )
        out_paths[split] = dst

    meta = {
        'num_regions': int(cfg.num_regions),
        'feature_dim': int(X_seq.shape[-1]),
        'population': population.tolist(),
        'avg_degree': avg_degree.tolist(),
        'selected_labels': labels,
        'selected_indices': region_idx.tolist(),
        'config': cfg.asdict(),
        'base_adj_density': float((base_adj > 0).mean()),
        'raw_shape': [int(cases_raw.shape[0]), int(cases_raw.shape[1])],
    }
    meta_path = os.path.join(cfg.processed_dir, 'meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    out_paths['meta'] = meta_path
    return out_paths


def load_meta(processed_dir: str):
    with open(os.path.join(processed_dir, 'meta.json'), 'r', encoding='utf-8') as f:
        return json.load(f)
