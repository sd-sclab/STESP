from __future__ import annotations

import numpy as np
import torch


def compute_recovered_ratio_from_cases(cases: np.ndarray, gamma: float, dt: float, population: np.ndarray) -> np.ndarray:
    i_ratio = np.clip(cases / population[None, :], 0.0, 1.0)
    r_ratio = np.zeros_like(i_ratio, dtype=np.float32)
    for t in range(1, len(i_ratio)):
        r_ratio[t] = np.clip(r_ratio[t - 1] + gamma * i_ratio[t - 1] * dt, 0.0, 1.0)
    return r_ratio.astype(np.float32)


def compute_target_infection_probability(current_i_ratio: np.ndarray, current_r_ratio: np.ndarray, next_i_ratio: np.ndarray, beta: float, avg_degree: np.ndarray, gamma: float, dt: float) -> np.ndarray:
    s_ratio = np.clip(1.0 - current_i_ratio - current_r_ratio, 0.0, 1.0)
    di_dt = (next_i_ratio - current_i_ratio) / max(float(dt), 1e-8)
    denom = np.clip(beta * avg_degree * current_i_ratio * s_ratio, 1e-8, None)
    y = (di_dt + gamma * current_i_ratio) / denom
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def mean_field_one_step(y_prob: torch.Tensor, current_i_ratio: torch.Tensor, current_r_ratio: torch.Tensor, avg_degree: torch.Tensor, beta: float, gamma: float, dt: float):
    s_ratio = torch.clamp(1.0 - current_i_ratio - current_r_ratio, min=0.0, max=1.0)
    flow = y_prob * beta * avg_degree * current_i_ratio * s_ratio
    next_s = torch.clamp(s_ratio - dt * flow, 0.0, 1.0)
    next_i = torch.clamp(current_i_ratio + dt * (flow - gamma * current_i_ratio), 0.0, 1.0)
    next_r = torch.clamp(current_r_ratio + dt * (gamma * current_i_ratio), 0.0, 1.0)
    return next_s, next_i, next_r
