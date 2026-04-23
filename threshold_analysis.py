from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def generate_activity_potentials(num_nodes: int, u: float, rng: np.random.Generator) -> np.ndarray:
    eps = 1e-6
    return np.power(rng.uniform(eps, 1.0, size=num_nodes), 1.0 / max(u - 1.0, eps)).astype(np.float32)


def build_snapshot(base_adj: np.ndarray, activity: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    n = base_adj.shape[0]
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        if rng.random() > float(activity[i]):
            continue
        neighbors = np.where(base_adj[i] > 0)[0]
        neighbors = neighbors[neighbors != i]
        if len(neighbors) == 0:
            neighbors = np.array([j for j in range(n) if j != i], dtype=int)
        if len(neighbors) == 0:
            continue
        chosen = rng.choice(neighbors, size=min(int(m), len(neighbors)), replace=False)
        A[i, chosen] = 1.0
    return A


def transmission_matrix(A: np.ndarray, r: np.ndarray, phi: np.ndarray) -> np.ndarray:
    V = (r[:, None] * np.exp(-phi[None, :])).astype(np.float32) * A
    np.fill_diagonal(V, 0.0)
    return V


def proxy_y_from_pressure(V: np.ndarray) -> np.ndarray:
    pressure = V.sum(axis=0)
    return np.clip(1.0 - np.exp(-pressure), 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser(description='Threshold analysis aligned with STESP theorem')
    ap.add_argument('--num_nodes', type=int, default=5)
    ap.add_argument('--u', type=float, default=2.0)
    ap.add_argument('--m', type=int, default=1)
    ap.add_argument('--beta', type=float, default=0.6)
    ap.add_argument('--gamma', type=float, default=0.3)
    ap.add_argument('--avg_degree', type=float, default=2.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_dir', type=str, default='threshold_routeA')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    base_adj = np.ones((args.num_nodes, args.num_nodes), dtype=np.float32) - np.eye(args.num_nodes, dtype=np.float32)
    activity = generate_activity_potentials(args.num_nodes, args.u, rng)
    A = build_snapshot(base_adj, activity, args.m, rng)

    mu_r_values = np.linspace(0.2, 0.9, 25)
    mu_phi_values = np.linspace(0.2, 2.0, 25)
    Rtotal = np.zeros((len(mu_r_values), len(mu_phi_values)), dtype=np.float32)
    for i, mu_r in enumerate(mu_r_values):
        for j, mu_phi in enumerate(mu_phi_values):
            r = np.clip(rng.normal(mu_r, 0.05, size=args.num_nodes), 0.0, 1.0)
            phi = np.clip(rng.normal(mu_phi, 0.05, size=args.num_nodes), 0.0, None)
            V = transmission_matrix(A, r, phi)
            y = proxy_y_from_pressure(V)
            Ri0 = (y * args.beta * args.avg_degree) / args.gamma
            weights = np.ones(args.num_nodes, dtype=np.float32) / args.num_nodes
            Rtotal[i, j] = float((weights * Ri0).sum())

    X, Y = np.meshgrid(mu_phi_values, mu_r_values)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Rtotal, cmap='viridis', edgecolor='none')
    ax.set_xlabel(r'$\mu_\phi$')
    ax.set_ylabel(r'$\mu_r$')
    ax.set_zlabel(r'$R_0^{total}$')
    fig.colorbar(surf, shrink=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'threshold_surface.png'), dpi=300)
    plt.close(fig)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(Rtotal, origin='lower', aspect='auto', cmap='viridis', extent=[mu_phi_values.min(), mu_phi_values.max(), mu_r_values.min(), mu_r_values.max()])
    cs = plt.contour(mu_phi_values, mu_r_values, Rtotal, levels=[1.0], colors='white', linewidths=2.0)
    plt.clabel(cs, inline=True, fmt={1.0: 'R0=1'})
    plt.xlabel(r'$\mu_\phi$')
    plt.ylabel(r'$\mu_r$')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'threshold_heatmap.png'), dpi=300)


if __name__ == '__main__':
    main()
