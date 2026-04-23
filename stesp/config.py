from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class STESPConfig:
    data_dir: str = 'data'
    processed_dir: str = 'processed_stesp_routeA'
    output_dir: str = 'checkpoints_stesp_routeA'
    cases_file: str = 'spain-covid.txt'
    adj_file: str = 'spain-adj.txt'
    label_file: str = 'spain-label.csv'
    population_file: str = ''
    explicit_indices_file: str = ''

    num_regions: int = 35
    select_mode: str = 'topk_total_cases'
    window: int = 20
    horizon: int = 1
    train_ratio: float = 0.5
    val_ratio: float = 0.2
    test_ratio: float = 0.3
    seed: int = 42

    activity_index_u: float = 2.0
    connections_per_active_node: int = 1
    mu_r: float = 0.5
    sigma_r: float = 0.1
    mu_phi: float = 1.0
    sigma_phi: float = 0.1
    r_case_alpha: float = 0.25
    r_growth_alpha: float = 0.20
    phi_case_alpha: float = 0.15
    phi_growth_alpha: float = 0.25
    edge_temperature: float = 1.0

    beta: float = 0.6
    gamma: float = 0.3
    dt: float = 1.0
    avg_degree_value: float = 2.0
    population_scale: float = 15.0
    min_population: int = 5000

    gcn_hidden: int = 64
    tcn_hidden: int = 64
    graph_layers: int = 2
    tcn_layers: int = 3
    kernel_size: int = 3
    dropout: float = 0.2
    batch_size: int = 16
    epochs: int = 220
    lr: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 40
    grad_clip: float = 5.0
    lambda_y: float = 0.20
    lambda_smooth: float = 1e-3
    device: str = 'cuda'

    def asdict(self) -> Dict:
        return asdict(self)
