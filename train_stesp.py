from __future__ import annotations

import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from stesp.config import STESPConfig
from stesp.data import STESPDataset, build_processed_dataset, collate_batch, ensure_dir, load_meta
from stesp.model import STESPModel, STESPModelConfig


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metric_pack(pred_counts, true_counts, pred_ratio, true_ratio):
    pred_flat = pred_counts.reshape(-1)
    true_flat = true_counts.reshape(-1)
    pred_r = pred_ratio.reshape(-1)
    true_r = true_ratio.reshape(-1)
    mae = float(np.mean(np.abs(pred_flat - true_flat)))
    rmse = float(np.sqrt(np.mean((pred_flat - true_flat) ** 2)))
    mape = float(np.mean(np.abs(pred_flat - true_flat) / np.maximum(np.abs(true_flat), 1.0)))
    smape = float(np.mean(2.0 * np.abs(pred_flat - true_flat) / np.maximum(np.abs(pred_flat) + np.abs(true_flat), 1.0)))
    ratio_rmse = float(np.sqrt(np.mean((pred_r - true_r) ** 2)))
    pcc = 0.0 if np.std(pred_flat) < 1e-8 or np.std(true_flat) < 1e-8 else float(np.corrcoef(pred_flat, true_flat)[0, 1])
    accuracy = float(max(0.0, 1.0 - ratio_rmse))
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'smape': smape, 'ratio_rmse': ratio_rmse, 'pcc': pcc, 'accuracy': accuracy}


def run_epoch(model, loader, optimizer, device, population, avg_degree, cfg: STESPConfig):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_n = 0
    pred_counts_all, true_counts_all, pred_ratio_all, true_ratio_all, pred_y_all, true_y_all = [], [], [], [], [], []
    for batch in loader:
        adj_seq = batch['adj_seq'].to(device)
        x_seq = batch['x_seq'].to(device)
        current_i = batch['current_i'].to(device)
        current_r = batch['current_r'].to(device)
        target_i = batch['target_i'].to(device)
        target_counts = batch['target_counts'].to(device)
        target_y = batch['target_y'].to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        out = model.predict_next(
            adj_seq=adj_seq,
            x_seq=x_seq,
            current_i=current_i,
            current_r=current_r,
            population=population.unsqueeze(0).expand(adj_seq.size(0), -1),
            avg_degree=avg_degree.unsqueeze(0).expand(adj_seq.size(0), -1),
        )
        pred_counts = out['pred_counts']
        pred_i = out['next_i_ratio']
        pred_y = out['y_prob']

        loss_count = F.mse_loss(pred_i, target_i)
        loss_y = F.mse_loss(pred_y, target_y)
        smooth_reg = ((pred_y[:, 1:] - pred_y[:, :-1]) ** 2).mean()
        loss = loss_count + cfg.lambda_y * loss_y + cfg.lambda_smooth * smooth_reg

        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        bs = adj_seq.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs
        pred_counts_all.append(pred_counts.detach().cpu().numpy())
        true_counts_all.append(target_counts.detach().cpu().numpy())
        pred_ratio_all.append(pred_i.detach().cpu().numpy())
        true_ratio_all.append(target_i.detach().cpu().numpy())
        pred_y_all.append(pred_y.detach().cpu().numpy())
        true_y_all.append(target_y.detach().cpu().numpy())

    pred_counts_np = np.concatenate(pred_counts_all, axis=0) if pred_counts_all else np.empty((0, population.numel()), dtype=np.float32)
    true_counts_np = np.concatenate(true_counts_all, axis=0) if true_counts_all else np.empty((0, population.numel()), dtype=np.float32)
    pred_ratio_np = np.concatenate(pred_ratio_all, axis=0) if pred_ratio_all else np.empty((0, population.numel()), dtype=np.float32)
    true_ratio_np = np.concatenate(true_ratio_all, axis=0) if true_ratio_all else np.empty((0, population.numel()), dtype=np.float32)
    pred_y_np = np.concatenate(pred_y_all, axis=0) if pred_y_all else np.empty((0, population.numel()), dtype=np.float32)
    true_y_np = np.concatenate(true_y_all, axis=0) if true_y_all else np.empty((0, population.numel()), dtype=np.float32)
    metrics = metric_pack(pred_counts_np, true_counts_np, pred_ratio_np, true_ratio_np)
    return total_loss / max(total_n, 1), metrics, pred_counts_np, true_counts_np, pred_ratio_np, true_ratio_np, pred_y_np, true_y_np


def save_curves(output_dir, train_losses, val_losses, val_accs):
    x = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(x, train_losses, label='Train Loss')
    plt.plot(x, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig5a_loss_curve.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(x, val_accs, label='Accuracy-like (1-ratio_rmse)')
    plt.xlabel('Epoch')
    plt.ylabel('Prediction Accuracy')
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig5b_accuracy_curve.png'), dpi=300)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(description='Route-A paper-aligned STESP training')
    p.add_argument('--data_dir', type=str, default='data')
    p.add_argument('--processed_dir', type=str, default='processed_stesp_routeA')
    p.add_argument('--output_dir', type=str, default='checkpoints_stesp_routeA')
    p.add_argument('--cases_file', type=str, default='spain-covid.txt')
    p.add_argument('--adj_file', type=str, default='spain-adj.txt')
    p.add_argument('--label_file', type=str, default='spain-label.csv')
    p.add_argument('--population_file', type=str, default='')
    p.add_argument('--explicit_indices_file', type=str, default='')
    p.add_argument('--rebuild_dataset', action='store_true')

    p.add_argument('--num_regions', type=int, default=35)
    p.add_argument('--select_mode', type=str, default='topk_total_cases', choices=['topk_total_cases', 'explicit_indices_file'])
    p.add_argument('--window', type=int, default=20)
    p.add_argument('--horizon', type=int, default=1)
    p.add_argument('--train_ratio', type=float, default=0.5)
    p.add_argument('--val_ratio', type=float, default=0.2)
    p.add_argument('--test_ratio', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=42)

    p.add_argument('--activity_index_u', type=float, default=2.0)
    p.add_argument('--connections_per_active_node', type=int, default=1)
    p.add_argument('--mu_r', type=float, default=0.5)
    p.add_argument('--sigma_r', type=float, default=0.1)
    p.add_argument('--mu_phi', type=float, default=1.0)
    p.add_argument('--sigma_phi', type=float, default=0.1)
    p.add_argument('--r_case_alpha', type=float, default=0.25)
    p.add_argument('--r_growth_alpha', type=float, default=0.20)
    p.add_argument('--phi_case_alpha', type=float, default=0.15)
    p.add_argument('--phi_growth_alpha', type=float, default=0.25)
    p.add_argument('--edge_temperature', type=float, default=1.0)

    p.add_argument('--beta', type=float, default=0.6)
    p.add_argument('--gamma', type=float, default=0.3)
    p.add_argument('--dt', type=float, default=1.0)
    p.add_argument('--avg_degree_value', type=float, default=2.0)
    p.add_argument('--population_scale', type=float, default=15.0)
    p.add_argument('--min_population', type=int, default=5000)

    p.add_argument('--gcn_hidden', type=int, default=64)
    p.add_argument('--tcn_hidden', type=int, default=64)
    p.add_argument('--graph_layers', type=int, default=2)
    p.add_argument('--tcn_layers', type=int, default=3)
    p.add_argument('--kernel_size', type=int, default=3)
    p.add_argument('--dropout', type=float, default=0.2)

    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=220)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--patience', type=int, default=40)
    p.add_argument('--grad_clip', type=float, default=5.0)
    p.add_argument('--lambda_y', type=float, default=0.20)
    p.add_argument('--lambda_smooth', type=float, default=1e-3)
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def main():
    args = parse_args()
    cfg_kwargs = {k: v for k, v in vars(args).items() if k in STESPConfig.__dataclass_fields__}
    cfg = STESPConfig(**cfg_kwargs)
    ensure_dir(cfg.output_dir)
    seed_everything(cfg.seed)

    if args.rebuild_dataset or not os.path.exists(os.path.join(cfg.processed_dir, 'meta.json')):
        build_processed_dataset(cfg)

    meta = load_meta(cfg.processed_dir)
    train_ds = STESPDataset(os.path.join(cfg.processed_dir, 'train.npz'))
    val_ds = STESPDataset(os.path.join(cfg.processed_dir, 'val.npz'))
    test_ds = STESPDataset(os.path.join(cfg.processed_dir, 'test.npz'))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_batch)

    device = torch.device('cuda' if cfg.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    population = torch.tensor(meta['population'], dtype=torch.float32, device=device)
    avg_degree = torch.tensor(meta['avg_degree'], dtype=torch.float32, device=device)

    model_cfg = STESPModelConfig(
        num_nodes=meta['num_regions'],
        feature_dim=meta['feature_dim'],
        gcn_hidden=cfg.gcn_hidden,
        tcn_hidden=cfg.tcn_hidden,
        graph_layers=cfg.graph_layers,
        tcn_layers=cfg.tcn_layers,
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout,
        beta=cfg.beta,
        gamma=cfg.gamma,
        dt=cfg.dt,
    )
    model = STESPModel(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_loss = float('inf')
    best_epoch = -1
    bad_epochs = 0
    train_losses, val_losses, val_accs = [], [], []
    ckpt_path = os.path.join(cfg.output_dir, 'best_stesp_routeA.pt')

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_metrics, *_ = run_epoch(model, train_loader, optimizer, device, population, avg_degree, cfg)
        va_loss, va_metrics, *_ = run_epoch(model, val_loader, None, device, population, avg_degree, cfg)
        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        val_accs.append(va_metrics['accuracy'])
        print(
            f"Epoch {epoch:03d} | train_loss={tr_loss:.6f} | val_loss={va_loss:.6f} | "
            f"val_mae={va_metrics['mae']:.4f} | val_rmse={va_metrics['rmse']:.4f} | "
            f"val_pcc={va_metrics['pcc']:.4f} | ratio_rmse={va_metrics['ratio_rmse']:.6f}"
        )
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save({
                'model_state': model.state_dict(),
                'model_config': model_cfg.asdict(),
                'data_config': cfg.asdict(),
                'meta': meta,
                'best_epoch': best_epoch,
                'best_val_loss': best_val_loss,
            }, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f'Early stopping at epoch {epoch}, best epoch {best_epoch}')
                break

    save_curves(cfg.output_dir, train_losses, val_losses, val_accs)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    tr_loss, tr_metrics, tr_pred_counts, tr_true_counts, tr_pred_ratio, tr_true_ratio, tr_pred_y, tr_true_y = run_epoch(model, train_loader, None, device, population, avg_degree, cfg)
    va_loss, va_metrics, va_pred_counts, va_true_counts, va_pred_ratio, va_true_ratio, va_pred_y, va_true_y = run_epoch(model, val_loader, None, device, population, avg_degree, cfg)
    te_loss, te_metrics, te_pred_counts, te_true_counts, te_pred_ratio, te_true_ratio, te_pred_y, te_true_y = run_epoch(model, test_loader, None, device, population, avg_degree, cfg)

    np.save(os.path.join(cfg.output_dir, 'train_pred_counts.npy'), tr_pred_counts)
    np.save(os.path.join(cfg.output_dir, 'val_pred_counts.npy'), va_pred_counts)
    np.save(os.path.join(cfg.output_dir, 'test_pred_counts.npy'), te_pred_counts)
    np.save(os.path.join(cfg.output_dir, 'test_true_counts.npy'), te_true_counts)
    np.save(os.path.join(cfg.output_dir, 'test_pred_i_ratio.npy'), te_pred_ratio)
    np.save(os.path.join(cfg.output_dir, 'test_true_i_ratio.npy'), te_true_ratio)
    np.save(os.path.join(cfg.output_dir, 'test_pred_y.npy'), te_pred_y)
    np.save(os.path.join(cfg.output_dir, 'test_true_y.npy'), te_true_y)

    summary = {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'train_metrics': tr_metrics,
        'val_metrics': va_metrics,
        'test_metrics': te_metrics,
        'model_config': model_cfg.asdict(),
        'data_config': cfg.asdict(),
        'selected_labels': meta['selected_labels'],
        'selected_indices': meta['selected_indices'],
    }
    with open(os.path.join(cfg.output_dir, 'training_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print('\n===== Final Results =====')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
