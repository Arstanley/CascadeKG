"""
Calibration Time Benchmark: LTT-Pareto vs Baseline
Demonstrates that calibration cost is dominated by model.decode, not grid search.
"""
import argparse
import time
import math
import os

import numpy as np

import torch
import wandb
from torch.utils.data import DataLoader
from torch_geometric.nn import GAE
from sentence_transformers import SentenceTransformer
from scipy.stats import binom

from data.CascadeDataset import CascadeDataset
from utils.data_utils import load_cascade_data_icews, get_auxiliary_data, collate_graphs
from models import RGCNEncoder, DistMultDecoder
from models.RGCNEncoder import TextConditionedRGCNEncoder
from trainers.train_loop import Trainer
from trainers.calibration import Calibrator
from utils.utils import *

# ── CLI ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Calibration time benchmark")
parser.add_argument('--dataset', type=str, default='icews14')
parser.add_argument('--error_rate', type=float, default=0.5)
parser.add_argument('--model_variant', type=str, default='text_conditioned')
parser.add_argument('--num_hops', type=int, default=2)
parser.add_argument('--hidden_dim', default=768)
parser.add_argument('--num_epochs', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.001)
args = parser.parse_args()

# ── Wandb disabled ────────────────────────────────────────────────────
wandb.init(mode="disabled")

# ── Device ────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# ── Data ──────────────────────────────────────────────────────────────
data_path = f'./data/{args.dataset}'
sentence_encoder = SentenceTransformer("all-MiniLM-L6-v2")

node_emb, entity2id, relation2id = get_auxiliary_data(data_path, args.dataset)
id2ent = {v: k for k, v in entity2id.items()}
id2rel = {v: k for k, v in relation2id.items()}

train_data = load_cascade_data_icews(data_path, "train", node_emb, entity2id, relation2id, max_hops=args.num_hops)
val_data   = load_cascade_data_icews(data_path, "valid", node_emb, entity2id, relation2id, max_hops=args.num_hops)
test_data  = load_cascade_data_icews(data_path, "test",  node_emb, entity2id, relation2id, max_hops=args.num_hops)

val_calibrate_data  = CascadeDataset(f"{data_path}/valid_gupdate_cascade.json", f"{data_path}/node_embeddings.pkl", entity2id, relation2id, num_hops=args.num_hops)
test_calibrate_data = CascadeDataset(f"{data_path}/test_gupdate_cascade.json",  f"{data_path}/node_embeddings.pkl", entity2id, relation2id, num_hops=args.num_hops)

num_relations = len(relation2id)
num_nodes = len(entity2id)
if num_relations > 0:
    num_relations = max(num_relations, max(relation2id.values()) + 1)
if num_nodes > 0:
    num_nodes = max(num_nodes, max(entity2id.values()) + 1)

# ── Model ─────────────────────────────────────────────────────────────
encoder = RGCNEncoder.TextConditionedRGCNEncoder(int(args.hidden_dim / 2), num_relations, 384)
model = GAE(encoder, DistMultDecoder.DistMultDecoder(num_relations, int(args.hidden_dim / 2)))
model = model.to(device)

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=collate_graphs)
val_loader   = DataLoader(val_data,   batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=collate_graphs)
test_loader  = DataLoader(test_data,  batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=collate_graphs)

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
best_model_path = f"best_model_{args.dataset}_{args.model_variant}_{args.num_hops}_cascade.pt"

trainer = Trainer(model, optimizer, args.batch_size, device, args.model_variant, sentence_encoder, best_model_path, args.num_epochs, test_loader)
model = trainer.train(train_loader, val_loader, test_loader)

# ── Helpers ───────────────────────────────────────────────────────────
def run_ltt_pareto(grid_step):
    """Run LTT-Pareto calibration + test evaluation, return (time, coverage, avg_set_size)."""
    cal = Calibrator(
        val_calibrate_data, model, device, sentence_encoder,
        args.model_variant, num_relations, id2ent, id2rel,
        grid_step=grid_step, error_rate=args.error_rate,
    )

    t0 = time.perf_counter()
    result = cal.calibrate_ltt_pareto()
    cal_time = time.perf_counter() - t0

    if result["status"] == "ok":
        best = result["best_config"]
        alpha1, alpha2 = best["alpha1"], best["alpha2"]
    else:
        best = result["best_empirical"]
        alpha1, alpha2 = best["alpha1"], best["alpha2"]

    n, mean_f1, fail_rate, avg_set_size, mean_prec, _ = cal._evaluate_alpha_combo_on_dataset(
        test_calibrate_data, device, alpha1, alpha2, use_percentage=True
    )
    coverage = 1.0 - fail_rate
    return cal_time, coverage, avg_set_size


def run_baseline():
    """Run baseline calibration + test evaluation, return (time, coverage, avg_set_size)."""
    cal = Calibrator(
        val_calibrate_data, model, device, sentence_encoder,
        args.model_variant, num_relations, id2ent, id2rel,
        error_rate=args.error_rate,
    )

    t0 = time.perf_counter()
    q1_hat, q2_hat = cal.baseline_calibration(val_calibrate_data)
    cal_time = time.perf_counter() - t0

    f1, precision, recall, _, fail_rate, set_size = cal.baseline_evaluation(test_calibrate_data, q1_hat, q2_hat)
    coverage = 1.0 - fail_rate
    return cal_time, coverage, set_size


# ── Run benchmarks ───────────────────────────────────────────────────
grid_configs = [
    ("Coarse", 0.20),
    ("Medium", 0.10),
    ("Fine",   0.05),
]

rows = []

# Baseline
print("\n>>> Running Baseline calibration …")
bt, bc, bs = run_baseline()
rows.append(("Baseline (no grid)", "-", bt, bc, bs))

for label, step in grid_configs:
    n_alpha = len(np.arange(0, 1.0 + 1e-9, step))
    combos = n_alpha * n_alpha
    print(f"\n>>> Running LTT-Pareto ({label}, step={step}, {combos} combos) …")
    lt, lc, ls = run_ltt_pareto(step)
    rows.append((f"LTT-Pareto ({label.lower()})", str(combos), lt, lc, ls))

# ── Print table ──────────────────────────────────────────────────────
print("\n")
print("=" * 72)
print(f"  Calibration Time Benchmark")
print(f"  Model: {args.model_variant} | Dataset: {args.dataset} | Error rate: {args.error_rate:.2f}")
print("=" * 72)
header = f"{'Method':<24}| {'Grid Combos':>11} | {'Cal. Time (s)':>13} | {'Coverage':>8} | {'Avg Set Size':>12}"
print(header)
print("-" * len(header))
for name, combos, t, cov, sz in rows:
    print(f"{name:<24}| {combos:>11} | {t:>13.2f} | {cov:>8.4f} | {sz:>12.2f}")
print("=" * 72)
