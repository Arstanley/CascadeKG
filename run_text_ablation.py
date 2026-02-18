"""
Ablation: Text Injection Mechanisms for TextConditionedRGCNEncoder
Compares 4 fusion strategies on icews14 with a fixed error rate.

Variants:
  text_conditioned  — linear projection + addition (current default)
  text_gated        — learned gate controls how much text shifts nodes
  text_concat_mlp   — concat(node, text) → MLP → fused node repr
  text_film         — FiLM: γ·x + β predicted from text (scale+shift)
"""
import subprocess
import sys
import os
import re
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────
VARIANTS = [
    "text_conditioned",   # baseline: add
    "text_gated",         # gated add
    "text_concat_mlp",    # concat + MLP
    "text_film",          # FiLM
]
ERROR_RATE      = 0.1
DATASET         = "icews14"
CALIBRATION     = "ltt_pareto"
NUM_HOPS        = 2
NUM_EPOCHS      = 2
LOG_DIR         = Path("ablation_logs")

LOG_DIR.mkdir(exist_ok=True)


def run_variant(variant):
    log_path = LOG_DIR / f"{variant}_er{ERROR_RATE}.log"
    cmd = [
        sys.executable, "main_cascade.py",
        "--model_variant",    variant,
        "--dataset",          DATASET,
        "--error_rate",       str(ERROR_RATE),
        "--calibration_type", CALIBRATION,
        "--num_hops",         str(NUM_HOPS),
        "--num_epochs",       str(NUM_EPOCHS),
    ]
    print(f"\n{'='*60}")
    print(f"  Running: {variant}  (log → {log_path})")
    print(f"{'='*60}")
    with open(log_path, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        print(f"  [WARNING] {variant} exited with code {result.returncode}")
    return log_path


def parse_log(log_path):
    """Extract final test metrics from a main_cascade.py log."""
    text = log_path.read_text()

    def find(pattern, cast=float, default=None):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else default

    return {
        "f1":        find(r"Mean F1:\s+([\d.]+)"),
        "precision": find(r"Mean Precision:\s+([\d.]+)"),
        "fail_rate": find(r"Failure rate:\s+([\d.]+)"),
        "set_size":  find(r"Avg\. set size:\s+([\d.]+)"),
        "pval":      find(r"Binomial p-value:\s+([\d.]+)"),
        "alpha1":    find(r"α₁\s*=\s*([\d.]+)"),
        "alpha2":    find(r"α₂\s*=\s*([\d.]+)"),
    }


# ── Run all variants ──────────────────────────────────────────────────
logs = {}
for v in VARIANTS:
    logs[v] = run_variant(v)

# ── Parse & print table ───────────────────────────────────────────────
print("\n\n")
print("=" * 80)
print(f"  Text Injection Ablation | Dataset: {DATASET} | Error rate: {ERROR_RATE}")
print("=" * 80)
hdr = (f"{'Variant':<22} | {'α₁':>5} | {'α₂':>5} | "
       f"{'F1':>7} | {'Precision':>9} | {'Fail Rate':>9} | "
       f"{'Avg Set Size':>12} | {'p-value':>8}")
print(hdr)
print("-" * len(hdr))

for v in VARIANTS:
    m = parse_log(logs[v])
    a1  = f"{m['alpha1']:.2f}"  if m['alpha1']   is not None else "N/A"
    a2  = f"{m['alpha2']:.2f}"  if m['alpha2']   is not None else "N/A"
    f1  = f"{m['f1']:.4f}"      if m['f1']        is not None else "N/A"
    pre = f"{m['precision']:.4f}" if m['precision'] is not None else "N/A"
    fr  = f"{m['fail_rate']:.4f}" if m['fail_rate'] is not None else "N/A"
    sz  = f"{m['set_size']:.1f}"  if m['set_size']  is not None else "N/A"
    pv  = f"{m['pval']:.5f}"    if m['pval']      is not None else "N/A"
    print(f"{v:<22} | {a1:>5} | {a2:>5} | {f1:>7} | {pre:>9} | {fr:>9} | {sz:>12} | {pv:>8}")

print("=" * 80)
print(f"\nFull logs saved in: {LOG_DIR.resolve()}/")
