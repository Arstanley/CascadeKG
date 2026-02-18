# run_variants.py
import subprocess

# Define all variants
model_variants = [
    # "rgcn_baseline",
    "transe",
    "hitter",
    "compgcn",
    "distmult",
    "rgat",
    "text_conditioned",  # <-- excluded
]

calibration_types = [
    # "ltt_pareto",
    # "baseline",
    "agg"
]

error_rates = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
]

if __name__ == "__main__":
    for variant in model_variants:
        for calibration_type in calibration_types:
            for error_rate in error_rates:
                print(f"\n=== Running model variant: {variant} with calibration type: {calibration_type} and error rate: {error_rate} ===\n")
                subprocess.run(
                    ["python3", "main_cascade.py", "--model_variant", variant, "--num_hops", "2", "--dataset", "icews14", "--error_rate", str(error_rate), "--calibration_type", calibration_type],
                    check=True
                )