# CASCADEKG

This repository contains the code for the paper **"Risk-Controlled Event-Driven Cascading Updates for Knowledge Graph Consistency Restoration"**.

## Core Components

### Models (`models/`)
- `RGCNEncoder.py` - Text-conditioned R-GCN variants (additive, gated, FiLM, concat+MLP)
- `DistMultDecoder.py`, `ComplExDecoder.py` - KG scoring decoders
- Baseline models: DistMult, TransE, CompGCN, R-GAT, HittER, SimKGC, KGT5

### Training & Evaluation
- `main_cascade.py` - Main training script for CASCADEKG
- `run_text_ablation.py` - Text fusion ablation study (Table in rebuttal)
- `run_kgc_baselines.py` - KGC baseline adaptation experiments
- `eval_kgc_baselines.py` - KGC baseline evaluation
- `benchmark_calibration_time.py` - LTT calibration timing analysis

### Core Logic
- `trainers/calibration.py` - LTT calibration implementation
- `trainers/cascade_prediction.py` - Cascade prediction pipeline
- `trainers/train_loop.py` - Training loop
- `utils/data_utils.py` - Data loading and preprocessing
- `data/CascadeDataset.py` - PyTorch Dataset implementation

## Usage

### Training CASCADEKG
```bash
python main_cascade.py --dataset icews14 --alpha1 0.6 --alpha2 0.3
```

### Text Fusion Ablation
```bash
python run_text_ablation.py --variant [text_conditioned|text_gated|text_film|text_concat_mlp]
```

### KGC Baseline Evaluation
```bash
python run_kgc_baselines.py --model [simkgc|kgt5]
```

### Calibration Time Benchmark
```bash
python benchmark_calibration_time.py --granularity [coarse|medium|fine]
```

## Requirements

```
torch>=1.10
transformers>=4.20
numpy
tqdm
```

## Datasets

The code expects processed ICEWS14-Event and YAGO-Event datasets in `./data/`. Data processing scripts are included in `data/icews14/` and `data/YAGO3-10/`.

