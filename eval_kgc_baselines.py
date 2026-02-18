"""
Evaluate SimKGC and KGT5 baselines on ICEWS14-Event test set.
Produces Table 1 style metrics: F1, Acc, Rec, MRR, AUROC.
"""
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from utils.data_utils import load_cascade_data_icews, get_auxiliary_data, collate_graphs
from trainers.train_loop import Trainer
from models.SimKGCModel import SimKGCModel
from models.KGT5Model import KGT5Model

DATASET = "icews14"
NUM_HOPS = 2
BATCH_SIZE = 4

data_path = f"./data/{DATASET}"

# GPU support
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

sentence_encoder = SentenceTransformer("all-MiniLM-L6-v2")

node_emb, entity2id, relation2id = get_auxiliary_data(data_path, DATASET)
num_relations = max(len(relation2id), max(relation2id.values()) + 1)
num_nodes = max(len(entity2id), max(entity2id.values()) + 1)

test_data = load_cascade_data_icews(data_path, "test", node_emb, entity2id, relation2id, max_hops=NUM_HOPS)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, collate_fn=collate_graphs)

HIDDEN_DIM = 768

variants = {
    "simkgc": SimKGCModel(num_nodes, num_relations, HIDDEN_DIM),
    "kgt5": KGT5Model(num_nodes, num_relations, HIDDEN_DIM),
}

results = {}

for name, model in variants.items():
    ckpt_path = f"best_model_{DATASET}_{name}_{NUM_HOPS}_cascade.pt"
    print(f"\nLoading {name} from {ckpt_path}...")
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model = model.to(device)

    trainer = Trainer(
        model, None, BATCH_SIZE, device, name,
        sentence_encoder, ckpt_path, 0, test_loader
    )

    mrr, f1, precision, recall, roc_auc, accuracy = trainer.test(test_loader)

    results[name] = {
        "F1": f1,
        "Acc": accuracy,
        "Rec": recall,
        "MRR": mrr,
        "AUROC": roc_auc,
    }
    print(f"  {name}: F1={f1:.3f}  Acc={accuracy:.3f}  Rec={recall:.3f}  MRR={mrr:.3f}  AUROC={roc_auc:.3f}")

# Print table
print("\n")
print("=" * 65)
print(f"  Table 1 Metrics | Dataset: ICEWS14-Event")
print("=" * 65)
hdr = f"{'Model':<12} | {'F1':>6} | {'Acc':>6} | {'Rec':>6} | {'MRR':>6} | {'AUROC':>6}"
print(hdr)
print("-" * len(hdr))

for name in ["simkgc", "kgt5"]:
    r = results[name]
    print(f"{name:<12} | {r['F1']:>6.3f} | {r['Acc']:>6.3f} | {r['Rec']:>6.3f} | {r['MRR']:>6.3f} | {r['AUROC']:>6.3f}")

print("=" * 65)
