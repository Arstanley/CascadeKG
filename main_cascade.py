import argparse
from torch.utils.data import DataLoader
from data.CascadeDataset import CascadeDataset
from utils.data_utils import load_cascade_data_icews, get_auxiliary_data, collate_graphs 
from sentence_transformers import SentenceTransformer
import wandb
import torch
from torch_geometric.data import Batch, Data
from trainers.train_loop import Trainer

from models import RGCNEncoder, DistMultDecoder
from models.RGCNEncoder import TextConditionedRGCNEncoder
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from models.DistMultModel import DistMultOnlyModel
from models.TransEModel import TransEOnlyModel
from models.RGATEncoder import RGATEncoder
from models.HittERModel import HittERModel
from models.CompGCNModel import CompGCNModel
from models.SimKGCModel import SimKGCModel
from models.KGT5Model import KGT5Model
from torch_geometric.nn import GAE
from itertools import product
from trainers.calibration import Calibrator

from utils.utils import *
import torch.nn.functional as F
import math

from scipy.stats import binom

from math import comb

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='icews14')
parser.add_argument('--num_epochs', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--num_hops', type=int, default=2)
parser.add_argument('--num_features', default=128)
parser.add_argument('--hidden_dim', default=768)
parser.add_argument('--out_dim', default=256)
parser.add_argument('--calibration_type', type=str, default='ltt_pareto')
parser.add_argument('--error_rate', type=float, default=0.5) # User specified error rate for uncertainty quantification

sentence_encoder = SentenceTransformer("all-MiniLM-L6-v2")

parser.add_argument(
    '--model_variant',
    default='text_conditioned',
    choices=['text_conditioned', 'text_gated', 'text_concat_mlp', 'text_film', 'rgcn_baseline', 'distmult', 'transe', 'rgat', 'hitter', 'compgcn', 'simkgc', 'kgt5'],
    help="Model variant to use."
)

args = parser.parse_args()

run = wandb.init(
    project=f"GraphUpdate-{args.dataset}-Cascade-Calibration-{args.calibration_type}",
    name=f"{args.model_variant}_{args.dataset}_{args.num_hops}_{args.error_rate}",
    config=vars(args)
)

data_path = f'./data/{args.dataset}'
def normalize_features(x):
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True) + 1e-6
    return (x - mean) / std

# GPU support
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

node_emb, entity2id, relation2id = get_auxiliary_data(data_path, args.dataset)
id2ent = {v: k for k, v in entity2id.items()}
id2rel = {v: k for k, v in relation2id.items()}

# Assume entity2id, relation2id, node_emb are preloaded mappings
train_data = load_cascade_data_icews(data_path, "train", node_emb, entity2id, relation2id, max_hops=args.num_hops)

val_data   = load_cascade_data_icews(data_path, "valid", node_emb, entity2id, relation2id, max_hops=args.num_hops)
val_calibrate_data = CascadeDataset(f"{data_path}/valid_gupdate_cascade.json", f"{data_path}/node_embeddings.pkl", entity2id, relation2id, num_hops=args.num_hops)

test_data  = load_cascade_data_icews(data_path, "test", node_emb, entity2id, relation2id, max_hops=args.num_hops)
test_calibrate_data = CascadeDataset(f"{data_path}/test_gupdate_cascade.json", f"{data_path}/node_embeddings.pkl", entity2id, relation2id, num_hops=args.num_hops)

num_relations = len(relation2id)
num_nodes = len(entity2id)
if len(relation2id) > 0:
    num_relations = max(num_relations, max(relation2id.values()) + 1)
if len(entity2id) > 0:
    num_nodes = max(num_nodes, max(entity2id.values()) + 1)

print(f"Num nodes: {num_nodes}, Num relations: {num_relations}")

# Build model
if args.model_variant == "text_conditioned":
    encoder = RGCNEncoder.TextConditionedRGCNEncoder(int(args.hidden_dim / 2), num_relations, 384)
    model = GAE(encoder, DistMultDecoder.DistMultDecoder(num_relations, int(args.hidden_dim / 2)))
elif args.model_variant == "text_gated":
    encoder = RGCNEncoder.TextConditionedRGCNEncoderGated(int(args.hidden_dim / 2), num_relations, 384)
    model = GAE(encoder, DistMultDecoder.DistMultDecoder(num_relations, int(args.hidden_dim / 2)))
elif args.model_variant == "text_concat_mlp":
    encoder = RGCNEncoder.TextConditionedRGCNEncoderConcat(int(args.hidden_dim / 2), num_relations, 384)
    model = GAE(encoder, DistMultDecoder.DistMultDecoder(num_relations, int(args.hidden_dim / 2)))
elif args.model_variant == "text_film":
    encoder = RGCNEncoder.TextConditionedRGCNEncoderFiLM(int(args.hidden_dim / 2), num_relations, 384)
    model = GAE(encoder, DistMultDecoder.DistMultDecoder(num_relations, int(args.hidden_dim / 2)))
elif args.model_variant == "rgcn_baseline":
    encoder = RGCNEncoder.RGCNEncoder(int(args.hidden_dim / 2), num_relations)
    model = GAE(encoder, DistMultDecoder.DistMultDecoder(num_relations, int(args.hidden_dim / 2)))
elif args.model_variant == "distmult":
    model = DistMultOnlyModel(num_nodes, num_relations, args.hidden_dim)
elif args.model_variant == "transe":
    model = TransEOnlyModel(num_nodes, num_relations, args.hidden_dim)
elif args.model_variant == "rgat":
    encoder = RGATEncoder(int(args.hidden_dim / 2), args.hidden_dim, num_relations)
    model = GAE(encoder, DistMultDecoder.DistMultDecoder(num_relations, int(args.hidden_dim)))
elif args.model_variant == "hitter":
    model = HittERModel(
        num_entities=num_nodes,
        num_relations=num_relations,
        hidden_dim=int(args.hidden_dim / 2),
        num_heads=8,
        num_bottom_layers=2,
        num_top_layers=2,
        dropout=0.1,
        max_neighbors=50
    )
elif args.model_variant == "compgcn":
    model = CompGCNModel(
        num_nodes=num_nodes,
        num_relations=num_relations,
        in_channels=int(args.hidden_dim / 2),
        hidden_channels=int(args.hidden_dim / 2),
        op='mult'
    )
elif args.model_variant == "simkgc":
    model = SimKGCModel(num_nodes, num_relations, args.hidden_dim)
elif args.model_variant == "kgt5":
    model = KGT5Model(num_nodes, num_relations, args.hidden_dim)
else:
    raise ValueError("Unknown model variant")

model = model.to(device)
# Data loaders
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=collate_graphs)
val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=collate_graphs)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=collate_graphs)

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

if __name__ == "__main__":
    best_model_path = f"best_model_{args.dataset}_{args.model_variant}_{args.num_hops}_cascade.pt"
    trainer = Trainer(model, optimizer, args.batch_size, device, args.model_variant, sentence_encoder, best_model_path, args.num_epochs, test_loader)
    model = trainer.train(train_loader, val_loader, test_loader)

    calibrator = Calibrator(val_calibrate_data, model, device, sentence_encoder, args.model_variant, num_relations, id2ent, id2rel, error_rate=args.error_rate)

    if args.calibration_type == 'ltt_pareto':
        calibrate_result = calibrator.calibrate_ltt_pareto()
        if calibrate_result["status"] == "ok":
            best_config = calibrate_result["best_config"]
            alpha1 = best_config["alpha1"]
            alpha2 = best_config["alpha2"]
        else:
            print("⚠️ no valid calibration")
            alpha1,  alpha2 = calibrate_result["best_empirical"]["alpha1"], calibrate_result["best_empirical"]["alpha2"]

        n_test, mean_f1_test, fail_rate_hat_test, avg_set_size_test, mean_precision_test, _ = calibrator._evaluate_alpha_combo_on_dataset(
            test_calibrate_data, device,
            alpha1, alpha2, use_percentage=True
        )

        k_test = int(math.floor(n_test * fail_rate_hat_test))
        p_test = float(binom.cdf(k_test, n_test, args.error_rate))

        print(f"\n✅ FINAL TEST RESULTS")
        print(f"----------------------------------------------")
        print(f"α₁ = {alpha1:.2f}, α₂ = {alpha2:.2f}")
        print(f"Mean F1:          {mean_f1_test:.4f}")
        print(f"Mean Precision:   {mean_precision_test:.4f}")
        print(f"Failure rate:     {fail_rate_hat_test:.4f} ({k_test}/{n_test})")
        print(f"Avg. set size:    {avg_set_size_test}")
        print(f"Binomial p-value: {p_test:.5f}")
        print(f"----------------------------------------------")

        wandb.log({
            "final_alpha1": alpha1,
            "final_alpha2": alpha2,
            "final_test_f1": mean_f1_test,
            "final_test_precision": mean_precision_test,
            "final_test_fail_rate": fail_rate_hat_test,
            "final_test_set_size": avg_set_size_test,
            "final_test_pval": p_test
        })
    elif args.calibration_type == 'baseline':
        q1_hat, q2_hat = calibrator.baseline_calibration(val_calibrate_data)
        f1, precision, recall, roc_auc, fail, set_size = calibrator.baseline_evaluation(test_calibrate_data, q1_hat, q2_hat)

        print(f"\n✅ FINAL TEST RESULTS -- baseline")
        print(f"----------------------------------------------")
        print(f"q1_hat = {q1_hat}, q2_hat = {q2_hat}")
        print(f"Mean F1:          {f1:.4f}")
        print(f"Mean Precision:   {precision:.4f}")
        print(f"Failure rate:     {fail:.4f}") 
        print(f"Avg. set size:    {set_size:.2f}")
        print(f"----------------------------------------------")

        wandb.log({
            "final_q1_hat": q1_hat,
            "final_q2_hat": q2_hat,
            "final_test_f1": f1,
            "final_test_precision": precision,
            "final_test_fail_rate": fail,
            "final_test_set_size": set_size,
        })
    elif args.calibration_type == 'agg':
        q_hat, recall = calibrator._calibrate_aggregate(val_calibrate_data)
        print('calibration recall:', recall)
        f1_macro, precision_macro, recall_macro, avg_size, coverage = calibrator._evaluate_aggregate(test_calibrate_data, q_hat)

        print(f"\n✅ FINAL TEST RESULTS -- aggregate")
        print(f"----------------------------------------------")
        print(f"q_hat = {q_hat:.2f}")
        print(f"Mean F1:          {f1_macro:.4f}")
        print(f"Mean Precision:   {precision_macro:.4f}")
        print(f"Mean Recall:      {recall_macro:.4f}")
        print(f"Avg. set size:    {avg_size:.2f}")
        print(f"Coverage:         {coverage:.2f}")
        print(f"Fail rate:        {1 - recall_macro:.2f}")
        print(f"----------------------------------------------")
        
        wandb.log({
            "final_q_hat": q_hat,
            "final_test_f1": f1_macro,
            "final_test_precision": precision_macro,
            "final_test_recall": recall_macro,
            "final_test_set_size": avg_size,
            "final_test_coverage": coverage,
            "final_test_fail_rate": 1 - recall_macro,
        })
