from models import RGCNEncoder, DistMultDecoder, ComplExDecoder
from models.RGCNEncoder import TextConditionedRGCNEncoder
from models.DistMultModel import DistMultOnlyModel
from models.TransEModel import TransEOnlyModel
from models.RGATEncoder import RGATEncoder
from torch_geometric.nn import GAE

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import gc
import wandb

from utils.data_utils import load_data_nba, load_data_icews
from utils.utils import *


# Sentence encoder
sentence_encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--num_features', default=128)
parser.add_argument('--hidden_dim', default=768)
parser.add_argument('--out_dim', default=256)
parser.add_argument('--lr', default=0.0001)
parser.add_argument('--num_epochs', default=100)
parser.add_argument('--dataset', default='ICEWS14')
parser.add_argument(
    '--model_variant',
    default='text_conditioned',
    choices=['text_conditioned', 'rgcn_baseline', 'distmult', 'transe', 'rgat'],
    help="Model variant to use."
)

parser.add_argument('--num_hops', type=int, default=3, help="Number of rounds for multi-hop inference.")
parser.add_argument('--discount', type=float, default=0.9, help="Discount factor γ for weighting further hops.")
args = parser.parse_args()

data_path = './data/icews14' if args.dataset == 'ICEWS14' else './data/NBATransaction'

run = wandb.init(
    project=f"GraphUpdate-{args.dataset}",
    name=f"{args.model_variant}_{args.dataset}_{args.num_hops}",
    config=vars(args)
)

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

# Load data
if args.dataset == 'NBATransaction':
    train_data, val_data, test_data, num_relations = load_data_nba(data_path)
elif args.dataset == 'ICEWS14':
    train_data, val_data, test_data, num_relations = load_data_icews(data_path)

num_nodes = train_data[0][1].num_nodes

# Build model
if args.model_variant == "text_conditioned":
    encoder = RGCNEncoder.TextConditionedRGCNEncoder(int(args.hidden_dim / 2), num_relations, 384)
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
else:
    raise ValueError("Unknown model variant")

model = model.to(device)

print(train_data[0])
# Data loaders
train_loader = DataLoader(train_data, batch_size=1, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=1, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True, pin_memory=True)

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)


def train_one_epoch():
    model.train()
    for i, batch in enumerate(tqdm(train_loader, total=len(train_data))):
        optimizer.zero_grad()
        trigger_event, subgraph_orig, subgraph_mod_init, subgraph_mod, paragraph = batch

        # Transfer to GPU
        subgraph_orig = subgraph_orig.to(device)
        subgraph_mod_init = subgraph_mod_init.to(device)
        subgraph_mod = subgraph_mod.to(device)

        if args.model_variant == 'text_conditioned':
            with torch.no_grad():
                z_text = sentence_encoder.encode(paragraph, convert_to_tensor=True).to(device)
                if z_text.dim() == 2 and z_text.size(0) == 1:
                    z_text = z_text.squeeze(0)

        if args.model_variant == 'rgat':
            with torch.no_grad():
                print(trigger_event)
                z_text = sentence_encoder.encode(" ".join([e[0] for e in trigger_event]), convert_to_tensor=True).to(device)
                if z_text.dim() == 2 and z_text.size(0) == 1:
                    z_text = z_text.squeeze(0)

        # Encode
        if args.model_variant == 'distmult':
            z = model.encode()
        elif args.model_variant == 'rgcn_baseline':
            z = model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type)
        elif args.model_variant == 'rgat':
            z = model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type)
        elif args.model_variant == 'transe':
            z = model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type)
        else:
            z = model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type, z_text)

        final_pos, final_neg = 0, 0
        z_cur = z

        for hop in range(args.num_hops):
            weight = args.discount ** hop
            pos_out = model.decode(z_cur, subgraph_mod.edge_index, subgraph_mod.edge_type)
            neg_edge_index = negative_sampling(subgraph_mod.edge_index, subgraph_mod.num_nodes)
            neg_out = model.decode(z_cur, neg_edge_index, subgraph_mod.edge_type)
            final_pos += weight * pos_out
            final_neg += weight * neg_out

        out = torch.cat([final_pos, final_neg])
        gt = torch.cat([torch.ones_like(final_pos), torch.zeros_like(final_neg)])
        loss = F.binary_cross_entropy_with_logits(out, gt)

        wandb.log({"loss": loss.item()})

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

        del subgraph_orig, subgraph_mod_init, subgraph_mod
        gc.collect()
        torch.cuda.empty_cache()


@torch.no_grad()
def test(data):
    model.eval()
    total_mrr = 0
    all_preds, all_labels = [], []

    for batch in data:
        trigger_event, subgraph_orig, subgraph_mod_init, subgraph_mod, paragraph = batch
        subgraph_orig = subgraph_orig.to(device)
        subgraph_mod_init = subgraph_mod_init.to(device)
        subgraph_mod = subgraph_mod.to(device)

        if args.model_variant == 'text_conditioned':
            with torch.no_grad():
                z_text = sentence_encoder.encode(paragraph, convert_to_tensor=True).to(device)
                if z_text.dim() == 2 and z_text.size(0) == 1:
                    z_text = z_text.squeeze(0)

        if args.model_variant == 'rgat':
            with torch.no_grad():
                print(trigger_event)
                z_text = sentence_encoder.encode(" ".join([e[0] for e in trigger_event]), convert_to_tensor=True).to(device)
                if z_text.dim() == 2 and z_text.size(0) == 1:
                    z_text = z_text.squeeze(0)

        # Encode
        if args.model_variant == 'distmult':
            z = model.encode()
        elif args.model_variant == 'rgcn_baseline':
            z = model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type)
        elif args.model_variant == 'rgat':
            z = model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type)
        elif args.model_variant == 'transe':
            z = model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type)
        elif args.model_variant == 'text_conditioned':
            z = model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type, z_text)

        # Compute MRR
        mrr = compute_mrr(z, subgraph_mod, subgraph_mod.edge_index, subgraph_mod.edge_type, model)
        total_mrr += mrr

        final_pos, final_neg = 0, 0
        z_cur = z

        for hop in range(args.num_hops):
            weight = args.discount ** hop
            pos_out = model.decode(z_cur, subgraph_mod.edge_index, subgraph_mod.edge_type)
            neg_edge_index = negative_sampling(subgraph_mod.edge_index, subgraph_mod.num_nodes)
            neg_out = model.decode(z_cur, neg_edge_index, subgraph_mod.edge_type)
            final_pos += weight * pos_out
            final_neg += weight * neg_out

        out = torch.cat([final_pos, final_neg])
        labels = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
        preds = (torch.sigmoid(out) > 0.5).long()

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    f1 = f1_score(all_labels.numpy(), all_preds.numpy())
    precision = precision_score(all_labels.numpy(), all_preds.numpy())
    recall = recall_score(all_labels.numpy(), all_preds.numpy())

    return total_mrr / len(data), f1, precision, recall


if __name__ == '__main__':
    best_val_mrr = 0
    best_model_path = f"best_model_{args.dataset}_{args.model_variant}_{args.num_hops}.pt"

    for epoch in range(args.num_epochs):
        print(f"Start Epoch: {epoch}")
        train_one_epoch()

        val_mrr, val_f1, val_precision, val_recall = test(val_loader)
        test_mrr, test_f1, test_precision, test_recall = test(test_loader)

        wandb.log({
            "val_mrr": val_mrr,
            "val_f1": val_f1,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "test_mrr": test_mrr,
            "test_f1": test_f1,
            "test_precision": test_precision,
            "test_recall": test_recall
        })

        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            final_test_mrr = test_mrr
            final_test_f1 = test_f1
            final_test_precision = test_precision
            final_test_recall = test_recall
            best_epoch = epoch

            torch.save(model.state_dict(), best_model_path)
            print(f"Update best model at epoch {epoch} with val_mrr {val_mrr:.4f}, "
                  f"test_mrr {test_mrr:.4f}, test_f1 {test_f1:.4f}, "
                  f"test_precision {test_precision:.4f}, test_recall {test_recall:.4f}")

        print(f"Epoch: {epoch:03d}, Val: {val_mrr:.4f}, Test: {test_mrr:.4f}")
        print(f"Best Epoch: {best_epoch}, Final Test MRR: {final_test_mrr:.4f}, "
              f"Final Test F1: {final_test_f1:.4f}")

    with open(f"final_results_{args.num_hops}.log", "a") as f:
        f.write(f"Dataset: {args.dataset}, Model: {args.model_variant}, Hops: {args.num_hops}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Final Test MRR: {final_test_mrr:.4f}, "
                f"F1: {final_test_f1:.4f}, "
                f"Precision: {final_test_precision:.4f}, "
                f"Recall: {final_test_recall:.4f}\n")
        f.write("=" * 60 + "\n")
