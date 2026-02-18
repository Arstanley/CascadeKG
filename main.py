from models import RGCNEncoder, DistMultDecoder, ComplExDecoder
from models.RGCNEncoder import TextConditionedRGCNEncoder
from models.DistMultModel import DistMultOnlyModel
from models.TransEModel import TransEOnlyModel
from models.RGATEncoder import RGATEncoder
from torch_geometric.nn import GAE
import argparse
import torch
from sklearn.metrics import roc_auc_score
from utils.data_utils import load_data_nba, load_data_icews 
from utils.utils import * 
import torch.nn.functional as F
from tqdm import tqdm
import gc
import wandb
from torch_geometric.loader import DataLoader 
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, precision_score, recall_score
from torch_geometric.data import Data

sentence_encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Parse Argument
parser = argparse.ArgumentParser()
parser.add_argument('--num_features', default=128)
parser.add_argument('--hidden_dim', default=768)
parser.add_argument('--out_dim', default=256)
parser.add_argument('--lr', default=0.0001)
parser.add_argument('--num_epochs', default=100)
parser.add_argument('--data_path', default='./data/icews14')
parser.add_argument('--dataset', default='ICEWS14')
parser.add_argument('--model_variant', default='text_conditioned',
                    choices=['text_conditioned', 'rgcn_baseline', 'distmult', 'transe', 'rgat'],
                    help="Model variant to use.")
parser.add_argument('--per_hop_topk', type=int, default=100,
                    help='How many predicted edges to add per hop (after thresholding).')
parser.add_argument('--score_threshold', type=float, default=0.5,
                    help='Sigmoid score threshold for accepting a predicted edge.')
parser.add_argument('--num_hops', type=int, default=3,
                    help="Number of rounds for multi-hop inference.")
parser.add_argument('--discount', type=float, default=0.7,
                    help="Discount factor γ for weighting further hops.")

args = parser.parse_args()

run = wandb.init(
    project=f"GraphUpdate-{args.dataset}",
    name=f"{args.model_variant}_{args.dataset}_{args.num_hops}",
    config=vars(args)
)

def normalize_features(x):
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True) + 1e-6
    return (x - mean) / std

def _metrics_from_counts(tp, fp, fn):
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return precision, recall, f1

# GPU Support 
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# Load data 
if args.dataset == 'NBATransaction':
    train_data, val_data, test_data, num_relations = load_data_nba(args.data_path) # TODO 
if args.dataset == 'ICEWS14':
    train_data, val_data, test_data, num_relations = load_data_icews(args.data_path) # TODO 

num_nodes = train_data[0][1].num_nodes

if args.model_variant == "text_conditioned":
    encoder = RGCNEncoder.TextConditionedRGCNEncoder(int(args.hidden_dim / 2), num_relations, 384)
    model = GAE(encoder, DistMultDecoder.DistMultDecoder(num_relations, int(args.hidden_dim/2)))
elif args.model_variant == "rgcn_baseline":
    encoder = RGCNEncoder.RGCNEncoder(int(args.hidden_dim / 2), num_relations)
    model = GAE(encoder, DistMultDecoder.DistMultDecoder(num_relations, int(args.hidden_dim/2)))
elif args.model_variant == "distmult":
    num_nodes = train_data[0][1].num_nodes  # subgraph_orig
    model = DistMultOnlyModel(num_nodes, num_relations, args.hidden_dim)
elif args.model_variant == "transe":
    num_nodes = train_data[0][1].num_nodes
    model = TransEOnlyModel(num_nodes, num_relations, args.hidden_dim)
elif args.model_variant == "rgat":
    encoder = RGATEncoder(int(args.hidden_dim / 2), args.hidden_dim, num_relations)
    model = GAE(encoder, DistMultDecoder.DistMultDecoder(num_relations, int(args.hidden_dim)))
else:
    raise ValueError("Unknown model variant")

model.load_state_dict(torch.load("./best_model_ICEWS14_transe_1.pt", weights_only=True))
model = model.to(device)

# Define Model and optimizer
# model = GAE(
#     # We need to divide it by 2 here because the embeddings need to be concatenated
#     RGCNEncoder.TextConditionedRGCNEncoder(int(args.hidden_dim/2), num_relations, 384),      
#     # DistMultDecoder.DistMultDecoder(num_relations, args.hidden_dim)
#     DistMultDecoder.DistMultDecoder(num_relations, args.hidden_dim)
# ).to(device)

train_loader = DataLoader(train_data, batch_size=1, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=1, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True, pin_memory=True)

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

# -------------------- Helper Functions for Multi-Hop Inferences -------------------- #
def _edge_tuples(edge_index, edge_type):
    # returns list of (u,v,r)
    ei0 = edge_index[0].tolist()
    ei1 = edge_index[1].tolist()
    et = edge_type.tolist()
    return list(zip(ei0, ei1, et))

def _dedup_append(ei, et, new_ei, new_et):
    """Append (new_ei,new_et) into (ei,et) with de-dup on (u,v,r)."""
    if new_ei.numel() == 0:
        return ei, et
    existing = set(_edge_tuples(ei, et))
    to_add = []
    for (u, v, r) in _edge_tuples(new_ei, new_et):
        if (u, v, r) not in existing:
            to_add.append((u, v, r))
            existing.add((u, v, r))
    if not to_add:
        return ei, et
    add_ei = torch.tensor([[u for (u,_,_) in to_add],
                           [v for (_,v,_) in to_add]], device=ei.device, dtype=ei.dtype)
    add_et = torch.tensor([r for (_,_,r) in to_add], device=et.device, dtype=et.dtype)
    ei = torch.cat([ei, add_ei], dim=1)
    et = torch.cat([et, add_et], dim=0)
    return ei, et

def _unique_nodes_from_edges(ei):
    return torch.unique(ei.flatten())

def _encode_with_variant(model, variant, graph_data, z_text=None):
    # graph_data: Data(x, edge_index, edge_type, num_nodes)
    if variant == 'distmult':
        return model.encode()
    elif variant in ('rgcn_baseline', 'rgat', 'transe'):
        return model.encode(graph_data.x, graph_data.edge_index, graph_data.edge_type)
    else:  # text_conditioned
        return model.encode(graph_data.x, graph_data.edge_index, graph_data.edge_type, z_text)

def _sample_relation_types(num_edges, num_relations, device):
    return torch.randint(low=0, high=num_relations, size=(num_edges,), device=device, dtype=torch.long)

def _sample_frontier_candidates(cur_ei, num_nodes, frontier_nodes, num_samples=1000, device='cpu'):
    """
    Sample candidate non-edges biased to touch the frontier.
    Returns edge_index [2, M].
    """
    frontier = frontier_nodes.tolist()
    if len(frontier) == 0:
        # fallback: uniform pairs
        src = torch.randint(0, num_nodes, (num_samples,), device=device)
        dst = torch.randint(0, num_nodes, (num_samples,), device=device)
        return torch.stack([src, dst], dim=0)

    # half from frontier->random, half random->frontier
    half = num_samples // 2
    src1 = torch.tensor(frontier, device=device)[torch.randint(0, len(frontier), (half,), device=device)]
    dst1 = torch.randint(0, num_nodes, (half,), device=device)
    dst2 = torch.tensor(frontier, device=device)[torch.randint(0, len(frontier), (num_samples - half,), device=device)]
    src2 = torch.randint(0, num_nodes, (num_samples - half,), device=device)
    cand = torch.cat([torch.stack([src1, dst1], dim=0),
                      torch.stack([src2, dst2], dim=0)], dim=1)

    # remove any existing edges
    existing = set(_edge_tuples(cur_ei, torch.zeros(cur_ei.size(1), device=device, dtype=torch.long)))  # ignore rel
    keep = []
    for i in range(cand.size(1)):
        u = int(cand[0, i]); v = int(cand[1, i])
        if (u, v, 0) not in existing:  # relation ignored for existence check
            keep.append(i)
    if not keep:
        return cand[:, :0]  # empty
    keep = torch.tensor(keep, device=device, dtype=torch.long)
    return cand[:, keep]
# -----------------------------------------------------------------------------------------

def train_one_epoch(): 
    model.train()
    for i, batch in enumerate(tqdm(train_loader, total=len(train_data))):  
        # -------------------------- TEST -------------------------- #
        if i >= 2:
            continue
        # ------------------------------------------------------------ #
        optimizer.zero_grad()
        trigger_event, subgraph_orig, subgraph_mod_init, subgraph_mod, paragraph = batch

        # # Normalize node features
        # subgraph_orig.x = normalize_features(subgraph_orig.x)
        # subgraph_mod_init.x = normalize_features(subgraph_mod_init.x)
        # subgraph_mod.x = normalize_features(subgraph_mod.x)

        # Transfer to GPU
        subgraph_orig = subgraph_orig.to(device)
        subgraph_mod_init = subgraph_mod_init.to(device)
        subgraph_mod = subgraph_mod.to(device)

        if args.model_variant == 'text_conditioned':
            with torch.no_grad():
                z_text = sentence_encoder.encode(paragraph, convert_to_tensor=True).to(device)
            # z_text = sentence_encoder.encode(paragraph, convert_to_tensor=True).to(device)  # [384] 
            if z_text.dim() == 2 and z_text.size(0) == 1:
                z_text = z_text.squeeze(0)  # make it [384] instead of [1, 384]
        if args.model_variant == 'rgat':
            with torch.no_grad():
                z_text = sentence_encoder.encode(" ".join([e[0] for e in trigger_event]), convert_to_tensor=True).to(device)
            if z_text.dim() == 2 and z_text.size(0) == 1:
                z_text = z_text.squeeze(0)  # make it [384] instead of [1, 384]

        if args.model_variant == 'distmult':
            z = model.encode()
        elif args.model_variant == 'rgcn_baseline':
            z = model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type)
        elif args.model_variant == 'rgat':
            z = model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type, )
        elif args.model_variant == 'transe':
            z = model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type)
        else:
            z = model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type, z_text)
            # z_modified = model.encode(subgraph_mod_init.x, subgraph_mod_init.edge_index, subgraph_mod_init.edge_type, z_text)
            # z = torch.concat([z_original, z_modified], dim=1)

        # === Build working graph that will be uppdated each hop === 
        work_ei = subgraph_mod_init.edge_index.clone()
        work_et = subgraph_mod_init.edge_type.clone()
        frontier_nodes = _unique_nodes_from_edges(work_ei)

        # For supervised loss, we will use the final ground-truth modified edges:
        gt_ei = subgraph_mod.edge_index
        gt_et = subgraph_mod.edge_type

        hop_losses = []

        for hop in range(args.num_hops):
            weight = args.discount ** hop

            # Re-encode on the CURRENT working graph
            work_graph = Data(
                x=subgraph_orig.x,
                edge_index=work_ei,
                edge_type=work_et,
                num_nodes=subgraph_orig.num_nodes
            ).to(device)

            z = _encode_with_variant(model, args.model_variant, work_graph,
                                     z_text if args.model_variant == 'text_conditioned' else None)

            # ---- Supervised edge loss for this hop ----
            # Score the remaining GT positives (those not yet in work graph)
            cur_set = set(_edge_tuples(work_ei, work_et))
            gt_set = set(_edge_tuples(gt_ei, gt_et))
            remaining = list(gt_set - cur_set)

            if len(remaining) > 0:
                pos_ei = torch.tensor([[u for (u,_,_) in remaining],
                                       [v for (_,v,_) in remaining]],
                                      device=device, dtype=work_ei.dtype)
                pos_et = torch.tensor([r for (_,_,r) in remaining],
                                      device=device, dtype=work_et.dtype)
                pos_scores = model.decode(z, pos_ei, pos_et)
            else:
                pos_scores = torch.tensor([], device=device)

            # Sample same-number negatives (non-edges) with random relation types
            num_pos = max(1, pos_scores.numel())
            neg_ei = negative_sampling(work_ei, subgraph_orig.num_nodes,
                                       num_neg_samples=num_pos).to(device)
            neg_et = _sample_relation_types(neg_ei.size(1), num_relations, device)
            neg_scores = model.decode(z, neg_ei, neg_et)

            # BCE loss for this hop
            if pos_scores.numel() > 0:
                labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)
                logits = torch.cat([pos_scores, neg_scores], dim=0)
            else:
                # no remaining GT positives; only negatives (tiny penalty)
                labels = torch.zeros_like(neg_scores)
                logits = neg_scores

            hop_losses.append(weight * F.binary_cross_entropy_with_logits(logits, labels))

            # ---- PREDICT new edges to ADD for next hop (heuristic) ----
            # Build candidate non-edges focusing on frontier
            cand_ei = _sample_frontier_candidates(work_ei, subgraph_orig.num_nodes,
                                                  frontier_nodes, num_samples=max(1000, 10*args.per_hop_topk),
                                                  device=device)
            if cand_ei.numel() > 0:
                cand_et = _sample_relation_types(cand_ei.size(1), num_relations, device)
                cand_scores = torch.sigmoid(torch.cat([pos_scores, neg_scores]))

                # Threshold first, then take top-k
                keep_mask = cand_scores >= args.score_threshold
                if keep_mask.any():
                    cand_ei = cand_ei[:, keep_mask]
                    cand_et = cand_et[keep_mask]
                    cand_scores = cand_scores[keep_mask]

                k = min(args.per_hop_topk, cand_ei.size(1))
                if k > 0:
                    _, idx = torch.topk(cand_scores, k)
                    new_ei = cand_ei[:, idx]
                    new_et = cand_et[idx]
                    work_ei, work_et = _dedup_append(work_ei, work_et, new_ei, new_et)
                    frontier_nodes = _unique_nodes_from_edges(new_ei)
                else:
                    # nothing qualifies this hop
                    pass
            else:
                # no candidates found; stop early
                break

        # Sum losses over hops (single backward per batch)
        loss = torch.stack(hop_losses).sum() if len(hop_losses) > 0 else torch.tensor(0., device=device)

        # print(final_pos)
        # print(final_pos.shape, final_neg.shape)
        # out = torch.cat([final_pos, final_neg])
        # gt = torch.cat([torch.ones_like(final_pos), torch.zeros_like(final_neg)])
        # loss = F.binary_cross_entropy_with_logits(out, gt)

        wandb.log({"loss": loss.item()})

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

        del subgraph_orig, subgraph_mod_init, subgraph_mod
        gc.collect()
        # del z_original, z_modified, z, pos_out, neg_out, out, gt
        # del cross_entropy_loss, reg_loss, loss
        torch.cuda.empty_cache()

@torch.no_grad()
def test_per_hop(data):
    model.eval()

    max_hops = args.num_hops
    # Accumulators across batches
    agg_tp = agg_fp = agg_fn = 0
    perhop_tp = [0] * max_hops
    perhop_fp = [0] * max_hops
    perhop_fn = [0] * max_hops

    # MRR
    mrr_agg_sum = 0.0
    perhop_mrr_sum = [0.0] * max_hops

    num_batches = 0

    for i, batch in enumerate(data):
        # -------------------------- TEST -------------------------- #
        # if i >= 2:
        #     continue
        # -------------------------------
        num_batches += 1
        trigger_event, subgraph_orig, subgraph_mod_init, subgraph_mod, paragraph = batch 
        subgraph_orig = subgraph_orig.to(device)
        subgraph_mod_init = subgraph_mod_init.to(device)
        subgraph_mod = subgraph_mod.to(device)

        # text feature (if needed)
        z_text = None
        if args.model_variant == 'text_conditioned':
            z_text = sentence_encoder.encode(paragraph, convert_to_tensor=True).to(device)
            if z_text.dim() == 2 and z_text.size(0) == 1:
                z_text = z_text.squeeze(0)
        elif args.model_variant == 'rgat':
            z_text = sentence_encoder.encode(" ".join([e[0] for e in trigger_event]), convert_to_tensor=True).to(device)
            if z_text.dim() == 2 and z_text.size(0) == 1:
                z_text = z_text.squeeze(0)

        # Ground truth "added" set (beyond init)
        base_init_set = set(_edge_tuples(subgraph_mod_init.edge_index, subgraph_mod_init.edge_type))
        gt_set_full   = set(_edge_tuples(subgraph_mod.edge_index,  subgraph_mod.edge_type))
        gt_added_set  = gt_set_full - base_init_set

        # Working graph & frontier
        work_ei = subgraph_mod_init.edge_index.clone()
        work_et = subgraph_mod_init.edge_type.clone()
        frontier_nodes = _unique_nodes_from_edges(work_ei)

        # For aggregate predictions across all hops:
        cum_pred_set = set()

        for hop in range(max_hops):
            # Encode on current working graph
            work_graph = Data(
                x=subgraph_orig.x,
                edge_index=work_ei,
                edge_type=work_et,
                num_nodes=subgraph_orig.num_nodes
            ).to(device)

            z = _encode_with_variant(model, args.model_variant, work_graph, z_text)

            # Remaining GT positives at this hop
            cur_set = set(_edge_tuples(work_ei, work_et))
            gt_set = set(_edge_tuples(subgraph_mod.edge_index, subgraph_mod.edge_type))
            remaining = list(gt_set - cur_set)

            # ---- Supervised edge loss for this hop ----
            # Score the remaining GT positives (those not yet in work graph)
            cur_set = set(_edge_tuples(work_ei, work_et))
            gt_set = set(_edge_tuples(subgraph_mod.edge_index, subgraph_mod.edge_type))
            remaining = list(gt_set - cur_set)

            if len(remaining) > 0:
                pos_ei = torch.tensor([[u for (u,_,_) in remaining],
                                       [v for (_,v,_) in remaining]],
                                      device=device, dtype=work_ei.dtype)
                pos_et = torch.tensor([r for (_,_,r) in remaining],
                                      device=device, dtype=work_et.dtype)
                pos_scores = model.decode(z, pos_ei, pos_et)
            else:
                pos_scores = torch.tensor([], device=device)

            # Sample same-number negatives (non-edges) with random relation types
            num_pos = max(1, pos_scores.numel())
            neg_ei = negative_sampling(work_ei, subgraph_orig.num_nodes,
                                       num_neg_samples=num_pos).to(device)
            neg_et = _sample_relation_types(neg_ei.size(1), num_relations, device)
            neg_scores = model.decode(z, neg_ei, neg_et) 


            # Compute MRR **at this hop** (on target subgraph)
            mrr_h = compute_mrr(z, subgraph_mod, subgraph_mod.edge_index, subgraph_mod.edge_type, model)
            perhop_mrr_sum[hop] += mrr_h

            # Concatenate as the candidate pool
            cand_ei = torch.cat([pos_ei, neg_ei], dim=1)
            cand_et = torch.cat([pos_et, neg_et], dim=0)
            cand_scores = torch.sigmoid(torch.cat([pos_scores, neg_scores], dim=0))
            print(cand_scores.shape)

            # Build labels aligned with cand_* for per-hop classification metrics
            cand_labels = torch.cat([
                torch.ones_like(pos_scores, dtype=torch.long),
                torch.zeros_like(neg_scores, dtype=torch.long)
            ], dim=0)            # Threshold then top-k

            keep_mask = cand_scores >= args.score_threshold
            # print(keep_mask)
            if keep_mask.any():
                # print(cand_ei)
                cand_ei_keep = cand_ei[:, keep_mask]
                cand_et_keep = cand_et[keep_mask]
                cand_scores_keep = cand_scores[keep_mask]
                cand_labels_keep = cand_labels[keep_mask]
            else:
                cand_ei_keep = cand_ei[:, :0]
                cand_et_keep = cand_et[:0]
                cand_scores_keep = cand_scores[:0]
                cand_labels_keep = cand_labels[:0]

            k = min(args.per_hop_topk, cand_ei_keep.size(1))
            if k > 0:
                _, idx = torch.topk(cand_scores_keep, k)
                new_ei = cand_ei_keep[:, idx]
                new_et = cand_et_keep[idx]
                new_labels = cand_labels_keep[idx]  # 1 for GT pos, 0 for sampled neg

                # ---- PER-HOP METRICS (JUST THIS HOP’S ADDITIONS) ----
                # You can either compute set-metrics as before:
                new_set_hop = set(_edge_tuples(new_ei, new_et))
                tp_h = len(new_set_hop & gt_added_set)
                fp_h = len(new_set_hop - gt_added_set)
                fn_h = len(gt_added_set) - tp_h  # recall@hop vs full GT (your choice)

                perhop_tp[hop] += tp_h
                perhop_fp[hop] += fp_h
                perhop_fn[hop] += fn_h

                # Update working graph and aggregate
                work_ei, work_et = _dedup_append(work_ei, work_et, new_ei, new_et)
                # frontier_nodes = _unique_nodes_from_edges(new_ei)
                cum_pred_set |= new_set_hop
            else:
                # nothing qualified this hop
                pass

        # Final encode for aggregate MRR (optional; you can keep last hop’s z)
        work_graph = Data(
            x=subgraph_orig.x,
            edge_index=work_ei,
            edge_type=work_et,
            num_nodes=subgraph_orig.num_nodes
        ).to(device)
        z_final = _encode_with_variant(model, args.model_variant, work_graph, z_text)
        mrr_final = compute_mrr(z_final, subgraph_mod, subgraph_mod.edge_index, subgraph_mod.edge_type, model)
        mrr_agg_sum += mrr_final

        # ---- Aggregate set metrics (everything we added over all hops) ----
        tp = len(cum_pred_set & gt_added_set)
        fp = len(cum_pred_set - gt_added_set)
        fn = len(gt_added_set - cum_pred_set)
        agg_tp += tp
        agg_fp += fp
        agg_fn += fn

    # === Reduce across batches ===
    # Aggregate metrics
    agg_precision, agg_recall, agg_f1 = _metrics_from_counts(agg_tp, agg_fp, agg_fn)
    mrr_agg = mrr_agg_sum / max(1, num_batches)

    # Per-hop metrics
    perhop_metrics = []
    for h in range(max_hops):
        ph_prec, ph_rec, ph_f1 = _metrics_from_counts(perhop_tp[h], perhop_fp[h], perhop_fn[h])
        ph_mrr = perhop_mrr_sum[h] / max(1, num_batches)
        perhop_metrics.append({
            "precision": ph_prec, "recall": ph_rec, "f1": ph_f1, "mrr": ph_mrr
        })

    results = {
        "aggregate": {"precision": agg_precision, "recall": agg_recall, "f1": agg_f1, "mrr": mrr_agg},
        "per_hop": perhop_metrics
    }

    # For backward-compat, return the legacy 4-tuple as well:
    return mrr_agg, agg_f1, agg_precision, agg_recall, results


if __name__ == '__main__':
    best_val_mrr = 0 
    best_model_path = f"best_model_{args.dataset}_{args.model_variant}_{args.num_hops}.pt"

    for epoch in range(args.num_epochs):
        print(f"Start Epoch: {epoch}")
        # train_one_epoch()

        # Validation
        val_mrr, val_f1, val_prec, val_rec, val_res = test_per_hop(val_loader)
        # Test
        test_mrr, test_f1, test_prec, test_rec, test_res = test_per_hop(test_loader)

        # Log aggregate
        log_dict = {
            "val_mrr": val_mrr,
            "val_f1": val_f1,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "test_mrr": test_mrr,
            "test_f1": test_f1,
            "test_precision": test_prec,
            "test_recall": test_rec,
        }

        # Log per-hop metrics (hop indices are 1-based in names)
        for i, ph in enumerate(val_res["per_hop"], start=1):
            log_dict[f"val_precision_hop{i}"] = ph["precision"]
            log_dict[f"val_recall_hop{i}"] = ph["recall"]
            log_dict[f"val_f1_hop{i}"] = ph["f1"]
            log_dict[f"val_mrr_hop{i}"] = ph["mrr"]
        log_dict["val_precision_agg"] = val_res["aggregate"]["precision"]
        log_dict["val_recall_agg"]    = val_res["aggregate"]["recall"]
        log_dict["val_f1_agg"]        = val_res["aggregate"]["f1"]
        log_dict["val_mrr_agg"]       = val_res["aggregate"]["mrr"]

        for i, ph in enumerate(test_res["per_hop"], start=1):
            log_dict[f"test_precision_hop{i}"] = ph["precision"]
            log_dict[f"test_recall_hop{i}"] = ph["recall"]
            log_dict[f"test_f1_hop{i}"] = ph["f1"]
            log_dict[f"test_mrr_hop{i}"] = ph["mrr"]
        log_dict["test_precision_agg"] = test_res["aggregate"]["precision"]
        log_dict["test_recall_agg"]    = test_res["aggregate"]["recall"]
        log_dict["test_f1_agg"]        = test_res["aggregate"]["f1"]
        log_dict["test_mrr_agg"]       = test_res["aggregate"]["mrr"]

        wandb.log(log_dict)

        # Model selection still on aggregate val MRR (or change if you prefer)
        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            final_test_mrr = test_mrr
            final_test_f1 = test_f1
            final_test_precision = test_prec
            final_test_recall = test_rec
            best_epoch = epoch

            torch.save(model.state_dict(), best_model_path)
            print(f"Update best model at epoch {epoch} with val_mrr {val_mrr:.4f}, test_mrr {test_mrr:.4f}")

        print(f'Epoch: {epoch:03d}, Val(agg): {val_mrr:.4f}, Test(agg): {test_mrr:.4f}')


    print(f'Best Epoch: {best_epoch}, Final Test MRR: {final_test_mrr:.4f}, Final Test F1: {final_test_f1:.4f}')

    with open(f"final_results_{args.num_hops}.log", "a") as f:
        f.write(f"Dataset: {args.dataset}, Model: {args.model_variant}, Hops: {args.num_hops}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Final Test MRR: {final_test_mrr:.4f}, F1: {final_test_f1:.4f}, Precision: {final_test_precision:.4f}, Recall: {final_test_recall:.4f}\n")
        f.write("=" * 60 + "\n")