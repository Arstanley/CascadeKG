
import argparse
import torch
from torch.utils.data import DataLoader
from utils.data_utils import load_cascade_data_icews, get_auxiliary_data, collate_graphs
from sentence_transformers import SentenceTransformer
from torch_geometric.nn import GAE
from models import RGCNEncoder, DistMultDecoder
from models.RGCNEncoder import TextConditionedRGCNEncoder
from models.DistMultModel import DistMultOnlyModel
import os
import torch.nn.functional as F
import json

# Disable wandb
os.environ["WANDB_MODE"] = "disabled"

def get_model(model_variant, num_nodes, num_relations, hidden_dim, device):
    if model_variant == "text_conditioned":
        encoder = RGCNEncoder.TextConditionedRGCNEncoder(int(hidden_dim / 2), num_relations, 384)
        model = GAE(encoder, DistMultDecoder.DistMultDecoder(num_relations, int(hidden_dim / 2)))
    elif model_variant == "distmult":
        model = DistMultOnlyModel(num_nodes, num_relations, hidden_dim)
    else:
        raise ValueError(f"Unknown model variant: {model_variant}")
    
    return model.to(device)

@torch.no_grad()
def compute_rank(ranks):
    # fair ranking prediction as the average
    # of optimistic and pessimistic ranking
    true = ranks[0]
    optimistic = (ranks > true).sum() + 1
    pessimistic = (ranks >= true).sum()
    return (optimistic + pessimistic).float() * 0.5

@torch.no_grad()
def get_detailed_ranks(z, data, edge_index, edge_type, model, id2ent, id2rel):
    """
    Returns a list of details for each edge in edge_index:
    {
        'head_rank': float,
        'tail_rank': float,
        'triplet': (s_name, r_name, o_name)
    }
    """
    results = []
    
    # Iterate over each target edge
    for i in range(edge_type.numel()):
        (src, dst), rel = edge_index[:, i], edge_type[i]
        
        src_id = src.item()
        dst_id = dst.item()
        rel_id = rel.item()
        
        # Get string names if available
        s_name = id2ent.get(src_id, str(src_id))
        o_name = id2ent.get(dst_id, str(dst_id))
        r_name = id2rel.get(rel_id, str(rel_id))
        
        # --- Tail Prediction (s, r, ?) ---
        # Mask true tails
        tail_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        # We need to mask out ALL true triplets in the graph (or at least known ones)
        # Using data.edge_index for masking
        for (heads, tails), types in [(data.edge_index, data.edge_type)]:
            tail_mask[tails[(heads == src) & (types == rel)]] = False
            
        # Add the ground truth back
        # tail_mask[dst] = True # Wait, compute_mrr logic usually constructs eval_edge_index with [dst, candidates...]
        
        # Candidate tails: [TrueTail, Candidate1, Candidate2...]
        # Actually standard eval often evaluates against ALL nodes.
        # Let's stick to the logic in utils.py:
        
        # 1. Identify valid candidates (all nodes except known positives)
        # But we must include the ground truth 'dst' for ranking.
        
        # Logic from utils.py:
        # tail_mask[tails[(heads == src) & (types == rel)]] = False
        # tail = torch.arange(data.num_nodes)[tail_mask]
        # tail = torch.cat([torch.tensor([dst]), tail])
        
        tail_mask[tails[(heads == src) & (types == rel)]] = False
        candidates = torch.arange(data.num_nodes, device=src.device)[tail_mask]
        # Prepend ground truth
        tail_candidates = torch.cat([torch.tensor([dst], device=src.device), candidates])
        
        head_repeated = torch.full_like(tail_candidates, fill_value=src)
        eval_edge_index = torch.stack([head_repeated, tail_candidates], dim=0)
        eval_edge_type = torch.full_like(tail_candidates, fill_value=rel)
        
        out = model.decode(z, eval_edge_index, eval_edge_type)
        tail_rank = compute_rank(out).item()
        
        # --- Head Prediction (?, r, o) ---
        head_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        for (heads, tails), types in [(data.edge_index, data.edge_type)]:
            head_mask[heads[(tails == dst) & (types == rel)]] = False
            
        candidates = torch.arange(data.num_nodes, device=src.device)[head_mask]
        head_candidates = torch.cat([torch.tensor([src], device=src.device), candidates])
        
        tail_repeated = torch.full_like(head_candidates, fill_value=dst)
        eval_edge_index = torch.stack([head_candidates, tail_repeated], dim=0)
        eval_edge_type = torch.full_like(head_candidates, fill_value=rel)
        
        out = model.decode(z, eval_edge_index, eval_edge_type)
        head_rank = compute_rank(out).item()
        
        results.append({
            'triplet': (s_name, r_name, o_name),
            'tail_rank': tail_rank,
            'head_rank': head_rank
        })
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='icews14')
    parser.add_argument('--num_hops', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--threshold_good', type=float, default=10.0, help="Max rank to consider 'good'")
    parser.add_argument('--threshold_bad', type=float, default=20.0, help="Min rank to consider 'bad'")
    
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    print(f"Using device: {device}")
    
    # 1. Load Data
    data_path = f'./data/{args.dataset}'
    node_emb, entity2id, relation2id = get_auxiliary_data(data_path, args.dataset)
    test_data = load_cascade_data_icews(data_path, "test", node_emb, entity2id, relation2id, max_hops=args.num_hops)
    
    # Helper maps
    id2ent = {v: k for k, v in entity2id.items()}
    id2rel = {v: k for k, v in relation2id.items()}
    
    num_relations = len(relation2id)
    num_nodes = len(entity2id)
    if len(relation2id) > 0:
        num_relations = max(num_relations, max(relation2id.values()) + 1)
    if len(entity2id) > 0:
        num_nodes = max(num_nodes, max(entity2id.values()) + 1)
        
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=True, collate_fn=collate_graphs)
    
    sentence_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # 2. Load Models
    print("Loading TextGCN (text_conditioned)...")
    model_tgcn = get_model("text_conditioned", num_nodes, num_relations, args.hidden_dim, device)
    ckpt_tgcn = f"best_model_{args.dataset}_text_conditioned_{args.num_hops}_cascade.pt"
    model_tgcn.load_state_dict(torch.load(ckpt_tgcn, map_location=device, weights_only=True))
    model_tgcn.eval()
    
    print("Loading DistMult...")
    model_dm = get_model("distmult", num_nodes, num_relations, args.hidden_dim, device)
    ckpt_dm = f"best_model_{args.dataset}_distmult_{args.num_hops}_cascade.pt"
    # Fallback to known path if needed, similar to other script
    if not os.path.exists(ckpt_dm):
        # Try finding *any* distmult
        print(f"Warning: {ckpt_dm} not found. Trying best_model_icews14_distmult_2.pt")
        ckpt_dm = "best_model_icews14_distmult_2.pt"
        
    if os.path.exists(ckpt_dm):
        model_dm.load_state_dict(torch.load(ckpt_dm, map_location=device, weights_only=True))
        model_dm.eval()
    else:
        print("Could not find DistMult checkpoint. Exiting.")
        return

    print("\nStarting search for case studies...")
    print(f"Criteria: TextGCN Rank <= {args.threshold_good} AND DistMult Rank >= {args.threshold_bad}")
    print("="*60)
    
    found_count = 0
    found_cases = []
    
    for batch_idx, batch in enumerate(test_loader):
        # Unpack batch (batch_size=1)
        trigger_event, subgraph_orig, subgraph_mod_init, subgraph_mod, paragraph = batch
        
        # paragraph is a list (batch_size=1), join it
        paragraph_text = "; ".join(paragraph)
        
        # Move to device
        subgraph_orig = subgraph_orig.to(device)
        subgraph_mod = subgraph_mod.to(device)
        
        # --- Inference TextGCN ---
        with torch.no_grad():
            z_text = sentence_encoder.encode(paragraph_text, convert_to_tensor=True).to(device)
            if z_text.dim() == 2 and z_text.size(0) == 1:
                z_text = z_text.squeeze(0)
            
            z_tgcn = model_tgcn.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type, z_text)
            
            # --- Inference DistMult ---
            z_dm = model_dm.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type)
            
            # --- Get Ranks ---
            # We evaluate on the target edges in subgraph_mod
            ranks_tgcn = get_detailed_ranks(z_tgcn, subgraph_mod, subgraph_mod.edge_index, subgraph_mod.edge_type, model_tgcn, id2ent, id2rel)
            ranks_dm = get_detailed_ranks(z_dm, subgraph_mod, subgraph_mod.edge_index, subgraph_mod.edge_type, model_dm, id2ent, id2rel)
            
            # Compare
            for i in range(len(ranks_tgcn)):
                r_tgcn = ranks_tgcn[i]
                r_dm = ranks_dm[i]
                
                # Debug print for first few
                if batch_idx < 5:
                     print(f"DEBUG: TGCN Tail={r_tgcn['tail_rank']:.1f}, Head={r_tgcn['head_rank']:.1f} | DM Tail={r_dm['tail_rank']:.1f}, Head={r_dm['head_rank']:.1f}")

                # Check Tail Rank difference
                if r_tgcn['tail_rank'] <= args.threshold_good and r_dm['tail_rank'] >= args.threshold_bad:
                    found_count += 1
                    s, r, o = r_tgcn['triplet']
                    
                    print(f"\n[Case #{found_count}] Found Example (Tail Prediction):")
                    print(f"Paragraph: {paragraph_text}")
                    print(f"Trigger Event: {trigger_event[0] if trigger_event else 'N/A'}")
                    print(f"Triplet: ({s}, {r}, {o})")
                    print(f"TextGCN Rank: {r_tgcn['tail_rank']:.1f}")
                    print(f"DistMult Rank: {r_dm['tail_rank']:.1f}")
                    print("-" * 40)
                    
                    found_cases.append({
                        'case_id': found_count,
                        'type': 'Tail Prediction',
                        'paragraph': paragraph_text,
                        'trigger_event': trigger_event[0] if trigger_event else None,
                        'triplet': [s, r, o],
                        'ranks': {
                            'textgcn_tail_rank': float(r_tgcn['tail_rank']),
                            'distmult_tail_rank': float(r_dm['tail_rank']),
                            'textgcn_head_rank': float(r_tgcn['head_rank']),
                            'distmult_head_rank': float(r_dm['head_rank'])
                        }
                    })
                    
                # Check Head Rank difference
                elif r_tgcn['head_rank'] <= args.threshold_good and r_dm['head_rank'] >= args.threshold_bad:
                    found_count += 1
                    s, r, o = r_tgcn['triplet']
                    
                    print(f"\n[Case #{found_count}] Found Example (Head Prediction):")
                    print(f"Paragraph: {paragraph_text}")
                    print(f"Trigger Event: {trigger_event[0] if trigger_event else 'N/A'}")
                    print(f"Triplet: ({s}, {r}, {o})")
                    print(f"TextGCN Rank: {r_tgcn['head_rank']:.1f}")
                    print(f"DistMult Rank: {r_dm['head_rank']:.1f}")
                    print("-" * 40)
                    
                    found_cases.append({
                        'case_id': found_count,
                        'type': 'Head Prediction',
                        'paragraph': paragraph_text,
                        'trigger_event': trigger_event[0] if trigger_event else None,
                        'triplet': [s, r, o],
                        'ranks': {
                            'textgcn_tail_rank': float(r_tgcn['tail_rank']),
                            'distmult_tail_rank': float(r_dm['tail_rank']),
                            'textgcn_head_rank': float(r_tgcn['head_rank']),
                            'distmult_head_rank': float(r_dm['head_rank'])
                        }
                    })

        if found_count >= 10:
            print("\nFound 10 examples. Stopping.")
            break
            
    # Save to file
    output_file = f"case_studies_{args.dataset}.json"
    with open(output_file, 'w') as f:
        json.dump(found_cases, f, indent=2)
    print(f"\nSaved {len(found_cases)} cases to {output_file}")

if __name__ == "__main__":
    main()


