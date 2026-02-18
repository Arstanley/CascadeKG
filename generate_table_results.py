
import argparse
import torch
from torch.utils.data import DataLoader
from data.CascadeDataset import CascadeDataset
from utils.data_utils import load_cascade_data_icews, get_auxiliary_data, collate_graphs
from sentence_transformers import SentenceTransformer
import wandb
from torch_geometric.nn import GAE
from trainers.train_loop import Trainer
from models import RGCNEncoder, DistMultDecoder
from models.RGCNEncoder import TextConditionedRGCNEncoder
from models.DistMultModel import DistMultOnlyModel
from models.TransEModel import TransEOnlyModel
from models.RGATEncoder import RGATEncoder
from models.HittERModel import HittERModel
from models.CompGCNModel import CompGCNModel
import os

# Disable wandb
os.environ["WANDB_MODE"] = "disabled"

def get_model(model_variant, num_nodes, num_relations, hidden_dim, device):
    if model_variant == "text_conditioned":
        encoder = RGCNEncoder.TextConditionedRGCNEncoder(int(hidden_dim / 2), num_relations, 384)
        model = GAE(encoder, DistMultDecoder.DistMultDecoder(num_relations, int(hidden_dim / 2)))
    elif model_variant == "rgcn_baseline":
        encoder = RGCNEncoder.RGCNEncoder(int(hidden_dim / 2), num_relations)
        model = GAE(encoder, DistMultDecoder.DistMultDecoder(num_relations, int(hidden_dim / 2)))
    elif model_variant == "distmult":
        model = DistMultOnlyModel(num_nodes, num_relations, hidden_dim)
    elif model_variant == "transe":
        model = TransEOnlyModel(num_nodes, num_relations, hidden_dim)
    elif model_variant == "rgat":
        encoder = RGATEncoder(int(hidden_dim / 2), hidden_dim, num_relations)
        model = GAE(encoder, DistMultDecoder.DistMultDecoder(num_relations, int(hidden_dim)))
    elif model_variant == "hitter":
        model = HittERModel(
            num_entities=num_nodes,
            num_relations=num_relations,
            hidden_dim=int(hidden_dim / 2),
            num_heads=8,
            num_bottom_layers=2,
            num_top_layers=2,
            dropout=0.1,
            max_neighbors=50
        )
    elif model_variant == "compgcn":
        model = CompGCNModel(
            num_nodes=num_nodes,
            num_relations=num_relations,
            in_channels=int(hidden_dim / 2),
            hidden_channels=int(hidden_dim / 2),
            op='mult'
        )
    else:
        raise ValueError(f"Unknown model variant: {model_variant}")
    
    return model.to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='icews14')
    parser.add_argument('--num_hops', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8) # Increased batch size for faster inference
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    data_path = f'./data/{args.dataset}'
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # Load data
    node_emb, entity2id, relation2id = get_auxiliary_data(data_path, args.dataset)
    test_data = load_cascade_data_icews(data_path, "test", node_emb, entity2id, relation2id, max_hops=args.num_hops)
    
    num_relations = len(relation2id)
    num_nodes = len(entity2id)
    if len(relation2id) > 0:
        num_relations = max(num_relations, max(relation2id.values()) + 1)
    if len(entity2id) > 0:
        num_nodes = max(num_nodes, max(entity2id.values()) + 1)

    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, collate_fn=collate_graphs)
    
    sentence_encoder = SentenceTransformer("all-MiniLM-L6-v2")

    models_to_evaluate = [
        ("DistMult", "distmult"),
        ("R-GAT", "rgat"),
        ("R-GCN", "rgcn_baseline"),
        ("TransE", "transe"),
        ("HittER", "hitter"),
        ("CompGCN", "compgcn"),
        ("TextGCN", "text_conditioned")
    ]

    print("\\begin{table}[t]")
    print("\\centering")
    print("\\small")
    print("\\caption{Direct performance comparison of Knowledge Graph and Graph Neural Network models across classification and ranking metrics.}")
    print("\\label{tab:model-performance}")
    print("\\begin{tabular}{l|cccc}")
    print("\\toprule")
    print("\\textbf{Model} & \\textbf{F1} & \\textbf{Acc.} & \\textbf{Rec.} & \\textbf{MRR} \\\\")
    print("\\midrule")

    for display_name, model_variant in models_to_evaluate:
        try:
            model = get_model(model_variant, num_nodes, num_relations, args.hidden_dim, device)
            
            # Construct checkpoint path
            checkpoint_path = f"best_model_{args.dataset}_{model_variant}_{args.num_hops}_cascade.pt"
            
            if not os.path.exists(checkpoint_path):
                # Try fallback names if specific one doesn't exist?
                # For now just warn
                print(f"% Warning: Checkpoint {checkpoint_path} not found for {display_name}")
                continue
            
            # Initialize Trainer just to use its test method
            # We don't need optimizer for testing
            trainer = Trainer(
                model=model,
                optimizer=None, 
                batch_size=args.batch_size,
                device=device,
                model_variant=model_variant,
                sentence_encoder=sentence_encoder,
                save_path=checkpoint_path,
                num_epochs=0,
                test_data_loader=test_loader
            )
            
            # Load model weights
            trainer.model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
            
            # Run evaluation
            mrr, f1, precision, recall, roc_auc, accuracy = trainer.test(test_loader)
            
            # Print row
            print(f"{display_name:<8} & {f1:.3f} & {accuracy:.3f} & {recall:.3f} & {mrr:.3f} & {roc_auc:.3f} & {precision:.3f} \\\\")
            
        except Exception as e:
            print(f"% Error evaluating {display_name}: {e}")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

if __name__ == "__main__":
    main()

