from models import RGCNEncoder, DistMultDecoder, EventRGAT
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

sentence_encoder = SentenceTransformer("all-mpnet-base-v2")

# Parse Argument
parser = argparse.ArgumentParser()
parser.add_argument('--num_features', default=128)
parser.add_argument('--hidden_dim', default=768)
parser.add_argument('--out_dim', default=256)
parser.add_argument('--lr', default=0.00001)
parser.add_argument('--num_epochs', default=100)
parser.add_argument('--data_path', default='./data/icews14')
parser.add_argument('--dataset', default='ICEWS14')

args = parser.parse_args()

run = wandb.init(project=f"GraphUpdate-{args.dataset}")

def normalize_features(x):
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True) + 1e-6
    return (x - mean) / std

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

# Define Model and optimizer
model = GAE(
    EventRGAT.EventConditionedEncoder(args.hidden_dim, [args.hidden_dim, args.hidden_dim], num_relations, 768),      
    DistMultDecoder.DistMultDecoder(num_relations, args.hidden_dim*2)
).to(device)

train_loader = DataLoader(train_data, batch_size=1, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=1, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True, pin_memory=True)

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

def train_one_epoch():
    model.train()

    for i, batch in enumerate(tqdm(train_loader, total=len(train_data))):
        # if i >= 5:
        #     break
        optimizer.zero_grad()
        trigger_event, subgraph_orig, subgraph_mod_init, subgraph_mod, paragraph = batch

        trigger_event = [cur[0] for cur in trigger_event] 
        trigger_event_str = " ".join(trigger_event)
        # Transfer to GPU
        subgraph_orig = subgraph_orig.to(device)
        subgraph_mod_init = subgraph_mod_init.to(device)
        
        z_text = sentence_encoder.encode(trigger_event_str, convert_to_tensor=True).to(device)  # [384]
        if z_text.dim() == 2 and z_text.size(0) == 1:
            z_text = z_text.squeeze(0)  # make it [384] instead of [1, 384]

        z_original = model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type, z_text) 
        z_modified = model.encode(subgraph_mod_init.x, subgraph_mod_init.edge_index, subgraph_mod_init.edge_type, z_text) # the graph with the initial addition/deletion of links 
        z = torch.concat([z_original, z_modified], dim=1) 
        # z = model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type, z_text) 

        pos_out = model.decode(z, subgraph_mod.edge_index, subgraph_mod.edge_type)

        neg_edge_index = negative_sampling(subgraph_mod.edge_index, subgraph_mod.num_nodes)
        neg_out = model.decode(z, neg_edge_index, subgraph_mod.edge_type) 

        # neg_out = torch.zeros(700, device='cuda')
        out = torch.cat([pos_out, neg_out])

        gt = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
        cross_entropy_loss = F.binary_cross_entropy_with_logits(out, gt)
        reg_loss = z.pow(2).mean() + model.decoder.rel_emb.pow(2).mean()
        loss = cross_entropy_loss + 1e-2 * reg_loss

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
def test(data):
    model.eval()
    total_mrr = 0

    for batch in data:
        trigger_event, subgraph_orig, subgraph_mod_init, subgraph_mod, paragraph = batch 

        trigger_event = [cur[0] for cur in trigger_event] 
        trigger_event_str = " ".join(trigger_event)

        z_text = sentence_encoder.encode(trigger_event_str, convert_to_tensor=True).to(device)  # [384]

        if z_text.dim() == 2 and z_text.size(0) == 1:
            z_text = z_text.squeeze(0)  # make it [384] instead of [1, 384]
            
        subgraph_orig.to(device)
        subgraph_mod_init.to(device)
        subgraph_mod.to(device)

        z_original = model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type, z_text) 
        z_modified = model.encode(subgraph_mod_init.x, subgraph_mod_init.edge_index, subgraph_mod_init.edge_type, z_text) # the graph with the initial addition/deletion of links 
        z = torch.concat([z_original, z_modified], dim=1) 

        # z = model.encode(subgraph_orig.x, subgraph_orig.edge_index, subgraph_orig.edge_type, z_text) 

        mrr = compute_mrr(z, subgraph_mod, subgraph_mod.edge_index, subgraph_mod.edge_type, model)
        total_mrr += mrr

    return total_mrr / len(data)

if __name__ == '__main__':
    best_val_auc = 0 
    for epoch in range(args.num_epochs):
        print(f"Start Epoch: {epoch}")
        train_one_epoch()
        val_mrr = test(val_loader)
        test_mrr = test(test_loader)

        wandb.log({"val_mrr": val_mrr, "test_mrr": test_mrr})
        
        if val_mrr > best_val_auc:
            best_val_auc = val_mrr
            final_test_auc = test_mrr
        print(f'Epoch: {epoch:03d}, Val: {val_mrr:.4f}, '
            f'Test: {test_mrr:.4f}')

    print(f'Final Test: {final_test_auc:.4f}')