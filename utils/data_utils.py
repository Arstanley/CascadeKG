import json
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
import pandas as pd
from torch_geometric.utils import subgraph
import pickle as pkl
import os
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch

# def collate_graphs(batch):
#     # batch is a list of tuples: (trigger_event, subgraph_before, subgraph_mid, subgraph_after, paragraph)
#     trigger_events = [b[0] for b in batch]
#     subgraphs_orig = [b[1] for b in batch]
#     subgraphs_mod_init = [b[2] for b in batch]
#     subgraphs_mod = [b[3] for b in batch]
#     paragraphs = [b[4] for b in batch]

#     # PyG merge
#     subgraphs_orig = Batch.from_data_list(subgraphs_orig)
#     subgraphs_mod_init = Batch.from_data_list(subgraphs_mod_init)
#     subgraphs_mod = Batch.from_data_list(subgraphs_mod)

#     return trigger_events, subgraphs_orig, subgraphs_mod_init, subgraphs_mod, paragraphs


def add_edges(graph: Data, new_edge_index: torch.Tensor, new_edge_type: torch.Tensor) -> Data:
    """
    Returns a new graph where new edges are added to the input graph.

    Args:
        graph (Data): A PyG Data object with attributes:
                      - x (node features)
                      - edge_index (2 x E tensor)
                      - edge_type (E tensor of relation types)
        new_edge_index (torch.Tensor): Tensor of shape (2, E_new)
        new_edge_type (torch.Tensor): Tensor of shape (E_new,)

    Returns:
        Data: A new PyG Data object with updated edge_index and edge_type.
    """
    # Sanity checks
    if new_edge_index.numel() == 0:
        # No new edges; just return a shallow clone
        return graph.clone()

    # Ensure on same device and dtype
    new_edge_index = new_edge_index.to(graph.edge_index.device)
    new_edge_type = new_edge_type.to(graph.edge_type.device)

    # ----- Merge edges -----
    merged_edge_index = torch.cat([graph.edge_index, new_edge_index], dim=1)
    merged_edge_type = torch.cat([graph.edge_type, new_edge_type], dim=0)

    # # ----- (Optional) remove duplicates -----
    # # Convert edges to unique tuples (u,v,rel)
    # # Sort lexicographically to ensure determinism
    # combined = torch.cat([
    #     merged_edge_index.t(),
    #     merged_edge_type.view(-1, 1)
    # ], dim=1)
    # combined_unique = torch.unique(combined, dim=0)
    # merged_edge_index = combined_unique[:, :2].t().contiguous()
    # merged_edge_type = combined_unique[:, 2].contiguous()

    # ----- Return new graph -----
    new_graph = Data(
        x=graph.x.clone(),
        edge_index=merged_edge_index,
        edge_type=merged_edge_type
    )

    # Preserve any additional fields you might have (e.g., node IDs)
    for key in graph.keys():
        if key not in ['x', 'edge_index', 'edge_type']:
            setattr(new_graph, key, getattr(graph, key))
    return new_graph


def make_local_graph(edge_index, node_ids, x, edge_type):
    edge_index_local, _, _ = subgraph(
        subset=node_ids,
        edge_index=edge_index,
        relabel_nodes=True,
        num_nodes=len(x)
    )
    x_local = x[node_ids]  # features for these nodes
    return Data(x=x_local, edge_index=edge_index_local, edge_type=edge_type)

def get_auxiliary_data(data_path, dataset_name):
    entity2id = get_entity2id(data_path, dataset_name)
    relation2id = get_relation2id(data_path, dataset_name)
    node_emb = create_node_features(entity2id, data_path)
    return node_emb, entity2id, relation2id

def create_node_features(entity2id, data_path):
    """
        Create node features by getting the SentenceTransformer embedding
        input: dict. entity->id
    """
    if os.path.exists(f"{data_path}/node_embeddings.pkl"):
        return pkl.load(open(f"{data_path}/node_embeddings.pkl", "rb"))

    sentence_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    entities = [entity for entity in entity2id]

    node_embeddings = torch.tensor(sentence_encoder.encode(entities))
    pkl.dump(node_embeddings, open(f"{data_path}/node_embeddings.pkl", "wb"))

    return torch.tensor(node_embeddings)

def triplets_to_graph(triplets, embeddings):
    """
    Convert a list of (src, rel, dst) triplets into a local PyG Data graph.

    triplets: list of (src_id, rel_id, dst_id) where src/dst are GLOBAL IDs
    embeddings: global node embedding matrix [num_nodes, dim]
    """

    # --- Extract components ---
    src = torch.tensor([t[0] for t in triplets], dtype=torch.long)
    dst = torch.tensor([t[2] for t in triplets], dtype=torch.long)
    edge_types = torch.tensor([t[1] for t in triplets], dtype=torch.long)
    edge_index_global = torch.stack([src, dst], dim=0)

    # --- Identify unique nodes in this subgraph ---
    node_ids = torch.unique(edge_index_global)
    node_ids = node_ids.sort().values  # ensure deterministic order

    # --- Relabel edges to LOCAL node indices (0..num_nodes-1) ---
    edge_index_local, _ = subgraph(
        subset=node_ids,
        edge_index=edge_index_global,
        relabel_nodes=True,
        num_nodes=embeddings.size(0)
    )

    # --- Gather node features for this subgraph ---
    x_local = embeddings[node_ids]

    # --- Package into PyG Data ---
    data = Data(
        x=embeddings,
        edge_index=edge_index_global,
        edge_type=edge_types,
        n_id=node_ids  # optional: map local->global
    )

    return data

def get_num_rel(path):
    with open(f'{path}/relation2id.txt') as f:
        lines = f.readlines()
    return int(lines[0])

def get_id2ent(path):
    ret = {}
    with open(f'{path}/id2entity.txt') as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip('\n').split("\t")
            id, token = int(cur[0]), cur[1]
            ret[id] = token
    return ret

def get_id2rel(path):
    ret = {}
    with open(f'{path}/id2entity.txt') as f: 
        lines = f.readlines()
        for line in lines:
            cur = line.strip('\n').split("\t")
            id, token = int(cur[0]), cur[1]
            ret[id] = token
    return ret

def get_entity2id(path, dataset='NBA'):
    entity2id = {}
    
    with open(f'{path}/entity2id.txt') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]        
        if dataset == 'NBA':
            for line in lines[1:]:
                ent_id = line.split("\t")
                ent, id = ent_id[0], int(ent_id[1])
                entity2id[ent] = id
        if dataset == 'icews14':
            for line in lines:
                ent_id = line.split("\t")
                ent, id = ent_id[0], int(ent_id[1])
                entity2id[ent] = id
        if dataset == 'YAGO3-10':
            for line in lines:
                ent_id = line.split("\t")
                ent, id = ent_id[0], int(ent_id[1])
                entity2id[ent] = id

    return entity2id

def get_relation2id(path, dataset='nba'):
    relation2id = {}

    with open(f'{path}/relation2id.txt') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]        
        if dataset == 'NBA':
            for line in lines[1:]:
                ent_id = line.split("\t")
                ent, id = ent_id[0], int(ent_id[1])
                relation2id[ent] = id
        if dataset == 'icews14':
            for line in lines:
                ent_id = line.split("\t")
                ent, id = ent_id[0], int(ent_id[1])
                relation2id[ent] = id
        if dataset == 'YAGO3-10':
            for line in lines:
                ent_id = line.split("\t")
                ent, id = ent_id[0], int(ent_id[1])
                relation2id[ent] = id
        
    return relation2id 

def load_data_icews(path):
    # Load Embedding
    entity2id = get_entity2id(path, dataset="ICEWS14")
    relation2id = get_relation2id(path, dataset="ICEWS14")
    node_emb = create_node_features(entity2id, path)
    train_data = load_split_data_icews(path, 'train', node_emb, entity2id, relation2id)
    valid_data = load_split_data_icews(path, 'valid', node_emb, entity2id, relation2id)
    test_data = load_split_data_icews(path, 'test', node_emb, entity2id, relation2id)

    num_rel = len(relation2id) 
    
    return train_data, valid_data, test_data, num_rel

# This is for training with the cascade data
def create_unified_sliced_graphs(before_triplets, intermediate_triplets, after_triplets, node_emb_tensor):
    """
    Creates three PyG Data objects (before, intermediate, after) that:
    1. Share the SAME Global->Local node mapping (crucial for the model to track nodes).
    2. Contain a SLICED embedding matrix (only the rows for nodes in these graphs).
    """
    # 1. Collect ALL unique nodes across the entire training step
    # We must include 'after' nodes so the model has embeddings for the target edges
    all_nodes = set()
    for triplets in [before_triplets, intermediate_triplets, after_triplets]:
        for s, r, o in triplets:
            all_nodes.add(s)
            all_nodes.add(o)
            
    unique_nodes = sorted(list(all_nodes))
    
    # 2. Slice the embeddings (The Memory Fix)
    # node_emb_tensor must be a torch tensor on CPU
    subset_ids = torch.tensor(unique_nodes, dtype=torch.long)
    x_sliced = node_emb_tensor[subset_ids] # Shape becomes [~50, 384] instead of [123000, 384]
    
    # 3. Create Unified Mapping
    global_to_local = {gid: i for i, gid in enumerate(unique_nodes)}
    
    # 4. Helper to build a single Data object using this shared map
    def build_pyg_data(triplets):
        if not triplets:
            return Data(
                x=x_sliced,
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_type=torch.zeros(0, dtype=torch.long),
                n_id=subset_ids
            )
            
        edge_index = []
        edge_type = []
        for s, r, o in triplets:
            edge_index.append([global_to_local[s], global_to_local[o]])
            edge_type.append(r)
            
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        
        return Data(
            x=x_sliced,          # Shared sliced embeddings
            edge_index=edge_index, 
            edge_type=edge_type, 
            n_id=subset_ids      # Store original IDs for reference
        )

    # 5. Construct the three graphs
    g_before = build_pyg_data(before_triplets)
    g_inter  = build_pyg_data(intermediate_triplets)
    g_after  = build_pyg_data(after_triplets)
    
    return g_before, g_inter, g_after

def load_cascade_data_icews(data_path, split, node_emb, entity2id, relation2id, max_hops=2):
    """
    Optimized loader for YAGO/ICEWS.
    Returns list of tuples: (trigger_event, subgraph_before, subgraph_intermediate, subgraph_after, paragraph)
    """
    # Ensure embeddings are a Tensor for efficient slicing
    if not torch.is_tensor(node_emb):
        node_emb = torch.tensor(node_emb).float()
    
    cascade_file = f"{data_path}/{split}_gupdate_cascade.json"
    print(f"Loading cascades from {cascade_file}...")
    
    with open(cascade_file, "r") as f:
        cascades = json.load(f)

    ret = []

    for root_idx, hops in cascades.items():
        # Iterate over hops up to max_hops
        for hop_level in range(min(len(hops), max_hops + 1)):
            hop = hops[hop_level]

            # Flatten hop2 if it is a list-of-lists
            if hop_level == 2 and len(hop) > 0 and isinstance(hop[0], list):
                records = [record for subhop in hop for record in subhop]
            else:
                records = hop

            for record in records:
                trigger_event, subgraph_before, subgraph_after, paragraph = record

                # 1. Convert everything to Global IDs first
                # (Same logic as your original code)
                trigger_event_triplet = [
                    entity2id[trigger_event[0]],
                    relation2id[trigger_event[1]],
                    entity2id[trigger_event[2]]
                ]

                subgraph_before_triplets = [
                    [entity2id[s], relation2id[r], entity2id[t]] for s, r, t in subgraph_before
                ]

                subgraph_after_triplets = [
                    [entity2id[s], relation2id[r], entity2id[t]] for s, r, t in subgraph_after
                ]
                
                # Intermediate = Before + Trigger
                subgraph_intermediate_triplets = subgraph_before_triplets + [trigger_event_triplet]

                # 2. Create SLICED PyG Graphs
                # This replaces your old 'triplets_to_graph' calls
                g_before, g_inter, g_after = create_unified_sliced_graphs(
                    subgraph_before_triplets, 
                    subgraph_intermediate_triplets, 
                    subgraph_after_triplets, 
                    node_emb
                )

                ret.append((trigger_event, g_before, g_inter, g_after, paragraph))

    print(f"Loaded {len(ret)} training samples.")
    return ret

def collate_graphs(batch):
    from torch_geometric.data import Batch
    
    triggers = [item[0] for item in batch]
    before_list = [item[1] for item in batch]
    inter_list = [item[2] for item in batch]
    after_list = [item[3] for item in batch]
    paragraphs = [item[4] for item in batch]
    
    # Use PyG Batch to combine the small sliced graphs
    batch_before = Batch.from_data_list(before_list)
    batch_inter = Batch.from_data_list(inter_list)
    batch_after = Batch.from_data_list(after_list)
    
    return triggers, batch_before, batch_inter, batch_after, paragraphs

def load_split_data_icews(data_path, split, node_emb, entity2id, relation2id):
    paragraph_data_path = f'{data_path}/{split}_gupdate_rule_paragraphs_finetuned.jsonl'
    if split == 'test':
        data_path = f'{data_path}/{split}_gupdate_rule.jsonl' 
    else:
        data_path = f'{data_path}/{split}_gupdate_rule_paragraphs.jsonl' 

    dataset_df = pd.read_json(path_or_buf=data_path, lines=True)
    paragraph_df = pd.read_json(path_or_buf=paragraph_data_path, lines=True)
    
    ret = []
    for i, row in dataset_df.iterrows():
        paragraph_data = paragraph_df.iloc[i]
         
        paragraph = paragraph_data['generated_paragraph']
        trigger_event = row['trigger_event'] # [s, r, t]
        subgraph_before = row['subgraph_before'] # [[s, r, t], [s, r, t], ... , [s, r, t]]
        subgraph_after = row['subgraph_after'] # [[s, r, t], [s, r, t], ... , [s, r, t]] 
        
        # create the ID triplets from the entity2id, relation2id
        trigger_event_triplet = [
            entity2id[trigger_event[0]],
            relation2id[trigger_event[1]],
            entity2id[trigger_event[2]]
        ]

        subgraph_before_triplets = [
            [entity2id[s], relation2id[r], entity2id[t]] for s, r, t in subgraph_before
        ]

        subgraph_after_triplets = [
            [entity2id[s], relation2id[r], entity2id[t]] for s, r, t in subgraph_after
        ]

        subgraph_before_pyG = triplets_to_graph(subgraph_before_triplets, node_emb)
        subgraph_before_triplets.append(trigger_event_triplet)
        subgraph_intermediate_pyG = triplets_to_graph(subgraph_before_triplets, node_emb)
        subgraph_after_pyG = triplets_to_graph(subgraph_after_triplets, node_emb)

        ret.append((trigger_event, subgraph_before_pyG, subgraph_intermediate_pyG, subgraph_after_pyG, paragraph)) 
    return ret

def load_data_nba(path):
    # Load Embedding
    node_emb = create_node_features(get_entity2id(path))
    id2ent = get_id2ent(path)
    id2rel = get_id2rel(path)

    # Load different splits 
    train_data = load_split_data_nba(path, 'train', node_emb, id2ent, id2rel)
    valid_data = load_split_data_nba(path, 'valid', node_emb, id2ent, id2rel)
    test_data = load_split_data_nba(path, 'test', node_emb, id2ent, id2rel)

    num_rel = get_num_rel(path)

    return train_data, valid_data, test_data, num_rel

def triplets_to_language(triplets, id2ent, id2rel):
    return f"{id2ent[triplets[0]]} {id2rel[triplets[1]]} {id2ent[triplets[2]]}"

def load_split_data_nba(data_path, split, node_emb, id2ent, id2rel):
    """
        Helper function to load data.
        Output: 
        1) the original subgraph.
        2) the subgraph with the directly changed triplets.
        3) the final modified subgraph 
    """
    paragraph_data_path = f'{data_path}/{split}_gupdate_paragraphs.jsonl'
    paragraph_df = pd.read_json(path_or_buf=paragraph_data_path, lines=True)

    data_path = f'{data_path}/NBAtransactions_{split}.json'
    with open(data_path) as f:
        data = json.load(f)
    ret = []
    for i,d in enumerate(tqdm(data)): 
        paragraph_data = paragraph_df.iloc[i]
        paragraph = paragraph_data['paragraph']

        subgraph_before = d['subgraph_before']
        subgraph_before_pyG = triplets_to_graph(subgraph_before, node_emb)

        subgraph_after = d['subgraph_after']
        subgraph_after_pyG = triplets_to_graph(subgraph_after, node_emb)

        entities = d['text_mentioned_entities']

        # Convert lists to sets for set operations
        G1_set = set(tuple(triplet) for triplet in subgraph_before)
        G2_set = set(tuple(triplet) for triplet in subgraph_after)

        # Calculate added and deleted triplets
        added_triplets = G2_set - G1_set
        deleted_triplets = G1_set - G2_set
        
        direct_change_triplets_add = []
        direct_change_triplets_del = []

        # Get the direct change triplets (IE-GOLD)
        direct_change_triplets_add = []
        direct_change_triplets_del = []
        for triplet in added_triplets:
            if triplet[0] in entities and triplet[2] in entities:
                direct_change_triplets_add.append(triplet) 
        for triplet in deleted_triplets:
            if triplet[0] in entities and triplet[2] in entities:
                direct_change_triplets_del.append(triplet)

        subgraph_intermediate = subgraph_before.copy()
        for tri in direct_change_triplets_add:
            if tri not in subgraph_intermediate:
                subgraph_intermediate.append(tri)
        for tri in direct_change_triplets_del:
            if tri in subgraph_intermediate:
                subgraph_intermediate.remove(tri)

        subgraph_intermediate_pyG = triplets_to_graph(subgraph_intermediate, node_emb)

        direct_change_triplets_add_text = [triplets_to_language(triplet, id2ent, id2rel) for triplet in direct_change_triplets_add]
        direct_change_triplets_del_text = [triplets_to_language(triplet, id2ent, id2rel) for triplet in direct_change_triplets_del]

        direct_change_triplets_add_text.extend(direct_change_triplets_del_text)
        ret.append((direct_change_triplets_add_text, subgraph_before_pyG, subgraph_intermediate_pyG, subgraph_after_pyG, paragraph))

    return ret

if __name__ == "__main__":
    data_path = './data/icews14'
    d = load_data_icews(data_path, 'train')

    subgraph_before = d['subgraph_before']
    subgraph_after = d['subgraph_after']
    entities = d['text_mentioned_entities']

    # Convert lists to sets for set operations
    G1_set = set(tuple(triplet) for triplet in subgraph_before)
    G2_set = set(tuple(triplet) for triplet in subgraph_after)

    # Calculate added and deleted triplets
    added_triplets = G2_set - G1_set
    deleted_triplets = G1_set - G2_set

    print(len(d))
    print(d[0]['text_mentioned_entities'])
    print(d[0].keys())