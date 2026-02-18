import torch
from tqdm import tqdm

def add_edges(subgraph, pred_edges, pred_types, source_subgraph=None):
    """
    Add predicted edges (pred_edges, pred_types) to 'subgraph'.
    Handles local/global node mapping automatically if source_subgraph differs.

    Args:
        subgraph (torch_geometric.data.Data): Target subgraph (to be updated)
        pred_edges (torch.Tensor): shape [2, E_pred], edge indices from source_subgraph
        pred_types (torch.Tensor): shape [E_pred], relation types corresponding to pred_edges
        source_subgraph (torch_geometric.data.Data, optional): The subgraph from which
            pred_edges were predicted. If provided, node indices will be remapped to match
            the target subgraph.
    
    Returns:
        updated_subgraph (torch_geometric.data.Data): Copy of `subgraph` with added edges
    """
    # Clone the target to avoid in-place modifications
    updated = subgraph.clone()

    if source_subgraph is not None:
        # Both subgraphs have node_id attributes referencing GLOBAL IDs
        src_global_ids = source_subgraph.n_id.cpu().tolist()
        tgt_global_ids = subgraph.n_id.cpu().tolist()

        # Mapping from global_id -> local_index in target
        global2local = {gid: i for i, gid in enumerate(tgt_global_ids)}

        # Map source-local edge indices to target-local indices via global ids
        mapped_edges = []
        for src_local, dst_local in pred_edges.t().tolist():
            src_gid = src_global_ids[src_local]
            dst_gid = src_global_ids[dst_local]

            # Add edge only if both nodes exist in target
            if src_gid in global2local and dst_gid in global2local:
                mapped_edges.append([global2local[src_gid], global2local[dst_gid]])

        if not mapped_edges:
            # Nothing to add
            return updated

        mapped_edges = torch.tensor(mapped_edges, dtype=torch.long).t().to(subgraph.edge_index.device)
        mapped_types = pred_types[:mapped_edges.size(1)].to(subgraph.edge_type.device)
    else:
        # No mapping needed; assume same subgraph indexing
        mapped_edges = pred_edges.to(subgraph.edge_index.device)
        mapped_types = pred_types.to(subgraph.edge_type.device)

    # Concatenate to form new edge_index and edge_type
    updated.edge_index = torch.cat([updated.edge_index, mapped_edges], dim=1)
    updated.edge_type = torch.cat([updated.edge_type, mapped_types], dim=0)

    return updated

def negative_sampling(edge_index, num_nodes, num_neg_samples=None):
    """
    Sample negative edges by corrupting either the subject or the object of each edge.
    
    Args:
        edge_index (LongTensor): [2, E] tensor of existing edges.
        num_nodes (int): number of nodes in the graph.
        num_negative_edges (int, optional): how many negative edges to sample.
            If None, will default to the same number as edges in edge_index.
    """
    E = edge_index.size(1)
    if num_neg_samples is None:
        num_neg_samples = E

    # Repeat edges if we need more negatives than positives
    if num_neg_samples <= E:
        # Subsample edges
        idx = torch.randint(E, (num_neg_samples,), device=edge_index.device)
    else:
        # Oversample with replacement
        idx = torch.randint(E, (num_neg_samples,), device=edge_index.device)

    base_edges = edge_index[:, idx].clone()

    # Decide which side to corrupt
    mask_1 = torch.rand(num_neg_samples, device=base_edges.device) < 0.5
    mask_2 = ~mask_1

    # Corrupt subjects or objects
    base_edges[0, mask_1] = torch.randint(num_nodes, (mask_1.sum(),), device=base_edges.device)
    base_edges[1, mask_2] = torch.randint(num_nodes, (mask_2.sum(),), device=base_edges.device)

    return base_edges

# Evaluation
@torch.no_grad()
def compute_rank(ranks):
    # fair ranking prediction as the average
    # of optimistic and pessimistic ranking
    true = ranks[0]
    optimistic = (ranks > true).sum() + 1
    pessimistic = (ranks >= true).sum()
    return (optimistic + pessimistic).float() * 0.5

@torch.no_grad()
def compute_mrr(z, data, edge_index, edge_type, model):
    ranks = []
    for i in range(edge_type.numel()):
        (src, dst), rel = edge_index[:, i], edge_type[i]

        # try all nodes as tails but delete true triplets
        tail_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        for (heads, tails), types in [
            (data.edge_index, data.edge_type),
        ]:
            tail_mask[tails[(heads == src) & (types == rel)]] = False

        tail = torch.arange(data.num_nodes)[tail_mask]
        tail = torch.cat([torch.tensor([dst]), tail])
        head = torch.full_like(tail, fill_value=src)
        eval_edge_index = torch.stack([head, tail], dim=0).to('cuda')
        eval_edge_type = torch.full_like(tail, fill_value=rel).to('cuda')

        out = model.decode(z, eval_edge_index, eval_edge_type)
        rank = compute_rank(out)
        ranks.append(rank)

        # Try all nodes as heads, but delete true triplets:
        head_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        for (heads, tails), types in [
            (data.edge_index, data.edge_type),
        ]:
            head_mask[heads[(tails == dst) & (types == rel)]] = False

        head = torch.arange(data.num_nodes)[head_mask]
        head = torch.cat([torch.tensor([src]), head])
        tail = torch.full_like(head, fill_value=dst)
        eval_edge_index = torch.stack([head, tail], dim=0).to('cuda')
        eval_edge_type = torch.full_like(head, fill_value=rel).to('cuda')

        out = model.decode(z, eval_edge_index, eval_edge_type)
        rank = compute_rank(out)
        ranks.append(rank)

    return (1. / torch.tensor(ranks, dtype=torch.float)).mean()