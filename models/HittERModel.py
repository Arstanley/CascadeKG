"""
HittER: Hierarchical Transformers for Knowledge Graph Embeddings
Based on: Chen et al., "HittER: Hierarchical Transformers for Knowledge Graph Embeddings", EMNLP 2021
Optimized vectorized implementation.
"""
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from models import DistMultDecoder


class HittEREncoder(nn.Module):
    """
    HittER Encoder with hierarchical Transformer architecture.
    
    Architecture:
    - Bottom block: extracts features of each entity-relation pair in the local neighborhood
    - Top block: aggregates relational information from outputs of the bottom block
    """
        
    def __init__(self, hidden_dim, num_relations, num_heads=8, num_bottom_layers=2, num_top_layers=2, dropout=0.1, max_neighbors=50):
        super(HittEREncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.max_neighbors = max_neighbors
        
        # Entity and relation embeddings
        self.entity_emb_dim = hidden_dim
        self.relation_emb_dim = hidden_dim
        
        # Learnable relation embeddings
        self.relation_emb = nn.Embedding(num_relations, hidden_dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        
        # Type embeddings (similar to BERT)
        self.type_embeddings = nn.Embedding(3, hidden_dim)  # [CLS], entity, relation
        
        # Bottom Transformer: processes entity-relation pairs
        bottom_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.bottom_transformer = TransformerEncoder(bottom_layer, num_layers=num_bottom_layers)
        
        # Top Transformer: aggregates relational information
        top_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.top_transformer = TransformerEncoder(top_layer, num_layers=num_top_layers)
        
        # Projection layers
        self.entity_proj = nn.Linear(hidden_dim, hidden_dim)
        self.relation_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # [CLS] token embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
    def forward(self, x, edge_index, edge_type):
        """
        Forward pass of HittER encoder (Vectorized).
        
        Args:
            x: node features [num_nodes, hidden_dim]
            edge_index: edge indices [2, num_edges]
            edge_type: relation types [num_edges]
        
        Returns:
            entity_embeddings: [num_nodes, hidden_dim] - updated entity embeddings
        """
        num_nodes = x.size(0)
        device = x.device
        
        if edge_index.size(1) == 0:
            return x
            
        # 1. Sort edges by source to group neighbors
        src, dst = edge_index
        perm = torch.argsort(src)
        src = src[perm]
        dst = dst[perm]
        rel = edge_type[perm]
        
        # 2. Compute rank of each neighbor within its source node's list
        degrees = torch.bincount(src, minlength=num_nodes)
        
        # Calculate start indices for each node in the sorted list
        # cumsum gives the end index of each block
        ends = torch.cumsum(degrees, dim=0)
        # Shift to get starts: [0, end_0, end_1, ...]
        starts = torch.cat([torch.zeros(1, device=device, dtype=torch.long), ends[:-1]])
        
        # Rank = global_index - start_index_of_source
        edge_indices = torch.arange(src.size(0), device=device)
        block_starts = starts[src]
        ranks = edge_indices - block_starts
        
        # 3. Filter to max_neighbors
        mask = ranks < self.max_neighbors
        
        src_filtered = src[mask]
        dst_filtered = dst[mask]
        rel_filtered = rel[mask]
        ranks_filtered = ranks[mask]
        
        # 4. Prepare features for pairs (source + relation)
        # Note: HittER uses (source + relation) as the feature, not target.
        # Original code: pair_emb = self.entity_proj(src_emb) + rel_emb
        # src_emb comes from x[src_nodes].
        
        src_emb = x[src_filtered]
        rel_emb_raw = self.relation_emb(rel_filtered)
        rel_emb = self.relation_proj(rel_emb_raw)
        
        pair_emb = self.entity_proj(src_emb) + rel_emb
        
        # 5. Scatter into dense batch [num_nodes, max_neighbors, dim]
        batch_size = num_nodes
        seq_len = self.max_neighbors
        
        # Initialize with zeros
        dense_pairs = torch.zeros(batch_size, seq_len, self.hidden_dim, device=device)
        
        # Fill at [src, rank]
        dense_pairs[src_filtered, ranks_filtered] = pair_emb
        
        # 6. Create Padding Mask [num_nodes, 1 + max_neighbors]
        # True means padded (ignored).
        padding_mask = torch.ones(batch_size, 1 + seq_len, dtype=torch.bool, device=device)
        
        # CLS token at index 0 is always valid
        padding_mask[:, 0] = False
        
        # Valid neighbors at 1 + rank
        padding_mask[src_filtered, 1 + ranks_filtered] = False
        
        # 7. Prepare input sequence: [CLS] + pairs
        cls_tokens = self.cls_token.expand(batch_size, 1, -1) # [N, 1, D]
        sequence = torch.cat([cls_tokens, dense_pairs], dim=1) # [N, 1+K, D]
        
        # 8. Add Type Embeddings
        # 0 for CLS, 1 for others
        type_ids = torch.ones(batch_size, 1 + seq_len, dtype=torch.long, device=device)
        type_ids[:, 0] = 0
        
        # Mask out padded positions in type_ids? Not strictly necessary as they are masked in attention,
        # but good for cleanliness.
        type_ids[padding_mask] = 2 # Or some dummy type, though 1 is fine.
        
        type_emb = self.type_embeddings(type_ids)
        sequence = sequence + type_emb
        
        # 9. Transformers
        # Bottom Transformer
        bottom_out = self.bottom_transformer(sequence, src_key_padding_mask=padding_mask)
        
        # Extract CLS and Pairs
        cls_repr = bottom_out[:, 0, :].unsqueeze(1) # [N, 1, D]
        pair_reprs = bottom_out[:, 1:, :] # [N, K, D]
        
        # Top Transformer
        # Input: CLS as query, Pairs as context
        top_input = torch.cat([cls_repr, pair_reprs], dim=1)
        top_out = self.top_transformer(top_input, src_key_padding_mask=padding_mask)
        
        final_cls = top_out[:, 0, :] # [N, D]
        
        # 10. Update embeddings
        # Only update nodes that had neighbors. 
        # Original code skipped nodes with 0 neighbors.
        has_neighbors = degrees > 0
        
        updated_x = x.clone()
        updated_x[has_neighbors] = final_cls[has_neighbors]
        
        return updated_x


class HittERModel(nn.Module):
    """
    Complete HittER model with encoder and decoder.
    """
    def __init__(
        self,
        num_entities,
        num_relations,
        hidden_dim=768,
        num_heads=8,
        num_bottom_layers=2,
        num_top_layers=2,
        dropout=0.1,
        max_neighbors=50
    ):
        super(HittERModel, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        
        # Entity embeddings (initialized randomly)
        self.entity_emb = nn.Embedding(num_entities, hidden_dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        
        # HittER encoder
        self.encoder = HittEREncoder(
            hidden_dim=hidden_dim,
            num_relations=num_relations,
            num_heads=num_heads,
            num_bottom_layers=num_bottom_layers,
            num_top_layers=num_top_layers,
            dropout=dropout,
            max_neighbors=max_neighbors
        )
        
        # Decoder (using DistMult)
        self.decoder = DistMultDecoder.DistMultDecoder(num_relations, hidden_dim)
    
    def encode(self, x, edge_index, edge_type, z_text=None):
        """
        Encode graph using HittER.
        
        Args:
            x: node features [num_nodes, hidden_dim] (can be entity embeddings)
            edge_index: edge indices [2, num_edges]
            edge_type: relation types [num_edges]
            z_text: text features (optional, for compatibility)
        
        Returns:
            z: updated entity embeddings [num_nodes, hidden_dim]
        """
        num_nodes = x.size(0) if x is not None and x.size(0) > 0 else 0
        
        # If x is None or empty, try to get nodes from edge_index
        if num_nodes == 0:
            if edge_index.size(1) > 0:
                unique_nodes = torch.unique(edge_index)
                num_nodes = unique_nodes.max().item() + 1
                x = self.entity_emb.weight[:num_nodes] if num_nodes <= self.num_entities else self.entity_emb(unique_nodes)
            else:
                # Empty graph, return empty embeddings
                return torch.empty(0, self.hidden_dim, device=edge_index.device if edge_index.numel() > 0 else torch.device('cpu'))
        
        # Use HittER encoder to process the graph
        # The encoder processes neighborhoods and updates embeddings
        z = self.encoder(x, edge_index, edge_type)
        
        return z
    
    def decode(self, z, edge_index, edge_type):
        """
        Decode edges using DistMult decoder.
        """
        return self.decoder(z, edge_index, edge_type)
