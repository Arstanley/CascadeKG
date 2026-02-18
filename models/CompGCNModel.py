import torch
import torch.nn as nn
from models.CompGCNEncoder import CompGCNEncoder

class CompGCNModel(nn.Module):
    def __init__(self, num_nodes, num_relations, in_channels, hidden_channels, num_layers=2, dropout=0.3, op='mult'):
        super().__init__()
        # Initial relation embeddings matching input dimension
        self.relation_emb = nn.Parameter(torch.Tensor(num_relations, in_channels))
        
        # Encoder
        self.encoder = CompGCNEncoder(in_channels, hidden_channels, num_relations, num_layers, dropout, op)
        
        # Helper storage for decoding
        self.final_rels = None
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.relation_emb)

    def encode(self, x, edge_index, edge_type, z_text=None):
        # x: [N, in_channels]
        # In this codebase, x corresponds to SentenceTransformer features (dim=384)
        
        # Forward pass through encoder
        z, final_rels = self.encoder(x, edge_index, edge_type, self.relation_emb)
        
        self.final_rels = final_rels
        return z

    def decode(self, z, edge_index, edge_type):
        # Decode using DistMult score with updated relations
        # z: [N, hidden_channels]
        # final_rels: [num_relations, hidden_channels]
        
        r = self.final_rels[edge_type]
        z_src = z[edge_index[0]]
        z_dst = z[edge_index[1]]
        
        # DistMult score: <h, r, t>
        score = torch.sum(z_src * r * z_dst, dim=1)
        return score



