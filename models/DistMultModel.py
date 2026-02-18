import torch
import torch.nn as nn
from models import DistMultDecoder

class DistMultOnlyModel(torch.nn.Module):
    def __init__(self, num_nodes, num_relations, embedding_dim):
        super().__init__()
        self.entity_emb = torch.nn.Embedding(num_nodes, embedding_dim)
        self.decoder = DistMultDecoder.DistMultDecoder(num_relations, embedding_dim)

    def encode(self, x=None, edge_index=None, edge_type=None, z_text=None):
        return self.entity_emb.weight  # shape: [num_nodes, dim]

    def decode(self, z, edge_index, edge_type):
        return self.decoder(z, edge_index, edge_type)