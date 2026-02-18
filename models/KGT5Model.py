import torch
import torch.nn as nn


class KGT5Model(nn.Module):
    """
    KGT5-inspired baseline: asymmetric encoder with dot-product scoring.
    Ignores graph structure entirely — encode() returns learned entity embeddings.
    """

    def __init__(self, num_nodes, num_relations, embedding_dim):
        super().__init__()
        self.entity_emb = nn.Embedding(num_nodes, embedding_dim)
        self.rel_emb = nn.Embedding(num_relations, embedding_dim)

        # Asymmetric projections
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.target_proj = nn.Linear(embedding_dim, embedding_dim)

        # Learnable scale and bias
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def encode(self, *args, **kwargs):
        return self.entity_emb.weight

    def decode(self, z, edge_index, edge_type):
        h = z[edge_index[0]]
        t = z[edge_index[1]]
        r = self.rel_emb(edge_type)

        query = self.query_proj(h + r)
        target = self.target_proj(t)

        # Dot-product scoring
        score = (query * target).sum(dim=1) * self.scale + self.bias
        return score
