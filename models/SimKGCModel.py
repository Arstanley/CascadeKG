import torch
import torch.nn as nn
import torch.nn.functional as F


class SimKGCModel(nn.Module):
    """
    SimKGC-inspired baseline: bi-encoder with cosine similarity scoring.
    Ignores graph structure entirely — encode() returns learned entity embeddings.
    """

    def __init__(self, num_nodes, num_relations, embedding_dim, mlp_hidden=None):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = embedding_dim

        self.entity_emb = nn.Embedding(num_nodes, embedding_dim)
        self.rel_emb = nn.Embedding(num_relations, embedding_dim)

        # Query encoder: MLP over (head + relation)
        self.query_mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, embedding_dim),
        )

        # Target encoder: MLP over tail
        self.target_mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, embedding_dim),
        )

        # Learnable temperature and bias
        self.temperature = nn.Parameter(torch.tensor(20.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def encode(self, *args, **kwargs):
        return self.entity_emb.weight

    def decode(self, z, edge_index, edge_type):
        h = z[edge_index[0]]
        t = z[edge_index[1]]
        r = self.rel_emb(edge_type)

        query = self.query_mlp(h + r)
        target = self.target_mlp(t)

        # Cosine similarity scaled by temperature + bias
        score = F.cosine_similarity(query, target, dim=1) * self.temperature + self.bias
        return score
