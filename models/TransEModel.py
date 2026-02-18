import torch.nn as nn
import torch

class TransEDecoder(nn.Module):
    def __init__(self, num_relations, embedding_dim, p_norm=1):
        super().__init__()
        self.rel_emb = nn.Embedding(num_relations, embedding_dim)
        self.p_norm = p_norm
        # learnable temperature and bias to turn distances into logits
        self.scale = nn.Parameter(torch.tensor(1.0))  # >0 encourages separation
        self.bias  = nn.Parameter(torch.tensor(0.0))

    def forward(self, z, edge_index, edge_type):
        h = z[edge_index[0]]
        t = z[edge_index[1]]
        r = self.rel_emb(edge_type)
        d = torch.norm(h + r - t, p=self.p_norm, dim=1)  # distance >= 0
        logit = -(self.scale * d) + self.bias            # higher is better
        return logit

class TransEOnlyModel(nn.Module):
    def __init__(self, num_nodes, num_relations, embedding_dim, p_norm=1):
        super().__init__()
        self.entity_emb = nn.Embedding(num_nodes, embedding_dim)
        self.decoder = TransEDecoder(num_relations, embedding_dim, p_norm)

    def encode(self, *args, **kwargs):
        return self.entity_emb.weight

    def decode(self, z, edge_index, edge_type):
        return self.decoder(z, edge_index, edge_type)