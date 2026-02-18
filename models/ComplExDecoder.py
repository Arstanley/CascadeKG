import torch
import torch.nn as nn

class ComplExDecoder(nn.Module):
    def __init__(self, num_relations, embedding_dim):
        super(ComplExDecoder, self).__init__()
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim // 2
        self.rel_emb_real = nn.Embedding(num_relations, self.embedding_dim)
        self.rel_emb_img = nn.Embedding(num_relations, self.embedding_dim)

        nn.init.xavier_uniform_(self.rel_emb_real.weight)
        nn.init.xavier_uniform_(self.rel_emb_img.weight)

    def forward(self, z, edge_index, edge_type):
        head_index = edge_index[0]
        tail_index = edge_index[1]
        rel_type = edge_type.squeeze()

        head_emb_real = z[head_index, :self.embedding_dim]
        head_emb_img = z[head_index, self.embedding_dim:]
        tail_emb_real = z[tail_index, :self.embedding_dim]
        tail_emb_img = z[tail_index, self.embedding_dim:]
        rel_emb_real = self.rel_emb_real(rel_type)
        rel_emb_img = self.rel_emb_img(rel_type)

        score = torch.sum(rel_emb_real * head_emb_real * tail_emb_real +
                          rel_emb_img * head_emb_real * (-tail_emb_img) +
                          rel_emb_real * head_emb_img * tail_emb_img +
                          rel_emb_img * head_emb_img * tail_emb_real, dim=1)

        return score