import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class EventConditionedRGATLayer(MessagePassing):
    """
    A relational GAT layer conditioned on an event description.
    """
    def __init__(self, in_dim, out_dim, num_relations, text_dim, leaky_slope=0.2):
        super().__init__(aggr='add')  # "Add" aggregation.
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations

        # Linear transforms
        self.lin_node = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_msg = nn.Linear(in_dim, out_dim, bias=False)
        self.rel_emb = nn.Embedding(num_relations, in_dim)
        self.lin_rel = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_text = nn.Linear(text_dim, out_dim, bias=False)

        # Attention parameters
        # we concatenate [h_u || h_v || rel_msg || text] -> 4*out_dim
        self.att = nn.Parameter(torch.Tensor(4 * out_dim))
        self.leaky = nn.LeakyReLU(leaky_slope)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_node.weight)
        nn.init.xavier_uniform_(self.lin_msg.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)
        nn.init.xavier_uniform_(self.lin_rel.weight)
        nn.init.xavier_uniform_(self.lin_text.weight)
        nn.init.xavier_uniform_(self.att.unsqueeze(0))

    def forward(self, x, edge_index, edge_type, z_text):
        """
        x: [N, in_dim] node features
        edge_index: [2, E] edge indices
        edge_type: [E] relation type ids
        z_text: [text_dim] event text embedding
        """
        # Precompute transforms
        h = self.lin_node(x)           # [N, out_dim]
        rel = self.rel_emb(edge_type) # [E, in_dim]
        rel_msg = self.lin_rel(rel)    # [E, out_dim]
        z = self.lin_text(z_text)      # [out_dim]

        # propagate arguments
        return self.propagate(
            edge_index,
            x=x,
            h=h,
            rel_msg=rel_msg,
            z=z
        )

    def message(self, x_j, h_j, h_i, rel_msg, z, index):
        """
        x_j: [E, in_dim] source node features
        h_j: [E, out_dim] transformed source embeddings
        h_i: [E, out_dim] transformed target embeddings
        rel_msg: [E, out_dim] relation message
        z: [out_dim] text embedding
        index: [E] target indices for softmax
        """
        # Prepare text for each edge
        z_expand = z.unsqueeze(0).expand(h_j.size(0), -1)  # [E, out_dim]
        # Compute unnormalized attention scores
        cat = torch.cat([h_j, h_i, rel_msg, z_expand], dim=-1)  # [E, 4*out_dim]
        e = self.leaky((cat * self.att).sum(dim=-1))             # [E]
        # Normalize
        alpha = softmax(e, index)                                # [E]
        # Compute message
        m = self.lin_msg(x_j)                                    # [E, out_dim]
        return m * alpha.unsqueeze(-1)                          # [E, out_dim]

    def update(self, aggr_out):
        # aggr_out: [N, out_dim]
        return F.relu(aggr_out)


class EventConditionedEncoder(nn.Module):
    """
    Stacks multiple EventConditionedRGAT layers.
    """
    def __init__(self, in_dim, hidden_dims, num_relations, text_dim, dropout=0.1):
        super().__init__()
        layers = []
        dims = [in_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers.append(
                EventConditionedRGATLayer(
                    in_dim=dims[i],
                    out_dim=dims[i+1],
                    num_relations=num_relations,
                    text_dim=text_dim
                )
            )
        self.layers = nn.ModuleList(layers)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type, z_text):
        h = x
        for layer in self.layers:
            h = layer(h, edge_index, edge_type, z_text)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

class DistMultDecoder(nn.Module):
    """
    A simple DistMult decoder for link scoring.
    """
    def __init__(self, embed_dim, num_relations):
        super().__init__()
        self.rel_weight = nn.Parameter(torch.Tensor(num_relations, embed_dim))
        nn.init.xavier_uniform_(self.rel_weight)

    def forward(self, h, edge_index, edge_type):
        # h: [N, embed_dim], compute score for edges
        src, dst = edge_index
        r = edge_type
        w = self.rel_weight[r]  # [E, embed_dim]
        score = torch.sum(h[src] * w * h[dst], dim=-1)
        return torch.sigmoid(score)  # [E]


# Example usage:
# encoder = EventConditionedEncoder(in_dim=128, hidden_dims=[128,128], num_relations=50, text_dim=256)
# decoder = DistMultDecoder(embed_dim=128, num_relations=50)
# x_encoded = encoder(x, edge_index, edge_type, z_text)
# preds = decoder(x_encoded, edge_index, edge_type)