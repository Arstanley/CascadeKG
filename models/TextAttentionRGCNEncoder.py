import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class TextAttentionRGCNEncoder(nn.Module):
    def __init__(self, hidden_dim, num_relations, text_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        # RGCN layers
        self.convs = nn.ModuleList()
        self.acts = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations))
            self.acts.append(nn.ReLU())

        # Text projection
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Attention mechanism
        self.attn_proj_node = nn.Linear(hidden_dim, hidden_dim)
        self.attn_proj_text = nn.Linear(hidden_dim, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_type, z_text):
        """
        x: [N, hidden_dim] - node features
        edge_index: [2, E]
        edge_type: [E]
        z_text: [text_dim] - event text embedding
        """
        # Project text
        z = self.text_proj(z_text)  # [hidden_dim]

        for i in range(self.num_layers):
            # RGCN update
            x = self.convs[i](x, edge_index, edge_type)
            x = self.acts[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Compute attention between node features and text
            h_node = self.attn_proj_node(x)              # [N, hidden_dim]
            h_text = self.attn_proj_text(z).unsqueeze(0) # [1, hidden_dim]
            attn_input = torch.tanh(h_node + h_text)     # [N, hidden_dim]
            alpha = torch.sigmoid(self.attn_score(attn_input))  # [N, 1]

            # Fuse: weighted combination
            x = alpha * x + (1 - alpha) * z  # [N, hidden_dim]

        return x