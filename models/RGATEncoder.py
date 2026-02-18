import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class RGATEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations, z_text_dim=384, heads=4):
        super().__init__()
        self.rel_emb = nn.Embedding(num_relations, in_channels)
        self.heads = heads

        # Project text into same space as input features
        self.text_proj = nn.Linear(z_text_dim, in_channels)

        self.att1 = GATConv(in_channels, out_channels // heads, heads=heads)
        self.att2 = GATConv(out_channels, out_channels // heads, heads=heads)

    def forward(self, x, edge_index, edge_type, z_text=None):
        if z_text is not None:
            z_text_proj = self.text_proj(z_text)  # [in_channels]
            x = x + z_text_proj.unsqueeze(0)  # Broadcast to all nodes

        x = self.att1(x, edge_index)
        x = F.relu(x)
        x = self.att2(x, edge_index)
        return x