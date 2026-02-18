# models/RGCNEncoder.py
import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv

class RGCNEncoder(nn.Module):
    def __init__(self, hidden_dim, num_relations, num_layers=2, dropout=0.3):
        super(RGCNEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.dropout = dropout

        # First RGCN layer
        self.convs.append(RGCNConv(in_channels=hidden_dim, out_channels=hidden_dim, num_relations=num_relations))
        self.acts.append(nn.ReLU())

        # Additional layers
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(in_channels=hidden_dim, out_channels=hidden_dim, num_relations=num_relations))
            self.acts.append(nn.ReLU())

    def forward(self, x, edge_index, edge_type):
        for conv, act in zip(self.convs, self.acts):
            x = conv(x, edge_index, edge_type)
            x = act(x)
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x


class TextConditionedRGCNEncoder(nn.Module):
    def __init__(self, hidden_dim, num_relations, text_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.rgcn = RGCNEncoder(hidden_dim, num_relations, num_layers, dropout)

    def forward(self, x, edge_index, edge_type, z_text):
        # z_text: [text_dim] or [1, text_dim]
        if z_text.dim() == 2 and z_text.size(0) == 1:
            z_text = z_text.squeeze(0)  # [text_dim]
        z_proj = self.text_proj(z_text)  # [hidden_dim]

        # Broadcast to all nodes 
        z_node = z_proj.unsqueeze(0).expand(x.size(0), -1)  # [N, hidden_dim]

        # Combine with input features (e.g., addition or gating)
        x = x + z_node  # or torch.cat([x, z_node], dim=1) if you change input dim of RGCN

        return self.rgcn(x, edge_index, edge_type)


class TextConditionedRGCNEncoderGated(nn.Module):
    def __init__(self, hidden_dim, num_relations, text_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.text_gate = nn.Linear(text_dim, hidden_dim)
        self.text_update = nn.Linear(text_dim, hidden_dim)
        self.rgcn = RGCNEncoder(hidden_dim, num_relations, num_layers, dropout)

    def forward(self, x, edge_index, edge_type, z_text):
        if z_text.dim() == 2 and z_text.size(0) == 1:
            z_text = z_text.squeeze(0)
        gate = torch.sigmoid(self.text_gate(z_text))
        update = self.text_update(z_text)
        z_node = (gate * update).unsqueeze(0).expand(x.size(0), -1)
        x = x + z_node
        return self.rgcn(x, edge_index, edge_type)


class TextConditionedRGCNEncoderConcat(nn.Module):
    def __init__(self, hidden_dim, num_relations, text_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.rgcn = RGCNEncoder(hidden_dim, num_relations, num_layers, dropout)

    def forward(self, x, edge_index, edge_type, z_text):
        if z_text.dim() == 2 and z_text.size(0) == 1:
            z_text = z_text.squeeze(0)
        z_proj = self.text_proj(z_text)
        z_node = z_proj.unsqueeze(0).expand(x.size(0), -1)
        x = self.fuse(torch.cat([x, z_node], dim=1))
        return self.rgcn(x, edge_index, edge_type)


class TextConditionedRGCNEncoderFiLM(nn.Module):
    def __init__(self, hidden_dim, num_relations, text_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.text_gamma = nn.Linear(text_dim, hidden_dim)
        self.text_beta = nn.Linear(text_dim, hidden_dim)
        self.rgcn = RGCNEncoder(hidden_dim, num_relations, num_layers, dropout)

    def forward(self, x, edge_index, edge_type, z_text):
        if z_text.dim() == 2 and z_text.size(0) == 1:
            z_text = z_text.squeeze(0)
        gamma = self.text_gamma(z_text)
        beta = self.text_beta(z_text)
        gamma = gamma.unsqueeze(0).expand(x.size(0), -1)
        beta = beta.unsqueeze(0).expand(x.size(0), -1)
        x = x * (1.0 + gamma) + beta
        return self.rgcn(x, edge_index, edge_type)
