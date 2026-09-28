# This version replicates original version
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm

class OriginalConv(MessagePassing):
    """Replica exacta de la GINConv original: ReLU(Linear(X + A_norm @ X))"""
    def __init__(self, input_dim, output_dim):
        super().__init__(aggr="add")
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, edge_index, edge_weight):
        agg = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = self.linear(x + agg)
        return F.relu(out)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j


class TGAE_Encoder_Original(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        hidden_layers = n_layers - 2
        self.in_proj = nn.Linear(input_dim, hidden_dim[0])
        self.convs = nn.ModuleList()
        for i in range(hidden_layers):
            self.convs.append(OriginalConv(input_dim + hidden_dim[i], hidden_dim[i + 1]))
        self.out_proj = nn.Linear(sum(hidden_dim), output_dim)

    def forward(self, x, edge_index, edge_weight):
        initial_x = x.clone()
        x = self.in_proj(x)
        hidden_states = [x]
        for layer in self.convs:
            x_cat = torch.cat([initial_x, x], dim=1)
            x = layer(x_cat, edge_index, edge_weight)
            hidden_states.append(x)
        x = torch.cat(hidden_states, dim=1)
        x = self.out_proj(x)
        return x


class TGAE_Original(nn.Module):
    def __init__(self, num_hidden_layers, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = TGAE_Encoder_Original(input_dim, hidden_dim, output_dim, num_hidden_layers + 2)

    def forward(self, x, edge_index, edge_weight):
        return self.encoder(x, edge_index, edge_weight)