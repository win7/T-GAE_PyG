import torch
import torch.nn as nn

from torch_geometric.nn import GINConv, GINEConv, global_mean_pool

# Encoder using GINConv from torch_geometric
class TGAE_Encoder_GIN(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
		super().__init__()
		hidden_layers = n_layers - 2
		self.in_proj = torch.nn.Linear(input_dim, hidden_dim[0])
		self.convs = torch.nn.ModuleList()
		
		for i in range(hidden_layers):
			mlp = nn.Sequential(
				nn.Linear(input_dim + hidden_dim[i], 2 * hidden_dim[i+1]),
				nn.LayerNorm(2 * hidden_dim[i+1]),
				nn.LeakyReLU(0.1),
				nn.Linear(2 * hidden_dim[i+1], 2 * hidden_dim[i+1]),
				nn.LeakyReLU(0.1),
				nn.Linear(2 * hidden_dim[i+1], hidden_dim[i+1])
			)
			""" mlp = torch.nn.Sequential(
                torch.nn.Linear(input_dim + hidden_dim[i], hidden_dim[i + 1]),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim[i + 1], hidden_dim[i + 1]),
            ) """
			self.convs.append(GINConv(mlp, eps=0.0, train_eps=False))
			# ResidualGINLayer(dims[i], dims[i + 1], alpha=alpha)
		self.out_proj = torch.nn.Linear(sum(hidden_dim), output_dim)
	
	def forward(self, x, edge_index):
		initial_x = x.clone()
		x = self.in_proj(x)
		hidden_states = [x]

		for layer in self.convs:
			x_cat = torch.cat([initial_x, x], dim=1)
			x = layer(x_cat, edge_index)
			hidden_states.append(x)

		x = torch.cat(hidden_states, dim=1)
		x = self.out_proj(x)
		return x

class TGAE_GIN(torch.nn.Module):
	def __init__(self, num_hidden_layers, input_dim, hidden_dim, output_dim):
		super().__init__()
		self.encoder = TGAE_Encoder_GIN(input_dim, hidden_dim, output_dim, num_hidden_layers + 2)

	def forward(self, x, edge_index):
		z = self.encoder(x, edge_index)
		return z

# Encoder using GINEConv from torch_geometric
class TGAE_Encoder_GINE(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, n_layers, edge_dim):
		super().__init__()

		hidden_layers = n_layers - 2
		self.in_proj = nn.Linear(input_dim, hidden_dim[0])
		self.convs = nn.ModuleList()
		
		for i in range(hidden_layers):
			mlp = nn.Sequential(
				nn.Linear(input_dim + hidden_dim[i], 2 * hidden_dim[i+1]),
				nn.LayerNorm(2 * hidden_dim[i+1]),
				nn.LeakyReLU(0.1),
				nn.Linear(2 * hidden_dim[i+1], 2 * hidden_dim[i+1]),
				nn.LeakyReLU(0.1),
				nn.Linear(2 * hidden_dim[i+1], hidden_dim[i+1])
			)
			""" mlp = torch.nn.Sequential(
                torch.nn.Linear(input_dim + hidden_dim[i], hidden_dim[i + 1]),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim[i + 1], hidden_dim[i + 1]),
            ) """
			self.convs.append(GINEConv(mlp, eps=0.0, train_eps=False, edge_dim=edge_dim))
		self.out_proj = nn.Linear(sum(hidden_dim), output_dim)

	def forward(self, x, edge_index, edge_attr):
		initial_x = x.clone()
		x = self.in_proj(x)
		hidden_states = [x]

		for layer in self.convs:
			x_cat = torch.cat([initial_x, x], dim=1)
			x = layer(x_cat, edge_index, edge_attr)
			hidden_states.append(x)
		
		x = torch.cat(hidden_states, dim=1)
		x = self.out_proj(x)
		return x

class TGAE_GINE(nn.Module):
	def __init__(self, num_hidden_layers, input_dim, hidden_dim, output_dim, edge_dim):
		super().__init__()

		self.encoder = TGAE_Encoder_GINE(input_dim, hidden_dim, output_dim, num_hidden_layers + 2, edge_dim)

	def forward(self, x, edge_index, edge_attr):
		z = self.encoder(x, edge_index, edge_attr)
		return z

""" class ResidualGINLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, alpha=0.2):
        super().__init__()
        self.alpha = alpha

        mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(out_dim, out_dim),
        )

        self.conv = GINConv(mlp, train_eps=False)

        self.proj = None
        if in_dim != out_dim:
            self.proj = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        identity = x if self.proj is None else self.proj(x)
        out = self.conv(x, edge_index)
        return identity + self.alpha * out """