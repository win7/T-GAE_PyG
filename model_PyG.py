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

class TGAE_Encoder_GINE(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
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
			self.convs.append(GINEConv(mlp, edge_dim=1)) # Change edge_attribute
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
	def __init__(self, num_hidden_layers, input_dim, hidden_dim, output_dim):
		super().__init__()

		self.encoder = TGAE_Encoder_GINE(input_dim, hidden_dim, output_dim, num_hidden_layers + 2)

	def forward(self, x, edge_index, edge_attr):
		z = self.encoder(x, edge_index, edge_attr)
		return z