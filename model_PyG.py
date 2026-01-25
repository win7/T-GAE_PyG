import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINConv, global_mean_pool

# Encoder usando GINConv de torch_geometric
class TGAE_Encoder(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
		super().__init__()
		hidden_layers = n_layers-2
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
			self.convs.append(GINConv(mlp))
		self.out_proj = torch.nn.Linear(sum(hidden_dim), output_dim)
	
	def forward(self, x, edge_index):
		# x: [num_nodes, input_dim]
		# edge_index: [2, num_edges]
		x0 = x.clone()

		# Primera proyección lineal
		x = self.in_proj(x)
		states = [x]

		# Aplicar cada capa GINConv
		for conv in self.convs:
			x= torch.cat([x0, x], dim=1)
			x = conv(x, edge_index)
			# x = F.relu(x)
			states.append(x)

		# Concatenar todos los estados, luego proyectar
		h = torch.cat(states, dim=1)
		out = self.out_proj(h)
		return out

# Modelo completo TGAE
class TGAE(torch.nn.Module):
	def __init__(self, num_hidden_layers, input_dim, hidden_dim, output_dim):
		super().__init__()
		self.encoder = TGAE_Encoder(input_dim, hidden_dim, output_dim, num_hidden_layers+2)

	def forward(self, x, edge_index):
		z = self.encoder(x, edge_index)
		return z

