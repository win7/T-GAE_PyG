"""Encoders T-GAE en PyTorch Geometric.

Esqueleto común (idéntico al de model.py original):
    in_proj -> [conv_i sobre cat(x_inicial, h)] x L -> cat(todos los estados) -> out_proj

Variantes (conv_type):
    "gin"     GINConv de PyG (suma cruda + MLP). Lo que el paper dice usar.
    "gine"    GINEConv con atributos de arista (extensión propia).
    "gcn"     GCNConv estándar: ReLU(Â X W).
    "gcnres"  Réplica de la capa de model.py original: ReLU(Linear(X + Â X)).
    "gatv2"   GATv2Conv multi-cabeza.
    "gatv2e"  GATv2Conv con atributos de arista (análogo a GINE).

Convenciones de uso:
    - edge_weight / edge_attr se pasan SIEMPRE por nombre.
    - "gcnres": si edge_weight es None, el encoder calcula una sola vez
      Â = D^-1/2 (A+I) D^-1/2 con gcn_norm. Si se pasa edge_weight, debe venir
      ya normalizado (salida de gcn_norm).
    - "gcn": edge_weight, si se pasa, debe ser el peso CRUDO (GCNConv normaliza).
    - No usar cached=True en GCNConv: el mismo encoder procesa dos grafos.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, GINConv, GINEConv, GCNConv, GATv2Conv
from torch_geometric.nn.conv.gcn_conv import gcn_norm

CONV_TYPES = ("gin", "gine", "gcn", "gcnres", "gatv2", "gatv2e")
NEEDS_EDGE_ATTR = ("gine", "gatv2e")

class GCNResConv(MessagePassing):
    """ReLU(Linear(X + Â X)) con Â ya normalizada (edge_weight de gcn_norm).

    Es la capa que model.py original llama "GINConv": su agregación es la de
    GCN (normalizada por grado), no la suma cruda de GIN.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__(aggr="add")
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, edge_weight):
        agg = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return F.relu(self.linear(x + agg))

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j


def _gin_mlp(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, 2 * out_dim),
        nn.LayerNorm(2 * out_dim),
        nn.LeakyReLU(0.1),
        nn.Linear(2 * out_dim, 2 * out_dim),
        nn.LeakyReLU(0.1),
        nn.Linear(2 * out_dim, out_dim),
    )


class TGAE_Encoder(nn.Module):
    def __init__(self, conv_type, input_dim, hidden_dim, output_dim,
                 num_hidden_layers, edge_dim=None, heads=4):
        super().__init__()
        if conv_type not in CONV_TYPES:
            raise ValueError(f"conv_type debe ser uno de {CONV_TYPES}, recibido '{conv_type}'")
        # out_proj usa sum(hidden_dim): la longitud debe ser exactamente L + 1.
        if len(hidden_dim) != num_hidden_layers + 1:
            raise ValueError(
                f"len(hidden_dim)={len(hidden_dim)} pero se esperaba "
                f"num_hidden_layers + 1 = {num_hidden_layers + 1}"
            )
        if conv_type in NEEDS_EDGE_ATTR and edge_dim is None:
            raise ValueError(f"conv_type='{conv_type}' requiere edge_dim")
        if conv_type.startswith("gatv2") and any(h % heads for h in hidden_dim[1:]):
            raise ValueError(f"Todos los hidden_dim[1:] deben ser divisibles por heads={heads}")

        self.conv_type = conv_type
        self.in_proj = nn.Linear(input_dim, hidden_dim[0])
        self.convs = nn.ModuleList(
            self._make_conv(conv_type, input_dim + hidden_dim[i], hidden_dim[i + 1], edge_dim, heads)
            for i in range(num_hidden_layers)
        )
        self.out_proj = nn.Linear(sum(hidden_dim), output_dim)

    @staticmethod
    def _make_conv(conv_type, in_dim, out_dim, edge_dim, heads):
        if conv_type == "gin":
            return GINConv(_gin_mlp(in_dim, out_dim), eps=0.0, train_eps=False)
        if conv_type == "gine":
            return GINEConv(_gin_mlp(in_dim, out_dim), eps=0.0, train_eps=False, edge_dim=edge_dim)
        if conv_type == "gcn":
            return GCNConv(in_dim, out_dim, add_self_loops=True, normalize=True)
        if conv_type == "gcnres":
            return GCNResConv(in_dim, out_dim)
        if conv_type == "gatv2":
            return GATv2Conv(in_dim, out_dim // heads, heads=heads, concat=True,
                             add_self_loops=True)
        return GATv2Conv(in_dim, out_dim // heads, heads=heads, concat=True,
                         edge_dim=edge_dim, add_self_loops=True, fill_value="mean")

    def _act(self, h):
        if self.conv_type == "gcn":
            # return F.relu(h)          # GCNConv no incluye activación
            return F.tanh(h)            # Me
        if self.conv_type.startswith("gatv2"):
            return F.elu(h)           # GATv2Conv no incluye activación
        return h                      # gin/gine: igual que antes; gcnres ya aplica ReLU

    def forward(self, x, edge_index, *, edge_weight=None, edge_attr=None):
        if self.conv_type == "gcnres" and edge_weight is None:
            edge_index, edge_weight = gcn_norm(edge_index, num_nodes=x.size(0),
                                               add_self_loops=True)
        initial_x = x                 # no hay operaciones in-place: clone() no es necesario
        h = self.in_proj(x)
        hidden_states = [h]
        for conv in self.convs:
            h_in = torch.cat([initial_x, h], dim=1)
            if self.conv_type in NEEDS_EDGE_ATTR:
                h = conv(h_in, edge_index, edge_attr)
            elif self.conv_type in ("gcn", "gcnres"):
                h = conv(h_in, edge_index, edge_weight)
            else:
                h = conv(h_in, edge_index)
            h = self._act(h)
            hidden_states.append(h)
        return self.out_proj(torch.cat(hidden_states, dim=1))


class TGAE(nn.Module):
    def __init__(self, conv_type, num_hidden_layers, input_dim, hidden_dim, output_dim,
                 edge_dim=None, heads=4):
        super().__init__()
        self.encoder = TGAE_Encoder(conv_type, input_dim, hidden_dim, output_dim,
                                    num_hidden_layers, edge_dim=edge_dim, heads=heads)

    def forward(self, x, edge_index, *, edge_weight=None, edge_attr=None):
        return self.encoder(x, edge_index, edge_weight=edge_weight, edge_attr=edge_attr)

    @torch.no_grad()
    def embed(self, x, edge_index, *, edge_weight=None, edge_attr=None, normalize=False):
        """Embeddings para alineamiento. normalize=True aplica L2 (equivale a usar
        distancia coseno al recuperar vecinos); el entrenamiento no se modifica."""
        was_training = self.training
        self.eval()
        z = self(x, edge_index, edge_weight=edge_weight, edge_attr=edge_attr)
        self.train(was_training)
        return F.normalize(z, p=2, dim=1) if normalize else z

    @staticmethod
    def decode(z):
        """Decoder no paramétrico del paper: sigmoide del producto interno (denso N x N)."""
        return torch.sigmoid(z @ z.t())


# Compatibilidad con los nombres anteriores -------------------------------------
def TGAE_GIN(num_hidden_layers, input_dim, hidden_dim, output_dim):
    return TGAE("gin", num_hidden_layers, input_dim, hidden_dim, output_dim)


def TGAE_GINE(num_hidden_layers, input_dim, hidden_dim, output_dim, edge_dim):
    return TGAE("gine", num_hidden_layers, input_dim, hidden_dim, output_dim, edge_dim=edge_dim)


def TGAE_GCN(num_hidden_layers, input_dim, hidden_dim, output_dim):
    return TGAE("gcn", num_hidden_layers, input_dim, hidden_dim, output_dim)


def TGAE_GCNRes(num_hidden_layers, input_dim, hidden_dim, output_dim):
    return TGAE("gcnres", num_hidden_layers, input_dim, hidden_dim, output_dim)


def TGAE_GATv2(num_hidden_layers, input_dim, hidden_dim, output_dim, heads=4):
    return TGAE("gatv2", num_hidden_layers, input_dim, hidden_dim, output_dim, heads=heads)


def TGAE_GATv2E(num_hidden_layers, input_dim, hidden_dim, output_dim, edge_dim, heads=4):
    return TGAE("gatv2e", num_hidden_layers, input_dim, hidden_dim, output_dim,
                edge_dim=edge_dim, heads=heads)
