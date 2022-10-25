from torch_geometric.nn import GCNConv, SAGEConv, GATConv, TransformerConv, GINConv, TAGConv
from torch_geometric.utils import dropout_adj
import torch
import torch.nn.functional as F
from utils import drop_edge

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, edge_drop=0., num_heads=1, model_type='gcn', residual=True, bn=False):
        super(GNN, self).__init__()
        self.dropout = dropout
        self.edge_drop = edge_drop
        self.num_heads = num_heads if model_type == 'gat' else 1
        self.model_type = model_type
        self.residual = residual
        self.convs = torch.nn.ModuleList()
        if bn:
            self.bns = torch.nn.ModuleList()
        else:
            self.bns = None

        if num_layers == 1:
            self.convs.append(self.generate_gnn_layer(in_channels, out_channels, last=True))
        else:
            for i in range(num_layers):
                in_channels_ = in_channels if i == 0 else hidden_channels * self.num_heads
                out_channels_ = out_channels if i == (num_layers - 1) else hidden_channels
                self.convs.append(
                    self.generate_gnn_layer(in_channels_, out_channels_, last=i == (num_layers-1)))
                if bn and i < num_layers - 1:
                    self.bns.append(torch.nn.BatchNorm1d(out_channels_ * self.num_heads))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.bns is not None:
            for bn in self.bns:
                bn.reset_parameters()
    
    def generate_gnn_layer(self, in_channels, out_channels, last=False):
        if self.model_type == 'gcn':
            conv = GCNConv(in_channels, out_channels, normalize=True, improved=False)
        elif self.model_type == 'sage':
            conv = SAGEConv(in_channels, out_channels)
        elif self.model_type == 'gat':
            conv = GATConv(in_channels, out_channels, self.num_heads, concat=not last)
        elif self.model_type == 'transformer':
            conv = TransformerConv(in_channels, out_channels, self.num_heads, concat=not last)
        elif self.model_type == 'gin':
            conv = GINConv(torch.nn.Sequential(
                            torch.nn.Linear(in_channels, out_channels), 
                            torch.nn.ReLU(), torch.nn.Dropout(self.dropout),
                            torch.nn.Linear(out_channels, out_channels)),
                            0.1)
        elif self.model_type == 'tag':
            conv = TAGConv(in_channels, out_channels, 3)
        elif self.model_type == 'mlp':
            conv = torch.nn.Linear(in_channels, out_channels)
        return conv

    def forward(self, x, edge_index):
        if len(self.convs) == 0:
            return x
        edge_index_ = edge_index[:, drop_edge(edge_index, self.edge_drop, training=self.training)]
        for i, conv in enumerate(self.convs[:-1]):
            if self.model_type == 'mlp':
                x = conv(x)
            else:
                x = conv(x, edge_index_)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.bns is not None:
                x = self.bns[i](x)

        if self.model_type == 'mlp':
            x = self.convs[-1](x)
        else:
            x = self.convs[-1](x, edge_index_)
        if len(self.convs) == 1:
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, method='DOT'):
        super(LinkPredictor, self).__init__()
        self.method = method
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels if method != 'CONCAT' else 2 * in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        # For methods that may rely on the order of vectors, we sort vectors by their last elements.
        mask = x_i[:,-1] < x_j[:,-1]
        x_i[mask], x_j[mask] = x_j[mask], x_i[mask]
        if self.method == 'DOT':
            # Element-wise multiplication, or inner dot with linears.
            x = x_i * x_j
        elif self.method == 'COS':
            x = torch.sum(x_i * x_j, dim=-1) / \
                torch.sqrt(torch.sum(x_i * x_i, dim=-1) * torch.sum(x_j * x_j, dim=-1)).clamp(min=1-9)
            return torch.sigmoid(x)
        elif self.method == 'SUM':
            x = x_i + x_j
        elif self.method == 'DIFF':
            # The difference between two nodes' coordinate features seems more appropriate.
            x = (x_i - x_j)
        elif self.method == 'MAX':
            x = x_i.clone()
            x[x_i < x_j] = x_j[x_i<x_j]
        elif self.method == 'MEAN':
            x = (x_i + x_j) / 2
        elif self.method == 'CONCAT':
            x = torch.cat([x_i, x_j], dim=1)

        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = F.relu(x)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)