import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl import function as fn
from dgl.base import DGLError
from dgl.nn.pytorch.conv import GraphConv, SAGEConv, GATConv
from dgl.transforms import DropEdge
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
from global_attention import LowRankAttention


class MLP(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        n_layers,
        n_hidden,
        activation,
        dropout,
        input_drop,
        bn=False,
    ):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.input_drop = input_drop
        self.activation = activation
        self.lins = nn.ModuleList()
        if bn and n_layers > 1:
            self.bns = nn.ModuleList()
        else:
            self.bns = None
        if n_layers == 1:
            self.lins.append(nn.Linear(in_feats, out_feats))
        else:
            for i in range(n_layers):
                in_feats_ = in_feats if i == 0 else n_hidden
                out_feats_ = out_feats if i == n_layers - 1 else n_hidden
                self.lins.append(nn.Linear(in_feats_, out_feats_))
                if bn and i < n_layers - 1:
                    self.bns.append(nn.BatchNorm1d(out_feats_))
        self.reset_parameters()
    
    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        if self.bns is not None:
            for bn in self.bns:
                bn.reset_parameters()
    
    def forward(self, g):
        x = g.ndata['feat']
        h = F.dropout(x, self.input_drop, training=self.training)
        for i in range(len(self.lins)):
            h = self.lins[i](h)
            if i < len(self.lins) - 1 or len(self.lins) == 1:
                if self.bns is not None:
                    h = self.bns[i](h)
                h = self.activation(h)
                h = F.dropout(h, self.dropout, training=self.training)
        return h
        

class GCN(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        n_layers,
        n_hidden,
        activation,
        dropout,
        input_drop,
        allow_zero_in_degree=False,
        norm='both',
        bn=False,
        global_attn=False,
        k=10,
    ):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.input_drop = input_drop
        self.activation = activation
        self.convs = nn.ModuleList()
        if bn and n_layers > 1:
            self.bns = nn.ModuleList()
        else:
            self.bns = None
        if global_attn:
            self.global_attn = nn.ModuleList()
            self.lins = nn.ModuleList()
        else:
            self.global_attn = None
            self.lins = None
        if n_layers == 1:
            self.convs.append(GraphConv(in_feats, out_feats, 
                norm=norm, allow_zero_in_degree=allow_zero_in_degree))
            if global_attn:
                self.global_attn.append(LowRankAttention(k, in_feats, dropout))
                self.lins.append(nn.Sequential(nn.Linear(2 * k + out_feats + in_feats, out_feats), nn.ReLU()))
        else:
            for i in range(n_layers):
                in_feats_ = in_feats if i == 0 else n_hidden
                out_feats_ = out_feats if i == n_layers - 1 else n_hidden
                self.convs.append(GraphConv(in_feats_, out_feats_,
                    norm=norm, allow_zero_in_degree=allow_zero_in_degree))
                if global_attn:
                    self.global_attn.append(LowRankAttention(k, in_feats_, dropout))
                    self.lins.append(nn.Sequential(nn.Linear(2 * k + in_feats_ + out_feats_, out_feats_), nn.ReLU()))
                if bn and i < n_layers - 1:
                    self.bns.append(nn.BatchNorm1d(out_feats_))
        self.reset_parameters()
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.bns is not None:
            for bn in self.bns:
                bn.reset_parameters()
        # if self.global_attn is not None:
        #     for attn in self.global_attn:
        #         attn.reset_parameters()
        # if self.lins is not None:
        #     for lin in self.lins:
        #         lin.reset_parameters()
    
    def forward(self, g):
        x = g.ndata['feat']
        h = F.dropout(x, self.input_drop, training=self.training)
        for i in range(len(self.convs)):
            h_last = h
            h = self.convs[i](g, h)
            
            if i < len(self.convs) - 1 or len(self.convs) == 1:

                if self.bns is not None:
                    h = self.bns[i](h)
                h = self.activation(h)
                h = F.dropout(h, self.dropout, training=self.training)

            if self.global_attn is not None:
                h_global = self.global_attn[i](h_last)
                h = self.lins[i](torch.cat([h, h_global, h_last], dim=-1))
                if i < len(self.convs) - 1 or len(self.convs) == 1:
                    h = F.relu(h)
        return h

class SAGE(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        n_layers,
        n_hidden,
        activation,
        dropout,
        input_drop,
        edge_drop,
        allow_zero_in_degree=False,
        norm='both',
        bn=False,
        global_attn=False,
        k=10,
    ):
        super(SAGE, self).__init__()
        self.dropout = dropout
        self.input_drop = input_drop
        self.activation = activation
        self.convs = nn.ModuleList()
        if bn and n_layers > 1:
            self.bns = nn.ModuleList()
        else:
            self.bns = None
        if global_attn:
            self.global_attn = nn.ModuleList()
            self.lins = nn.ModuleList()
        else:
            self.global_attn = None
            self.lins = None
        if n_layers == 1:
            self.convs.append(SAGEConv(in_feats, out_feats, 'mean'))
            if global_attn:
                self.global_attn.append(LowRankAttention(k, in_feats, dropout))
                self.lins.append(nn.Sequential(nn.Linear(2 * k + out_feats + in_feats, out_feats), nn.ReLU()))
        else:
            for i in range(n_layers):
                in_feats_ = in_feats if i == 0 else n_hidden
                out_feats_ = out_feats if i == n_layers - 1 else n_hidden
                self.convs.append(SAGEConv(in_feats_, out_feats_, 'mean'))
                if global_attn:
                    self.global_attn.append(LowRankAttention(k, in_feats_, dropout))
                    self.lins.append(nn.Sequential(nn.Linear(2 * k + in_feats_ + out_feats_, out_feats_), nn.ReLU()))
                if bn and i < n_layers - 1:
                    self.bns.append(nn.BatchNorm1d(out_feats_))
        self.reset_parameters()
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.bns is not None:
            for bn in self.bns:
                bn.reset_parameters()
    
    def forward(self, g):
        x = g.ndata['feat']
        h = F.dropout(x, self.input_drop, training=self.training)
        for i in range(len(self.convs)):
            h_last = h.clone()
            h = self.convs[i](g, h)
            if i < len(self.convs) - 1 or len(self.convs) == 1:
                if self.bns is not None:
                    h = self.bns[i](h)
                h = self.activation(h)
                h = F.dropout(h, self.dropout, training=self.training)
            if self.global_attn is not None:
                h_global = self.global_attn[i](h_last)
                h = self.lins[i](torch.cat([h, h_global, h_last], dim=-1))
                if i < len(self.convs) - 1 or len(self.convs) == 1:
                    h = F.relu(h)
        return h

# class GATConv(nn.Module):
#     def __init__(
#         self,
#         node_feats,
#         out_feats,
#         n_heads=1,
#         attn_drop=0.0,
#         edge_drop=0.0,
#         negative_slope=0.2,
#         residual=False,
#         activation=None,
#         use_attn_dst=True,
#         allow_zero_in_degree=True,
#         norm="none",
#     ):
#         super(GATConv, self).__init__()
#         self._n_heads = n_heads
#         self._in_src_feats, self._in_dst_feats = expand_as_pair(node_feats)
#         self._out_feats = out_feats
#         self._allow_zero_in_degree = allow_zero_in_degree
#         self._norm = norm

#         # feat fc
#         self.src_fc = nn.Linear(self._in_src_feats, out_feats * n_heads, bias=False)
#         if residual:
#             self.dst_fc = nn.Linear(self._in_src_feats, out_feats * n_heads)
#             self.bias = None
#         else:
#             self.dst_fc = self.src_fc
#             self.bias = nn.Parameter(torch.FloatTensor(1, n_heads, out_feats))

#         # attn fc
#         self.attn_src_fc = nn.Linear(self._in_src_feats, n_heads, bias=False)
#         if use_attn_dst:
#             self.attn_dst_fc = nn.Linear(self._in_src_feats, n_heads, bias=False)
#         else:
#             self.attn_dst_fc = None
#         self.attn_edge = nn.Linear(self._out_feats, n_heads, bias=False)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.edge_drop = edge_drop
#         self.leaky_relu = nn.LeakyReLU(negative_slope, inplace=True)
#         self.activation = activation

#         self.reset_parameters()

#     def reset_parameters(self):
#         gain = nn.init.calculate_gain("relu")
#         nn.init.xavier_normal_(self.src_fc.weight, gain=gain)
#         if self.dst_fc is not None:
#             nn.init.xavier_normal_(self.dst_fc.weight, gain=gain)

#         nn.init.xavier_normal_(self.attn_src_fc.weight, gain=gain)
#         if self.attn_dst_fc is not None:
#             nn.init.xavier_normal_(self.attn_dst_fc.weight, gain=gain)
#         nn.init.xavier_normal_(self.attn_edge.weight, gain=gain)

#         if self.bias is not None:
#             nn.init.zeros_(self.bias)

#     def set_allow_zero_in_degree(self, set_value):
#         self._allow_zero_in_degree = set_value

#     def forward(self, graph, feat_src):
#         with graph.local_scope():
#             if not self._allow_zero_in_degree:
#                 if (graph.in_degrees() == 0).any():
#                     raise DGLError('There are 0-in-degree nodes in the graph, '
#                                    'output for those nodes will be invalid. '
#                                    'This is harmful for some applications, '
#                                    'causing silent performance regression. '
#                                    'Adding self-loop on the input graph by '
#                                    'calling `g = dgl.add_self_loop(g)` will resolve '
#                                    'the issue. Setting ``allow_zero_in_degree`` '
#                                    'to be `True` when constructing this module will '
#                                    'suppress the check and let the code run.')
                                   
#             if graph.is_block:
#                 feat_dst = feat_src[: graph.number_of_dst_nodes()]
#             else:
#                 feat_dst = feat_src

#             if self._norm == 'adj':
#                 degs = graph.in_degrees()
#                 # degs = graph.out_degrees().float().clamp(min=1)
#                 norm = torch.pow(degs, -0.5)
#                 shp = norm.shape + (1,) * (feat_src.dim() - 1)
#                 norm = torch.reshape(norm, shp)
#                 feat_src = feat_src * norm

#             feat_src_fc = self.src_fc(feat_src).view(-1, self._n_heads, self._out_feats)
#             feat_dst_fc = self.dst_fc(feat_dst).view(-1, self._n_heads, self._out_feats)
#             attn_src = self.attn_src_fc(feat_src).view(-1, self._n_heads, 1)

#             # NOTE: GAT paper uses "first concatenation then linear projection"
#             # to compute attention scores, while ours is "first projection then
#             # addition", the two approaches are mathematically equivalent:
#             # We decompose the weight vector a mentioned in the paper into
#             # [a_l || a_r], then
#             # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
#             # Our implementation is much efficient because we do not need to
#             # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
#             # addition could be optimized with DGL's built-in function u_add_v,
#             # which further speeds up computation and saves memory footprint.
#             graph.srcdata.update({"feat_src_fc": feat_src_fc, "attn_src": attn_src})
#             graph.dstdata.update({"_feat_dst_fc": feat_dst_fc})
#             if self.attn_dst_fc is not None:
#                 attn_dst = self.attn_dst_fc(feat_dst).view(-1, self._n_heads, 1)
#                 graph.dstdata.update({"attn_dst": attn_dst})
#                 graph.apply_edges(fn.u_add_v("attn_src", "attn_dst", "attn_node"))
#             else:
#                 graph.apply_edges(fn.copy_u("attn_src", "attn_node"))

#             e = graph.edata["attn_node"]
#             # graph.apply_edges(fn.u_mul_v("feat_src_fc", "_feat_dst_fc", "edge_fc"))
#             # e = self.attn_edge(graph.edata["edge_fc"])
#             e = self.leaky_relu(e)
            
#             if self.training and self.edge_drop > 0:
#                 perm = torch.randperm(graph.number_of_edges(), device=e.device)
#                 bound = int(graph.number_of_edges() * self.edge_drop)
#                 eids = perm[bound:]
                
#             else:
#                 eids = torch.arange(graph.number_of_edges(), device=e.device)
#             graph.edata["a"] = torch.zeros_like(e)
#             graph.edata["a"][eids] = self.attn_drop(edge_softmax(graph, e[eids], eids=eids))

#             # message passing
#             graph.update_all(fn.u_mul_e("feat_src_fc", "a", "m"), fn.sum("m", "feat_src_fc"))

#             rst = graph.dstdata["feat_src_fc"]

#             if self._norm == 'adj':
#                 degs = graph.in_degrees()
#                 norm = torch.pow(degs, 0.5)
#                 shp = norm.shape + (1,) * (feat_dst.dim())
#                 norm = torch.reshape(norm, shp)
#                 rst = rst * norm

#             # residual
#             if self.bias is None:
#                 rst += feat_dst_fc
#             else:
#                 rst += self.bias

#             # activation
#             if self.activation is not None:
#                 rst = self.activation(rst, inplace=True)

#             return rst


class GAT(nn.Module):
    def __init__(
        self,
        node_feats,
        n_classes,
        n_layers,
        n_heads,
        n_hidden,
        activation,
        dropout,
        input_drop,
        attn_drop,
        edge_drop,
        use_attn_dst=True,
        allow_zero_in_degree=False,
        norm="none",
        bn=False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()
        if bn:
            self.norms = nn.ModuleList()
        else:
            self.norms = None

        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else node_feats
            out_hidden = n_hidden
            # bias = i == n_layers - 1

            self.convs.append(
                GATConv(
                    in_hidden,
                    out_hidden,
                    n_heads,
                    attn_drop=attn_drop,
                    allow_zero_in_degree=allow_zero_in_degree,
                    residual=True,
                )
            )
            if bn and i < n_layers - 1:
                self.norms.append(nn.BatchNorm1d(out_hidden * n_heads))

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.norms is not None:
            for norm in self.norms:
                norm.reset_parameters()

    def forward(self, g):
        if not isinstance(g, list):
            subgraphs = [g] * self.n_layers
        else:
            subgraphs = g

        h = subgraphs[0].srcdata["feat"]
        h = self.input_drop(h)

        h_last = None

        for i in range(self.n_layers - 1):
            
            h = self.convs[i](subgraphs[i], h).flatten(1, -1)

            # if h_last is not None and h_last.size(-1) == h.size(-1):
            #     h += h_last[: h.shape[0], :]

            h_last = h
            if self.norms is not None:
                h = self.norms[i](h)
            h = self.activation(h, inplace=True)
            h = self.dropout(h)
        h = self.convs[-1](subgraphs[-1], h).mean(1)
        return h

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
