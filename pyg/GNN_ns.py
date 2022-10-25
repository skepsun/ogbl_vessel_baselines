import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.utils import to_undirected, add_self_loops

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from logger import Logger
from gcnconv import GCNConv


class PositiveLinkNeighborSampler(NeighborSampler):
    def __init__(self, edge_index, sizes, num_nodes=None, **kwargs):
        edge_idx = torch.arange(edge_index.size(1))
        super(PositiveLinkNeighborSampler,
              self).__init__(edge_index, sizes, edge_idx, num_nodes, **kwargs)

    def sample(self, edge_idx):
        if not isinstance(edge_idx, torch.Tensor):
            edge_idx = torch.tensor(edge_idx)
        row, col, _ = self.adj_t.coo()
        batch = torch.cat([row[edge_idx], col[edge_idx]], dim=0)
        return super(PositiveLinkNeighborSampler, self).sample(batch)


class NegativeLinkNeighborSampler(NeighborSampler):
    def __init__(self, edge_index, sizes, num_nodes=None, **kwargs):
        edge_idx = torch.arange(edge_index.size(1))
        super(NegativeLinkNeighborSampler,
              self).__init__(edge_index, sizes, edge_idx, num_nodes, **kwargs)

    def sample(self, edge_idx):
        num_nodes = self.adj_t.sparse_size(0)
        batch = torch.randint(0, num_nodes, (2 * len(edge_idx), ),
                              dtype=torch.long)
        return super(NegativeLinkNeighborSampler, self).sample(batch)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, add_self_loops=True, improved=True))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, add_self_loops=True, improved=True))
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads,
                 dropout):
        super(GAT, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads, concat=True))
        self.convs.append(GATConv(hidden_channels, out_channels, heads, concat=False))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, method='DOT'):
        super(LinkPredictor, self).__init__()
        self.method = method
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        if self.method == 'DOT':
            # For end-to-end SEAL model, the final score for the subgraph induced by two nodes
            # is calculated based on pooling.
            x = x_i * x_j
        elif self.method == 'COS':
            x = torch.sum(x_i * x_j, dim=-1) / \
                torch.sqrt(torch.sum(x_i * x_i, dim=-1) * torch.sum(x_j * x_j, dim=-1)).clamp(min=1-9)
            return torch.sigmoid(x)
        elif self.method == 'SUM':
            x = x_i + x_j
        elif self.method == 'DIFF':
            # The difference between two nodes' coordinate features seems more appropriate.
            # The risk is when x_i and x_j are swapped, the output is opposite.
            x = (x_i - x_j).abs()

        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = F.relu(x)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(model, predictor, x, pos_loader, neg_loader, optimizer, device):
    model.train()
    predictor.train()

    pbar = tqdm(range(len(pos_loader)))
    pbar.set_description('Training')

    total_loss = total_examples = 0
    for i, (pos_data, neg_data) in enumerate(zip(pos_loader, neg_loader)):
        optimizer.zero_grad()

        batch_size, n_id, adjs = pos_data
        adjs = [adj.to(device) for adj in adjs]
        h = model(x[n_id], adjs)
        h_src, h_dst = h.chunk(2, dim=0)
        pos_out = predictor(h_src, h_dst)
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        batch_size, n_id, adjs = neg_data
        adjs = [adj.to(device) for adj in adjs]
        h = model(x[n_id], adjs)
        h_src, h_dst = h.chunk(2, dim=0)
        neg_out = predictor(h_src, h_dst)
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        num_examples = h_src.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

        pbar.update(1)


    pbar.close()

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, x, subgraph_loader, split_edge, evaluator,
         batch_size, device):
    model.eval()
    predictor.eval()

    h = model.inference(x, subgraph_loader, device).to(device)

    pos_train_edge = split_edge['train']['edge'].to(h.device)
    neg_train_edge = split_edge['train']['edge_neg'].to(h.device)
    pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    neg_train_preds = []
    for perm in DataLoader(range(neg_train_edge.size(0)), batch_size):
        edge = neg_train_edge[perm].t()
        neg_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_train_pred = torch.cat(neg_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    train_rocauc = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_train_pred,
        })[f'rocauc']

    valid_rocauc = evaluator.eval({
        'y_pred_pos': pos_valid_pred,
        'y_pred_neg': neg_valid_pred,
        })[f'rocauc']

    test_rocauc = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'rocauc']

    return train_rocauc, valid_rocauc, test_rocauc


def main():
    parser = argparse.ArgumentParser(description='OGBL-Vessel (NS)')
    parser.add_argument('--root', type=str, default='/mnt/ssd/ssd/dataset')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--node_feat_process', type=str, default='node_normalize')
    parser.add_argument('--add_self_loops', action='store_true')
    parser.add_argument('--model', type=str, default='gcn')
    parser.add_argument('--predictor', type=str, default='DOT')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=1024 * 64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-vessel', root=args.root)
    split_edge = dataset.get_edge_split()
    data = dataset[0]
    if args.node_feat_process == 'node_normalize':
        # normalize x,y,z coordinates 
        data.x = torch.nn.functional.normalize(data.x, dim=0)
    elif args.node_feat_process == 'channel_normalize':
        data.x = torch.nn.functional.normalize(data.x, dim=1)
    elif args.node_feat_process == 'max-min':
        data.x = (data.x - data.x.min(dim=0)[0]) / (data.x.max(0)[0] - data.x.min(0)[0] + 1e-9)
    elif args.node_feat_process == 'z-score':
        data.x = (data.x - data.x.mean(0)) / (data.x.std(0) + 1e-9)
    elif args.node_feat_process == 'log':
        data.x = data.x.abs().clamp(min=1e-9).log() * data.x / data.x.abs().clamp(min=1e-9)
    elif args.node_feat_process == 'none':
        pass
    else:
        raise(Exception(f'The preprocessing method of node features: {args.node_feat_process} has not been implemented'))


    data.x = data.x.to(torch.float)
    if args.use_node_embedding:
        # data.x = torch.cat([data.x, torch.load('embedding.pt')], dim=-1)
        data.x = torch.load('../embedding.pt')
    x = data.x.to(device)

    if args.add_self_loops:
        data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)

    pos_loader = PositiveLinkNeighborSampler(data.edge_index, sizes=[10, 5],
                                             num_nodes=x.size(0),
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.num_workers)

    neg_loader = NegativeLinkNeighborSampler(data.edge_index, sizes=[10, 5],
                                             num_nodes=x.size(0),
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers)

    subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                      batch_size=64*1024, shuffle=False,
                                      num_workers=args.num_workers)

    if args.model == 'gcn':
        model = GCN(x.size(-1), args.hidden_channels, args.hidden_channels,
                    args.num_layers, args.dropout).to(device)
    elif args.model == 'sage':
        model = SAGE(x.size(-1), args.hidden_channels, args.hidden_channels,
                    args.num_layers, args.dropout).to(device)
    elif args.model == 'gat':
        model = GAT(x.size(-1), args.hidden_channels, args.hidden_channels,
                    args.num_layers, args.num_heads, args.dropout).to(device)

    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout, method=args.predictor).to(device)

    evaluator = Evaluator(name='ogbl-vessel')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, x, pos_loader, neg_loader,
                         optimizer, device)
            print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}')

            if epoch % args.eval_steps == 0:
                result = test(model, predictor, x, subgraph_loader, split_edge,
                              evaluator, batch_size=64 * 1024, device=device)
                logger.add_result(run, result)

                train_mrr, valid_mrr, test_mrr = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {train_mrr:.4f}, '
                      f'Valid: {valid_mrr:.4f}, '
                      f'Test: {test_mrr:.4f}')

        print('Neighborsampling')
        logger.print_statistics(run)
    print('Neighborsampling')
    logger.print_statistics()


if __name__ == "__main__":
    main()
