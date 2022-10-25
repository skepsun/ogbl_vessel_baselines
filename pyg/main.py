import argparse
from logging import raiseExceptions
import time

import torch
import torch_geometric.transforms as T
from ogb.linkproppred import Evaluator, PygLinkPropPredDataset
from torch.utils.data import DataLoader
from torch_geometric.utils import add_self_loops
from torch_sparse import SparseTensor
from logger import Logger
from models import GNN, LinkPredictor
from utils import count_parameters


def train(model, predictor, data, split_edge, optimizer, batch_size):
    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)

    neg_train_edge = split_edge['train']['edge_neg'].to(data.x.device) 

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):

        optimizer.zero_grad()

        h = model(data.x, data.edge_index)

        edge = pos_train_edge[perm].t()
        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        # pos_out = predictor(h[edge[1]], h[edge[0]])
        # pos_loss += -torch.log(pos_out + 1e-15).mean()

        # random element of previously sampled negative edges
        # negative samples are obtained by using spatial sampling criteria

        edge = neg_train_edge[perm].t()
        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        # neg_out = predictor(h[edge[1]], h[edge[0]])
        # neg_loss += -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples

@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()

    h = model(data.x, data.edge_index)

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
    parser = argparse.ArgumentParser(description='Simple OGBL-VESSEL Baselines.')
    # Common settings
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100) 
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.001) 
    # Specific tricks
    parser.add_argument('--use_node2vec_embedding', action='store_true')
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--node_feat_process', type=str, default='node_normalize')
    parser.add_argument('--directed_graph', action='store_true')
    parser.add_argument('--add_self_loops', action='store_true')
    # Diffusion trick
    parser.add_argument('--diffusion', action='store_true')
    parser.add_argument('--K', type=int, default=1)
    # Model selection
    parser.add_argument('--model', type=str, default='gcn')
    parser.add_argument('--predictor', type=str, default='DOT')
    # Model settings
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--edge_drop', type=float, default=0.0)
    parser.add_argument('--residual', action='store_true')
    parser.add_argument('--bn', action='store_true')
    
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset('ogbl-vessel', root='/mnt/ssd/ssd/dataset')
    data = dataset[0]

    split_edge = dataset.get_edge_split()

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

    
    if args.directed_graph:
        # Is a directed vessel between two bifurcation points informative?
        data.edge_index = split_edge['train']['edge'].t()
    if args.add_self_loops:
        data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)

    
    data.edge_attr = data.x[data.edge_index[0, :]] - data.x[data.edge_index[1, :]]
    print(data)
    
    data.x = data.x.to(torch.float)
    if args.use_node2vec_embedding:
        data.x = torch.cat([data.x, torch.load('../embedding.pt')], dim=-1)
        # data.x = torch.load('../embedding.pt')
    if args.use_node_embedding:
        emb = torch.nn.Embedding(data.num_nodes,
                             args.hidden_channels)
        data.x = torch.cat([data.x, emb.weight], dim=1)
        # data.x = emb.weight
    data = data.to(device)

    data.adj_t = SparseTensor(row=data.edge_index[1], col=data.edge_index[0], sparse_sizes=[data.num_nodes, data.num_nodes])
    if args.model == 'gcn':
        data.adj_t = data.adj_t.set_diag()
        deg = data.adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        data.adj_t = deg_inv_sqrt.view(-1, 1) * data.adj_t * deg_inv_sqrt.view(1, -1)
    
    else:
        if args.add_self_loops:
            data.adj_t = data.adj_t.set_diag()
        deg = data.adj_t.sum(dim=1).to(torch.float)
        inv_deg = deg.pow(-1)
        
        inv_deg[inv_deg == float('inf')] = 0
        data.adj_t = inv_deg.view(-1, 1) * data.adj_t

    if args.diffusion:
        # precompute multi-hop features
        xs = [data.x]
        for k in range(args.K):
            xs.append(data.adj_t.matmul(xs[-1]))
            xs[-1][deg==0] = xs[0][deg==0]
        data.xs = xs
        # data.x = xs[-1]
        # import pdb; pdb.set_trace()
        data.x = torch.cat(data.xs, dim=1)
    
    # Note that MLP is included in GNN module.
    model = GNN(data.num_features, args.hidden_channels,
                args.hidden_channels, args.num_layers,
                args.dropout,
                edge_drop=args.edge_drop, 
                num_heads=args.num_heads, 
                model_type=args.model,
                residual=args.residual,
                bn=args.bn).to(device)

    predictor = LinkPredictor(args.hidden_channels if args.num_layers > 0 else data.num_features, args.hidden_channels, 1,
                              args.num_layers, args.dropout, method=args.predictor).to(device)

    evaluator = Evaluator(name='ogbl-vessel')
    logger = Logger(args.runs, args)   

    for run in range(args.runs):
        
        model.reset_parameters()
        predictor.reset_parameters()
        params_count = count_parameters(model) + count_parameters(predictor)
        params = list(model.parameters()) + list(predictor.parameters())
        if args.use_node_embedding:
            torch.nn.init.xavier_uniform_(emb.weight)
            params_count += count_parameters(emb)
            params += list(emb.parameters())
        print(f'Params: {params_count}')
        optimizer = torch.optim.Adam(params, lr=args.lr)
        best_val = 0
        final_test = 0

        for epoch in range(1, 1 + args.epochs):
            tic = time.time()
            loss = train(model, predictor, data, split_edge, optimizer,
                         args.batch_size)
            toc = time.time()
            train_time = toc - tic

            if epoch % args.eval_steps == 0:
                tic = time.time()
                result = test(model, predictor, data, split_edge, evaluator,
                               args.batch_size)
                toc = time.time()
                eval_time = toc - tic
                logger.add_result(run, result)
                if best_val < result[1]:
                    best_val = result[1]
                    final_test = result[2]

                train_roc_auc, valid_roc_auc, test_roc_auc = result
                if epoch % args.log_steps == 0:
                    print(f'Run: {run + 1:02d}, '
                        f'Epoch: {epoch:02d}, '
                        f'Time: {train_time:.2f}/{eval_time:.2f}, '
                        f'Loss: {loss:.4f}, '
                        f'Tr/Va/Te: {train_roc_auc:.4f}/{valid_roc_auc:.4f}/{test_roc_auc:.4f}, '
                        f'Best Va/Te: {best_val:.4f}/{final_test:.4f}')

        print(f'{args.model}')
        logger.print_statistics(run)

    print(f'{args.model}')
    logger.print_statistics()

if __name__ == "__main__":
    main()
