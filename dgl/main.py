import argparse
import os
import time

import torch
import torch.nn.functional as F
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
from torch.utils.data import DataLoader

import dgl
import dgl.function as fn
from logger import Logger
from models import GAT, GCN, MLP, SAGE, LinkPredictor

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def train(model, predictor, g, emb, split_edge, optimizer, batch_size):
    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(g.device)

    neg_train_edge = split_edge['train']['edge_neg'].to(g.device) 

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):

        optimizer.zero_grad()

        h = model(g)

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

        if g.ndata['feat'].requires_grad:
            torch.nn.utils.clip_grad_norm_(emb.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples

@torch.no_grad()
def test(model, predictor, g, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()

    h = model(g)

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

def run(run_id, model, predictor, emb, g, split_edge, evaluator, logger, args):
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
        loss = train(model, predictor, g, emb, split_edge, optimizer,
                        args.batch_size)
        toc = time.time()
        train_time = toc - tic

        if epoch % args.eval_steps == 0:
            tic = time.time()
            result = test(model, predictor, g, split_edge, evaluator,
                            args.batch_size)
            toc = time.time()
            eval_time = toc - tic
            logger.add_result(run_id, result)
            if best_val < result[1]:
                best_val = result[1]
                final_test = result[2]

            train_roc_auc, valid_roc_auc, test_roc_auc = result
            print(f'Run: {run_id + 1:02d}, '
                f'Epoch: {epoch:02d}, '
                f'Time: {train_time:.2f}/{eval_time:.2f}, '
                f'Loss: {loss:.4f}, '
                f'Tr/Va/Te: {train_roc_auc:.4f}/{valid_roc_auc:.4f}/{test_roc_auc:.4f}, '
                f'Best Va/Te: {best_val:.4f}/{final_test:.4f}')
    return

def count_parameters(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def main():
    parser = argparse.ArgumentParser(description='Simple OGBL-VESSEL baselines (DGL version).')
    # Common settings
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.001) 
    parser.add_argument('--epochs', type=int, default=100) 
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)

    # Specific tricks
    parser.add_argument('--add_self_loops', action='store_true')
    parser.add_argument('--use_node2vec_embedding', action='store_true')
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--node_feat_process', type=str, default='node_normalize')
    parser.add_argument('--feat2edge', action='store_true')
    # Model selection
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--predictor', type=str, default='DOT')
    # Model parameters
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--n_hidden', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--input_drop', type=float, default=0.0)
    parser.add_argument('--attn_drop', type=float, default=0.0)
    parser.add_argument('--edge_drop', type=float, default=0.0)
    parser.add_argument('--bn', action='store_true')

    # Normalization settings for gat
    parser.add_argument('--norm', type=str, default='none', choices=['none', 'adj'])

    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = DglLinkPropPredDataset('ogbl-vessel', root='/mnt/ssd/ssd/dataset')
    g = dataset[0]

    split_edge = dataset.get_edge_split()

    if args.node_feat_process == 'node_normalize':
        # normalize x,y,z coordinates 
        g.ndata['feat'] = torch.nn.functional.normalize(g.ndata['feat'], dim=0)
    elif args.node_feat_process == 'channel_normalize':
        g.ndata['feat'] = torch.nn.functional.normalize(g.ndata['feat'], dim=1)
    elif args.node_feat_process == 'max-min':
        g.ndata['feat'] = (g.ndata['feat'] - g.ndata['feat'].min(dim=0)[0]) / (g.ndata['feat'].max(0)[0] - g.ndata['feat'].min(0)[0] + 1e-9)
    elif args.node_feat_process == 'z-score':
        g.ndata['feat'] = (g.ndata['feat'] - g.ndata['feat'].mean(0)) / (g.ndata['feat'].std(0) + 1e-9)
    elif args.node_feat_process == 'log':
        g.ndata['feat'] = g.ndata['feat'].abs().clamp(min=1e-9).log() * g.ndata['feat'] / g.ndata['feat'].abs().clamp(min=1e-9)
    elif args.node_feat_process == 'none':
        pass
    else:
        raise(Exception(f'The preprocessing method of node features: {args.node_feat_process} has not been implemented'))


    if args.add_self_loops:
        # There exist 74654 isolated nodes in this graph, 
        # we should add self-loops to ensure their features can be correctly preserved and encoded.
        g = g.remove_self_loop().add_self_loop()

        # optional, only add self-loops to isolated nodes
        # degs = g.in_degrees()
        # isolated_idx = (degs == 0).nonzero().squeeze(1)
        # g.add_edges(isolated_idx, isolated_idx)

    if args.feat2edge:
        # Move features from nodes to edges, then aggregate.
        src_feat = g.ndata['feat'][g.edges()[0]]
        dst_feat = g.ndata['feat'][g.edges()[1]]
        mask = src_feat[:, -1] < dst_feat[:, -1]
        src_feat[mask], dst_feat[mask] = dst_feat[mask], src_feat[mask]
        g.edata['feat'] = (src_feat + dst_feat).to(torch.float).relu()
        # degs = g.in_degrees()
        # deg_inv_sqrt = degs.pow(-0.5)
        # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # gcn_weights = deg_inv_sqrt[g.edges()[0]] * deg_inv_sqrt[g.edges()[1]]
        # g.edata['feat'] = g.edata['feat'] * gcn_weights.view(-1, 1)
        g.update_all(fn.copy_e('feat', 'm'), fn.mean('m', 'feat'))

    # g.edata['feat'][g.edata['feat'][:,-1]<0]
    # mask = g.edata['feat'].norm(2, dim=1) > 0.0015
    # g = dgl.remove_edges(g, mask.nonzero().squeeze(1))

    print(g)
    g.ndata['feat'] = g.ndata['feat'].to(torch.float)
    if args.use_node2vec_embedding:
        # g.ndata['feat'] = torch.cat([g.ndata['feat'], torch.load('../embedding.pt')], dim=-1)
        g.ndata['feat'] = torch.load('../embedding.pt')
    if args.use_node_embedding:
        emb = torch.nn.Embedding(g.ndata['feat'].size(0),
                             args.n_hidden)
        # g.ndata['feat'] = torch.cat([g.ndata['feat'], emb.weight], dim=1)
        g.ndata['feat'] = emb.weight
    else:
        emb = None
    g = g.to(device)
    # As we mentioned, the isolated nodes with zero in-degrees will hurt model performance,
    # but we can still force to train models with them.
    allow_zero_in_degree = True

    if args.model == 'mlp':
        model = MLP(g.ndata['feat'].size(1), args.n_hidden, 
            args.n_layers, args.n_hidden, F.relu,
            args.dropout, args.input_drop, 
            bn=args.bn).to(device)
    elif args.model == 'gcn':
        model = GCN(g.ndata['feat'].size(1), args.n_hidden, 
            args.n_layers, args.n_hidden, F.relu,
            args.dropout, args.input_drop, 
            allow_zero_in_degree=allow_zero_in_degree,
            bn=args.bn,
            global_attn=False).to(device)
    elif args.model == 'sage':
        model = SAGE(g.ndata['feat'].size(1), args.n_hidden, 
            args.n_layers, args.n_hidden, F.relu,
            args.dropout, args.input_drop, args.edge_drop,
            allow_zero_in_degree=allow_zero_in_degree,
            bn=args.bn,
            global_attn=False).to(device)
    elif args.model == 'gat':
        model = GAT(g.ndata['feat'].size(1), args.n_hidden,
            args.n_layers, args.n_heads, args.n_hidden, 
            F.relu, args.dropout, args.input_drop, args.attn_drop, args.edge_drop,
            allow_zero_in_degree=allow_zero_in_degree,
            norm=args.norm,
            bn=args.bn).to(device)

    predictor = LinkPredictor(args.n_hidden, args.n_hidden, 1,
                              args.n_layers, args.dropout, method=args.predictor).to(device)

    evaluator = Evaluator(name='ogbl-vessel')
    logger = Logger(args.runs, args)   

    for run_id in range(args.runs):
        run(run_id, model, predictor, emb, g, split_edge, evaluator, logger, args)
        logger.print_statistics(run_id)

    print('GNN')
    logger.print_statistics()

if __name__ == "__main__":
    main()
