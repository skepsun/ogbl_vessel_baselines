import torch

def drop_edge(edge_index, p=0., training=True):
    num_edges = edge_index.size(1)
    if not training:
        mask = torch.ones((num_edges,)).bool()
    else:
        mask = torch.rand((num_edges,)) > p
    return mask.to(edge_index.device)

def count_parameters(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
