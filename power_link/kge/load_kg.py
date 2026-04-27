from collections import defaultdict as ddict

import dgl
import numpy as np
import torch

from .config import load_config
from .knowledge_graph import load_data

def build_graph(num_ent, p, train_data, num_rel):
    g = dgl.DGLGraph()
    g.add_nodes(num_ent)

    if not p.rat:
        g.add_edges(train_data[:, 0], train_data[:, 2], data={'edge_type':torch.from_numpy(train_data[:,1])})
        g.add_edges(train_data[:, 2], train_data[:, 0], data={'edge_type':torch.from_numpy(train_data[:,1]+num_rel)})
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        norm = in_deg ** -0.5
        norm[torch.isinf(norm).bool()] = 0
        g.ndata['xxx'] = norm
        g.apply_edges(
            lambda edges: {'xxx': edges.dst['xxx'] * edges.src['xxx']})

        norm = g.edata.pop('xxx').squeeze()
        # * The first-half of edges and the second-half are edges of opposite directions
        edge_type =  torch.tensor(np.concatenate([train_data[:, 1], train_data[:, 1] + num_rel]))
    else:
        if p.ss > 0:
            sampleSize = p.ss
        else:
            sampleSize = num_ent - 1
        g.add_edges(train_data[:, 0], np.random.randint(
            low=0, high=sampleSize, size=train_data[:, 2].shape))
        g.add_edges(train_data[:, 2], np.random.randint(
            low=0, high=sampleSize, size=train_data[:, 0].shape))
    return g, norm, edge_type

def process(dataset, num_rel):
    """
    pre-process dataset
    :param dataset: a dictionary containing 'train', 'valid' and 'test' data.
    :param num_rel: relation number
    :return:
    """
    sr2o = ddict(set)
    for subj, rel, obj in dataset['train']:
        sr2o[(subj, rel)].add(obj)
        sr2o[(obj, rel + num_rel)].add(subj)
    sr2o_train = {k: list(v) for k, v in sr2o.items()}
    for split in ['valid', 'test']:
        for subj, rel, obj in dataset[split]:
            sr2o[(subj, rel)].add(obj)
            sr2o[(obj, rel + num_rel)].add(subj)
    sr2o_all = {k: list(v) for k, v in sr2o.items()}
    triplets = ddict(list)

    for (subj, rel), obj in sr2o_train.items():
        triplets['train'].append({'triple': (subj, rel, -1), 'label': sr2o_train[(subj, rel)]})
    for split in ['valid', 'test']:
        for subj, rel, obj in dataset[split]:
            triplets[f"{split}_tail"].append({'triple': (subj, rel, obj), 'label': sr2o_all[(subj, rel)]})
            triplets[f"{split}_head"].append(
                {'triple': (obj, rel + num_rel, subj), 'label': sr2o_all[(obj, rel + num_rel)]})
    triplets = dict(triplets)
    return triplets

def load_graph(p, device):
    data = load_data(p.dataset)
    num_ent, train_data, valid_data, test_data, num_rels = data.num_nodes, data.train, data.valid, data.test, data.num_rels
    triplets = process({'train': train_data, 'valid': valid_data, 'test': test_data},
                            num_rels)
    p.embed_dim = p.k_w * \
        p.k_h if p.embed_dim is None else p.embed_dim  # output dim of gnn
    # data_iter = get_data_iter()
    g, edge_norm, edge_type = build_graph(num_ent, p, train_data, num_rels) # * g is built with training data
    g = g.to(device)
    graph_meta = {}
    graph_meta['num_rels'] = num_rels
    graph_meta['num_ent'] = num_ent
    graph_meta['edge_type'] = edge_type.to(device)
    graph_meta['edge_norm'] = edge_norm.to(device)
    return triplets, g, graph_meta
