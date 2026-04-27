"""Train a CompGCN/RGCN/WGCN encoder + TransE/DistMult/ConvE decoder.

Adapted from the "Rethinking GCNs in KGC" CIKM '21 codebase. The trained model
+ JSON config are written under ``./saved_models/`` and consumed by
``run_powerlink.py`` / ``evaluate.py``.
"""

import argparse
import json
import logging
import os
import random
import time
from pprint import pprint

import dgl
import numpy as np
import torch
from torch.utils.data import DataLoader

from power_link.kge import (
    GCN_ConvE,
    GCN_DistMult,
    GCN_TransE,
    load_data,
    process,
)
from power_link.kge.data_set import TestDataset, TrainDataset
from power_link.kge.lte_models import ConvE, DistMult, TransE


class Runner(object):
    def __init__(self, params):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.p = params
        self.data = load_data(self.p.dataset)
        self.num_ent = self.data.num_nodes
        self.train_data = self.data.train
        self.valid_data = self.data.valid
        self.test_data = self.data.test
        self.num_rels = self.data.num_rels
        self.triplets = process(
            {'train': self.train_data, 'valid': self.valid_data, 'test': self.test_data},
            self.num_rels)

        self.p.embed_dim = self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim
        self.data_iter = self.get_data_iter()

        if self.p.gpu >= 0:
            self.g = self.build_graph().to(self.device)
        else:
            self.g = self.build_graph()
        self.edge_type, self.edge_norm = self.get_edge_dir_and_norm()
        self.model = self.get_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.p.lr, weight_decay=self.p.l2)
        self.best_val_mrr, self.best_epoch, self.best_val_results = 0., 0., {}
        os.makedirs('./logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'./logs/{self.p.encoder}_{self.p.score_func}_{self.p.dataset.lower()}'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        pprint(vars(self.p))

    def fit(self):
        save_root = './saved_models'
        os.makedirs(save_root, exist_ok=True)
        save_path = f'{save_root}/{self.p.encoder}_{self.p.score_func}_{self.p.dataset.lower()}.pt'

        if self.p.restore:
            self.load_model(save_path)
            self.logger.info('Successfully Loaded previous model')

        print('Start training...')
        for epoch in range(self.p.max_epochs):
            start_time = time.time()
            train_loss = self.train()
            val_results = self.evaluate('valid')
            if val_results['mrr'] > self.best_val_mrr:
                self.best_val_results = val_results
                self.best_val_mrr = val_results['mrr']
                self.best_epoch = epoch
                self.save_model(save_path)
            msg = (f"[Epoch {epoch}]: Training Loss: {train_loss:.5}, "
                   f"Valid MRR: {val_results['mrr']:.5}, "
                   f"Best Valid MRR: {self.best_val_mrr:.5}, "
                   f"Cost: {time.time() - start_time:.2f}s")
            print(msg)
            self.logger.info(msg)
        self.logger.info(vars(self.p))
        self.load_model(save_path)
        self.logger.info(f'Loading best model in {self.best_epoch} epoch, Evaluating on Test data')
        start = time.time()
        test_results = self.evaluate('test')
        end = time.time()
        self.logger.info(
            f"MRR: Tail {test_results['left_mrr']:.5}, Head {test_results['right_mrr']:.5}, Avg {test_results['mrr']:.5}")
        self.logger.info(
            f"MR: Tail {test_results['left_mr']:.5}, Head {test_results['right_mr']:.5}, Avg {test_results['mr']:.5}")
        self.logger.info(f"hits@1 = {test_results['hits@1']:.5}")
        self.logger.info(f"hits@3 = {test_results['hits@3']:.5}")
        self.logger.info(f"hits@10 = {test_results['hits@10']:.5}")
        self.logger.info(f"time = {end - start}")

    def train(self):
        self.model.train()
        losses = []
        train_iter = self.data_iter['train']
        for step, (triplets, labels) in enumerate(train_iter):
            if self.p.gpu >= 0:
                triplets, labels = triplets.to(self.device), labels.to(self.device)
            subj, rel = triplets[:, 0], triplets[:, 1]
            pred = self.model.score_all(self.g, subj, rel) if hasattr(self.model, 'score_all') \
                else self.model(self.g, subj, rel)
            loss = self.model.calc_loss(pred, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        return float(np.mean(losses))

    def evaluate(self, split):
        def get_combined_results(left, right):
            results = dict()
            assert left['count'] == right['count']
            count = float(left['count'])
            results['left_mr'] = round(left['mr'] / count, 5)
            results['left_mrr'] = round(left['mrr'] / count, 5)
            results['right_mr'] = round(right['mr'] / count, 5)
            results['right_mrr'] = round(right['mrr'] / count, 5)
            results['mr'] = round((left['mr'] + right['mr']) / (2 * count), 5)
            results['mrr'] = round((left['mrr'] + right['mrr']) / (2 * count), 5)
            for k in [1, 3, 10]:
                results[f'left_hits@{k}'] = round(left[f'hits@{k}'] / count, 5)
                results[f'right_hits@{k}'] = round(right[f'hits@{k}'] / count, 5)
                results[f'hits@{k}'] = round((results[f'left_hits@{k}'] + results[f'right_hits@{k}']) / 2, 5)
            return results

        self.model.eval()
        left_result = self.predict(split, 'tail')
        right_result = self.predict(split, 'head')
        return get_combined_results(left_result, right_result)

    def predict(self, split='valid', mode='tail'):
        with torch.no_grad():
            results = dict()
            test_iter = self.data_iter[f'{split}_{mode}']
            for step, (triplets, labels) in enumerate(test_iter):
                triplets, labels = triplets.to(self.device), labels.to(self.device)
                subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2]
                pred = self.model.score_all(self.g, subj, rel) if hasattr(self.model, 'score_all') \
                    else self.model(self.g, subj, rel)
                b_range = torch.arange(pred.shape[0], device=self.device)
                target_pred = pred[b_range, obj]
                pred = torch.where(labels.bool(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, obj] = target_pred
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                    b_range, obj]
                ranks = ranks.float()
                results['count'] = torch.numel(ranks) + results.get('count', 0)
                results['mr'] = torch.sum(ranks).item() + results.get('mr', 0)
                results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0)
                for k in [1, 3, 10]:
                    results[f'hits@{k}'] = torch.numel(ranks[ranks <= k]) + results.get(f'hits@{k}', 0)
        return results

    def save_model(self, path):
        state = {
            'model': self.model.state_dict(),
            'best_val': self.best_val_results,
            'best_epoch': self.best_epoch,
            'optimizer': self.optimizer.state_dict(),
            'args': vars(self.p),
        }
        torch.save(state, path)

    def load_model(self, path):
        state = torch.load(path)
        self.best_val_results = state['best_val']
        self.best_val_mrr = self.best_val_results['mrr']
        self.best_epoch = state['best_epoch']
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])

    def build_graph(self):
        g = dgl.DGLGraph()
        g.add_nodes(self.num_ent)
        if not self.p.rat:
            g.add_edges(self.train_data[:, 0], self.train_data[:, 2])
            g.add_edges(self.train_data[:, 2], self.train_data[:, 0])
        else:
            sample_size = self.p.ss if self.p.ss > 0 else self.num_ent - 1
            g.add_edges(self.train_data[:, 0],
                        np.random.randint(low=0, high=sample_size, size=self.train_data[:, 2].shape))
            g.add_edges(self.train_data[:, 2],
                        np.random.randint(low=0, high=sample_size, size=self.train_data[:, 0].shape))
        return g

    def get_data_iter(self):
        def get_data_loader(dataset_class, split):
            return DataLoader(
                dataset_class(self.triplets[split], self.num_ent, self.p),
                batch_size=self.p.batch_size,
                shuffle=True,
                num_workers=self.p.num_workers,
                pin_memory=True,
            )
        return {
            'train': get_data_loader(TrainDataset, 'train'),
            'valid_head': get_data_loader(TestDataset, 'valid_head'),
            'valid_tail': get_data_loader(TestDataset, 'valid_tail'),
            'test_head': get_data_loader(TestDataset, 'test_head'),
            'test_tail': get_data_loader(TestDataset, 'test_tail'),
        }

    def get_edge_dir_and_norm(self):
        in_deg = self.g.in_degrees(range(self.g.number_of_nodes())).float()
        norm = in_deg ** -0.5
        norm[torch.isinf(norm).bool()] = 0
        self.g.ndata['xxx'] = norm
        self.g.apply_edges(lambda edges: {'xxx': edges.dst['xxx'] * edges.src['xxx']})
        if self.p.gpu >= 0:
            norm = self.g.edata.pop('xxx').squeeze().to(self.device)
            edge_type = torch.tensor(np.concatenate(
                [self.train_data[:, 1], self.train_data[:, 1] + self.num_rels])).to(self.device)
        else:
            norm = self.g.edata.pop('xxx').squeeze()
            edge_type = torch.tensor(np.concatenate(
                [self.train_data[:, 1], self.train_data[:, 1] + self.num_rels]))
        return edge_type, norm

    def get_model(self):
        common_kwargs = dict(
            num_ent=self.num_ent, num_rel=self.num_rels, num_base=self.p.num_bases,
            init_dim=self.p.init_dim, gcn_dim=self.p.gcn_dim, embed_dim=self.p.embed_dim,
            n_layer=self.p.n_layer, edge_type=self.edge_type, edge_norm=self.edge_norm,
            bias=self.p.bias, gcn_drop=self.p.gcn_drop, opn=self.p.opn, hid_drop=self.p.hid_drop,
            wni=self.p.wni, wsi=self.p.wsi, encoder=self.p.encoder,
            use_bn=(not self.p.nobn), ltr=(not self.p.noltr),
        )
        if self.p.n_layer > 0:
            score_func = self.p.score_func.lower()
            if score_func == 'transe':
                model = GCN_TransE(gamma=self.p.gamma, **common_kwargs)
            elif score_func == 'distmult':
                model = GCN_DistMult(**common_kwargs)
            elif score_func == 'conve':
                model = GCN_ConvE(
                    input_drop=self.p.input_drop, conve_hid_drop=self.p.conve_hid_drop,
                    feat_drop=self.p.feat_drop, num_filt=self.p.num_filt, ker_sz=self.p.ker_sz,
                    k_h=self.p.k_h, k_w=self.p.k_w, **common_kwargs)
            else:
                raise KeyError(f'score function {self.p.score_func} not recognized.')
        else:
            score_func = self.p.score_func.lower()
            if score_func == 'transe':
                model = TransE(self.num_ent, self.num_rels, params=self.p)
            elif score_func == 'distmult':
                model = DistMult(self.num_ent, self.num_rels, params=self.p)
            elif score_func == 'conve':
                model = ConvE(self.num_ent, self.num_rels, params=self.p)
            else:
                raise NotImplementedError

        if self.p.gpu >= 0:
            model.to(self.device)
        return model


def parse_args():
    parser = argparse.ArgumentParser(description='Pretrain a KGC model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', default='test_run', help='Run name for saving/restoring models')
    parser.add_argument('--data', dest='dataset', default='fb15k237', help='Dataset to use')
    parser.add_argument('--score_func', default='conve', help='Score function for link prediction')
    parser.add_argument('--opn', default='corr', help='Composition operation in CompGCN')
    parser.add_argument('--batch', dest='batch_size', default=256, type=int)
    parser.add_argument('--gpu', type=int, default=0, help='-1 for CPU')
    parser.add_argument('--epoch', dest='max_epochs', type=int, default=500)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lbl_smooth', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--restore', action='store_true', help='Restore from a saved checkpoint')
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--num_bases', default=-1, type=int)
    parser.add_argument('--init_dim', default=100, type=int)
    parser.add_argument('--gcn_dim', default=200, type=int)
    parser.add_argument('--embed_dim', default=None, type=int)
    parser.add_argument('--n_layer', default=1, type=int)
    parser.add_argument('--gcn_drop', default=0.1, type=float)
    parser.add_argument('--hid_drop', default=0.3, type=float)
    parser.add_argument('--conve_hid_drop', default=0.3, type=float)
    parser.add_argument('--feat_drop', default=0.2, type=float)
    parser.add_argument('--input_drop', default=0.2, type=float)
    parser.add_argument('--k_w', default=20, type=int)
    parser.add_argument('--k_h', default=10, type=int)
    parser.add_argument('--num_filt', default=200, type=int)
    parser.add_argument('--ker_sz', default=7, type=int)
    parser.add_argument('--gamma', default=9.0, type=float, help='TransE gamma')
    parser.add_argument('--rat', action='store_true', help='Random adjacency tensors (ablation)')
    parser.add_argument('--wni', action='store_true', help='Without neighbor information')
    parser.add_argument('--wsi', action='store_true', help='Without self-loop information')
    parser.add_argument('--ss', default=-1, type=int, help='Sample size for neighbor sampling')
    parser.add_argument('--nobn', action='store_true', help='No batch normalisation')
    parser.add_argument('--noltr', action='store_true', help='No linear transform of relation embeddings')
    parser.add_argument('--encoder', default='compgcn', type=str, choices=['compgcn', 'rgcn', 'wgcn'])
    parser.add_argument('--x_ops', default='', help='LTE entity ops')
    parser.add_argument('--r_ops', default='', help='LTE relation ops')
    return parser.parse_args()


def main():
    args = parse_args()
    args.name = f'{args.encoder}_{args.score_func}_{args.dataset.lower()}'

    save_dir = './saved_models'
    os.makedirs(save_dir, exist_ok=True)
    config_path = f'{save_dir}/config_{args.encoder.lower()}_{args.score_func.lower()}_{args.dataset.lower()}.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=3)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    runner = Runner(args)
    runner.fit()


if __name__ == '__main__':
    main()
