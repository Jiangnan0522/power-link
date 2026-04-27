"""Seeding, configuration loading, and miscellaneous IO helpers."""

import pickle
import random

import numpy as np
import torch
import yaml


def load_pred_paths(dir):
    with open(dir, "rb") as f:
        paths = pickle.load(f)
    return paths


class COLORS:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def cuda_usage():
    u = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) // (1024 ** 2)
    return u


def get_label(label, num_ent):
    """Build a one-hot tensor of shape ``[num_ent]`` from a list of object indices."""
    y = np.zeros([num_ent], dtype=np.float32)
    y[label] = 1
    return torch.tensor(y, dtype=torch.float32)


def set_seed(seed):
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_args(args):
    for k, v in vars(args).items():
        print(f'{k:25} {v}')


def set_config_args(args, config_path, dataset_name, model_name=''):
    with open(config_path, "r") as conf:
        config = yaml.load(conf, Loader=yaml.FullLoader)[dataset_name]
        if model_name:
            config = config[model_name]
    for key, value in config.items():
        setattr(args, key, value)
    return args


def idx_split(idx, ratio, seed=0):
    """Randomly split ``idx`` into two parts at ``ratio`` : ``1 - ratio``."""
    set_seed(seed)
    n = len(idx)
    cut = int(n * ratio)
    idx_idx_shuffle = torch.randperm(n)
    idx1_idx, idx2_idx = idx_idx_shuffle[:cut], idx_idx_shuffle[cut:]
    idx1, idx2 = idx[idx1_idx], idx[idx2_idx]
    assert ((torch.cat([idx1, idx2]).sort()[0] == idx.sort()[0]).all())
    return idx1, idx2


def eids_split(eids, val_ratio, test_ratio, seed=0):
    """Split ``eids`` into train / valid / test according to the given ratios."""
    train_ratio = (1 - val_ratio - test_ratio)
    train_eids, pred_eids = idx_split(eids, train_ratio, seed)
    val_eids, test_eids = idx_split(pred_eids, val_ratio / (1 - train_ratio), seed)
    return train_eids, val_eids, test_eids


def negative_sampling(graph, pred_etype=None, num_neg_samples=None):
    """Sample negative edges. Adapted from PyG ``negative_sampling``."""
    pos_src_nids, pos_tgt_nids = graph.edges(etype=pred_etype)
    if pred_etype is None:
        N = graph.num_nodes()
        M = N * N
    else:
        src_ntype, _, tgt_ntype = graph.to_canonical_etype(pred_etype)
        src_N, N = graph.num_nodes(src_ntype), graph.num_nodes(tgt_ntype)
        M = src_N * N

    pos_M = pos_src_nids.shape[0]
    neg_M = num_neg_samples or pos_M
    neg_M = min(neg_M, M - pos_M)

    alpha = abs(1 / (1 - 1.1 * (pos_M / M)))
    size = min(M, int(alpha * neg_M))
    perm = torch.tensor(random.sample(range(M), size))

    idx = pos_src_nids * N + pos_tgt_nids
    mask = torch.isin(perm, idx.to('cpu')).to(torch.bool)
    perm = perm[~mask][:neg_M].to(pos_src_nids.device)

    neg_src_nids = torch.div(perm, N, rounding_mode='floor')
    neg_tgt_nids = perm % N

    return neg_src_nids, neg_tgt_nids
