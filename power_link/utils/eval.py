"""Helpers for converting graph-level edge / path labels into computation-graph
identifiers, and for measuring AUC / hit rates of an explainer's edge mask."""

from collections import defaultdict

import dgl
import torch
from sklearn.metrics import roc_auc_score


def get_comp_g_edge_labels(comp_g, edge_labels):
    """Translate ``edge_labels`` (over the original graph) onto ``comp_g``."""
    ntype_to_tensor_nids_to_comp_g_nids = {}
    ntypes_to_comp_g_max_nids = {}
    ntypes_to_nids = comp_g.ndata[dgl.NID]
    for ntype in ntypes_to_nids.keys():
        nids = ntypes_to_nids[ntype]
        max_nid = nids.max().item() if nids.numel() > 0 else -1
        ntypes_to_comp_g_max_nids[ntype] = max_nid

        nids_to_comp_g_nids = torch.zeros(max_nid + 1).long() - 1
        nids_to_comp_g_nids[nids] = torch.arange(nids.shape[0])
        ntype_to_tensor_nids_to_comp_g_nids[ntype] = nids_to_comp_g_nids

    comp_g_edge_labels = {}
    for can_etype in edge_labels:
        start_ntype, etype, end_ntype = can_etype
        start_nids, end_nids = edge_labels[can_etype]
        start_max = ntypes_to_comp_g_max_nids[start_ntype]
        end_max = ntypes_to_comp_g_max_nids[end_ntype]

        start_mask = start_nids <= start_max
        end_mask = end_nids <= end_max
        keep = end_mask & start_mask

        start_nids = start_nids[keep]
        end_nids = end_nids[keep]

        comp_g_start_nids = ntype_to_tensor_nids_to_comp_g_nids[start_ntype][start_nids]
        comp_g_end_nids = ntype_to_tensor_nids_to_comp_g_nids[end_ntype][end_nids]
        comp_g_eids = comp_g.edge_ids(comp_g_start_nids.tolist(), comp_g_end_nids.tolist(), etype=etype)

        num_edges = comp_g.num_edges(etype=can_etype)
        comp_g_eid_mask = torch.zeros(num_edges)
        comp_g_eid_mask[comp_g_eids] = 1
        comp_g_edge_labels[can_etype] = comp_g_eid_mask

    return comp_g_edge_labels


def get_comp_g_path_labels(comp_g, path_labels):
    """Translate path-shaped labels onto edge-id sequences in ``comp_g``."""
    ntype_to_tensor_nids_to_comp_g_nids = {}
    ntypes_to_nids = comp_g.ndata[dgl.NID]
    for ntype in ntypes_to_nids.keys():
        nids = ntypes_to_nids[ntype]
        max_nid = nids.max().item() if nids.numel() > 0 else -1
        nids_to_comp_g_nids = torch.zeros(max_nid + 1).long() - 1
        nids_to_comp_g_nids[nids] = torch.arange(nids.shape[0])
        ntype_to_tensor_nids_to_comp_g_nids[ntype] = nids_to_comp_g_nids

    comp_g_path_labels = []
    for path in path_labels:
        comp_g_path = []
        for can_etype, start_nid, end_nid in path:
            start_ntype, etype, end_ntype = can_etype
            comp_g_start_nid = ntype_to_tensor_nids_to_comp_g_nids[start_ntype][start_nid].item()
            comp_g_end_nid = ntype_to_tensor_nids_to_comp_g_nids[end_ntype][end_nid].item()
            comp_g_eid = comp_g.edge_ids(comp_g_start_nid, comp_g_end_nid, etype=can_etype)
            comp_g_path += [(can_etype, comp_g_eid)]
        comp_g_path_labels += [comp_g_path]
    return comp_g_path_labels


def eval_edge_mask_auc(edge_mask_dict, edge_labels):
    """ROC-AUC of an edge mask against ground-truth edge labels."""
    y_true = []
    y_score = []
    for can_etype in edge_labels:
        y_true += [edge_labels[can_etype]]
        y_score += [edge_mask_dict[can_etype].detach().sigmoid()]

    y_true = torch.cat(y_true)
    y_score = torch.cat(y_score)
    return roc_auc_score(y_true, y_score)


def eval_edge_mask_topk_path_hit(edge_mask_dict, path_labels, topks=(10,)):
    """For each ``k``, what fraction of ground-truth paths are entirely covered by the top-k edges?"""
    cat_edge_mask = torch.cat([v for v in edge_mask_dict.values()])
    M = len(cat_edge_mask)
    topks = {k: min(k, M) for k in topks}

    topk_to_path_hit = defaultdict(list)
    for r, k in topks.items():
        threshold = cat_edge_mask.topk(k)[0][-1].item()
        hard_edge_mask_dict = {etype: edge_mask_dict[etype] >= threshold for etype in edge_mask_dict}
        hit = eval_hard_edge_mask_path_hit(hard_edge_mask_dict, path_labels)
        topk_to_path_hit[r] += [hit]
    return topk_to_path_hit


def eval_hard_edge_mask_path_hit(hard_edge_mask_dict, path_labels):
    """1 iff at least one ground-truth path is fully retained by ``hard_edge_mask_dict``."""
    for path in path_labels:
        hit_path = 1
        for can_etype, eid in path:
            if not hard_edge_mask_dict[can_etype][eid]:
                hit_path = 0
                break
        if hit_path:
            return 1
    return 0


def eval_path_explanation_edges_path_hit(path_explanation_edges, path_labels):
    """1 iff at least one ground-truth path is a subset of ``path_explanation_edges``."""
    for path in path_labels:
        hit_path = 1
        for edge in path:
            if edge not in path_explanation_edges:
                hit_path = 0
                break
        if hit_path:
            return 1
    return 0
