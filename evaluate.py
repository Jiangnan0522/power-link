"""Evaluate explanations produced by run_powerlink.py / run_gnnexp.py / run_pgexp.py.

Computes mask-based and path-based Fidelity+/Fidelity-, ranking drop (HΔR),
sparsity, and average path length.
"""

import argparse
import os

import dgl
import numpy as np
import torch
from tqdm.auto import tqdm

from power_link.eval_metrics import (
    cal_fidelity,
    cal_ranking_diff,
    cal_ranking_drop_hit,
    cal_rankings,
    cal_sparsity,
)
from power_link.kge import build_graph, build_kgc_model_from_config, load_config, load_data
from power_link.utils import load_pred_paths


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Power-Link explanations')
    parser.add_argument('--explain_results_dir', default='./saved_explanations_experiment')
    parser.add_argument('--kge_model_config_path', type=str,
                        default='./saved_models/config_compgcn_transe_fb15k237.json',
                        help='The path of the config file for the KGC model')
    parser.add_argument('--unpath', action='store_true', help='Explanations were trained without path loss (ablation)')
    parser.add_argument('--euclidean', action='store_true', help='Explanations used the Euclidean combination method')
    parser.add_argument('--power_order', type=int, default=3)
    parser.add_argument('--without_mi', action='store_true', help='Explanations were trained without the MI loss')
    parser.add_argument('--saved_model_dir', type=str, default='./saved_models')
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--mask_nature', type=str, default='soft', choices=['soft', 'hard'])
    parser.add_argument('--fidelity_format', type=str, default='prob', choices=['prob', 'acc'])
    parser.add_argument('--max_num_samples', type=int, default=-1, help='Maximum number of samples to evaluate. -1 = all.')
    parser.add_argument('--num_paths', type=int, default=5, help='Top-N paths used for path-based fidelity. -1 = all.')
    return parser.parse_args()


def get_edge_eid(edge, g):
    """Translate a (canonical_etype, src_nid, tgt_nid) triple into an eid in ``g``."""
    u, v = edge[1], edge[2]
    eid = ((g.edges()[0] == u) & (g.edges()[1] == v)).nonzero()
    return eid, edge[0]


def kgnid2compgnid(comp_g, kg_triplet):
    """Map a triplet's KG node ids onto computation-graph node ids."""
    comp_g = comp_g.cpu()
    kgnid2nid_mapping = dict(zip(comp_g.nodes[:].data[dgl.NID].tolist(), comp_g.nodes().tolist()))
    return [
        kgnid2nid_mapping[kg_triplet[0]],
        kg_triplet[1],
        kgnid2nid_mapping[kg_triplet[2]],
    ]


def select_device(device_id):
    if torch.cuda.is_available() and device_id >= 0:
        return torch.device('cuda', index=device_id)
    return torch.device('cpu')


def explanation_filename(p, args):
    parts = ['explain_results_' + p.name]
    if args.unpath:
        parts.append('_unpath')
    if args.euclidean:
        parts.append('_euc')
    if args.power_order != 3:
        parts.append(f'_o{args.power_order}')
    if args.without_mi:
        parts.append('_no_mi')
    return ''.join(parts)


def main():
    args = parse_args()
    device = select_device(args.device_id)

    # Load model + config + KG.
    p = load_config(args.kge_model_config_path)
    data = load_data(p.dataset)
    num_nodes = data.num_nodes
    num_rels = data.num_rels
    g, edge_norm, edge_type = build_graph(num_nodes, p, data.train, num_rels)
    edge_norm = edge_norm.to(device)
    edge_type = edge_type.to(device)

    mp_g_meta = {
        'num_ent': num_nodes, 'num_rels': num_rels,
        'edge_type': edge_type, 'edge_norm': edge_norm,
    }
    model = build_kgc_model_from_config(p, device, mp_g_meta)
    state = torch.load(f'{args.saved_model_dir}/{p.name}.pt', map_location=device)['model']
    state.pop('bias', None)
    model.load_state_dict(state)
    del state
    torch.cuda.empty_cache()
    model.eval()

    explain_results_path = os.path.join(args.explain_results_dir, explanation_filename(p, args))
    explain_results = load_pred_paths(explain_results_path)

    p_orig, p_masked, p_maskout = [], [], []
    p_path_masked, p_path_maskout = [], []
    rankings_orig, rankings_masked, rankings_maskout = [], [], []
    rankings_path_masked, rankings_path_maskout = [], []
    sparsity, path_length, comp_g_sparsity = [], [], []

    n = len(explain_results) if args.max_num_samples == -1 else args.max_num_samples
    for explain_result in tqdm(explain_results[:n]):
        triplet, comp_g_paths, comp_g_edge_mask_dict, comp_g = list(explain_result.values())
        comp_g_edge_mask_dict = {k: v.to(device) for k, v in comp_g_edge_mask_dict.items()}
        triplet = kgnid2compgnid(comp_g, triplet)
        triplet = [torch.tensor(x, device=device) for x in triplet]

        if any(torch.isnan(x).any().item() for x in comp_g_edge_mask_dict.values()):
            continue

        if args.mask_nature == 'soft':
            mask_weight = {etype: comp_g_edge_mask_dict[etype].sigmoid() for etype in comp_g_edge_mask_dict}
        else:
            mask_weight = {etype: torch.where(comp_g_edge_mask_dict[etype].sigmoid() > 0.5, 1.0, 0.0)
                           for etype in comp_g_edge_mask_dict}

        edge_ids = {etype: [] for etype in comp_g_edge_mask_dict}
        if args.num_paths != -1:
            comp_g_paths = comp_g_paths[:args.num_paths]
        len_path = 0
        for path in comp_g_paths:
            len_path += len(path)
            for edge in path:
                eid, etype = get_edge_eid(edge, comp_g)
                edge_ids[etype].extend(eid)
        mask_path_weight = {}
        for etype, weights in comp_g_edge_mask_dict.items():
            m_temp = torch.zeros_like(weights, device=device)
            indices = torch.cat(edge_ids[etype]) if len(edge_ids[etype]) > 0 else []
            m_temp[indices] = 1.0
            mask_path_weight[etype] = m_temp

        mask_plus = mask_weight
        mask_minus = {etype: 1.0 - mask_weight[etype] for etype in comp_g_edge_mask_dict}
        mask_path_plus = mask_path_weight
        mask_path_minus = {etype: 1.0 - mask_path_weight[etype] for etype in comp_g_edge_mask_dict}
        sps = cal_sparsity(mask_weight, format=args.fidelity_format)

        comp_g = comp_g.to(device)
        with torch.no_grad():
            pred_prob, _, _, pred_prob_all = model(triplet[0], triplet[1], triplet[2], comp_g, return_embds=True)
            prob_masked, _, _, all_probs_masked = model(triplet[0], triplet[1], triplet[2], comp_g, mask_plus, return_embds=True)
            prob_maskout, _, _, all_probs_maskout = model(triplet[0], triplet[1], triplet[2], comp_g, mask_minus, return_embds=True)
            prob_path_masked, _, _, all_probs_path_masked = model(triplet[0], triplet[1], triplet[2], comp_g, mask_path_plus, return_embds=True)
            prob_path_maskout, _, _, all_probs_path_maskout = model(triplet[0], triplet[1], triplet[2], comp_g, mask_path_minus, return_embds=True)

        rankings_orig.append(cal_rankings(pred_prob, pred_prob_all))
        rankings_masked.append(cal_rankings(prob_masked, all_probs_masked))
        rankings_maskout.append(cal_rankings(prob_maskout, all_probs_maskout))
        rankings_path_masked.append(cal_rankings(prob_path_masked, all_probs_path_masked))
        rankings_path_maskout.append(cal_rankings(prob_path_maskout, all_probs_path_maskout))

        p_orig.append(pred_prob.item())
        p_masked.append(prob_masked.item())
        p_maskout.append(prob_maskout.item())
        p_path_masked.append(prob_path_masked.item())
        p_path_maskout.append(prob_path_maskout.item())
        sparsity.append(sps)
        comp_g_sparsity.append(comp_g.number_of_edges() / (comp_g.number_of_nodes() ** 2))
        path_length.append(len_path / max(1, len(comp_g_paths)))

    avg_fidelity_mask = cal_fidelity(np.array(p_orig), np.array(p_masked), np.array(p_maskout), format=args.fidelity_format)
    avg_fidelity_path = cal_fidelity(np.array(p_orig), np.array(p_path_masked), np.array(p_path_maskout), format=args.fidelity_format)
    avg_ranking_diff = cal_ranking_diff(rankings_maskout, rankings_orig)
    avg_path_ranking_diff = cal_ranking_diff(rankings_path_maskout, rankings_orig)
    avg_ranking_hit = cal_ranking_drop_hit(rankings_maskout, rankings_orig)
    avg_path_ranking_hit = cal_ranking_drop_hit(rankings_path_maskout, rankings_orig)
    avg_sparsity = float(np.mean(sparsity))
    avg_comp_g_sparsity = float(np.mean(comp_g_sparsity))
    avg_path_length = float(np.mean(path_length))

    print('*' * 80)
    print(f'Average Mask Fidelity: fidelity+ {avg_fidelity_mask[0]:.4f}, fidelity- {avg_fidelity_mask[1]:.4f}')
    print(f'Average Ranking Diff: {avg_ranking_diff:.4f}')
    print(f'Average Ranking Drop Hit Rate: {avg_ranking_hit}')
    print('-' * 80)
    print(f'Average Path Fidelity: fidelity+ {avg_fidelity_path[0]:.4f}, fidelity- {avg_fidelity_path[1]:.4f}')
    print(f'Average Path Ranking Diff: {avg_path_ranking_diff}')
    print(f'Average Path Ranking Drop Hit Rate: {avg_path_ranking_hit}')
    print(f'Average Path Length: {avg_path_length:.4f}')
    print(f'Average Sparsity: {avg_sparsity:.4f}')
    print(f'Average Comp G Sparsity: {avg_comp_g_sparsity:.4f}')
    print('*' * 80)


if __name__ == '__main__':
    main()
