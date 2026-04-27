"""Run PowerLinkExplainer over a saved KGC checkpoint and dump explanations."""

import argparse
import json
import logging
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from power_link.data_processing import get_test_triplets
from power_link.explainer import PowerLinkExplainer
from power_link.kge import build_kgc_model_from_config, load_config, load_graph
from power_link.utils import COLORS, print_args, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Explain link predictor with PowerLink')
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--max_num_samples', type=int, default=-1, help='Maximum number of samples to explain. Use all if -1')
    parser.add_argument('--saved_model_dir', type=str, default='saved_models')
    parser.add_argument('--lr', type=float, default=0.01, help='Explainer learning rate')
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight on the on-path edge regulariser')
    parser.add_argument('--beta', type=float, default=1.0, help='Weight on the off-path edge regulariser')
    parser.add_argument('--num_hops', type=int, default=2, help='Computation graph number of hops')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of TES training epochs per sample')
    parser.add_argument('--num_paths', type=int, default=40, help='Maximum number of paths returned per sample')
    parser.add_argument('--max_path_length', type=int, default=5, help='Max length of generated paths')
    parser.add_argument('--k_core', type=int, default=2, help='k-core pruning depth')
    parser.add_argument('--prune_max_degree', type=int, default=200, help='Drop nodes with degree > this. -1 disables.')
    parser.add_argument('--save_explanation', default=False, action='store_true', help='Pickle the explanations to disk')
    parser.add_argument('--save_explanation_dir', type=str, default='saved_explanations_experiment')
    parser.add_argument('--config_path', type=str, default='', help='Saved configuration args (yaml)')
    parser.add_argument('--power_order', type=int, default=3, help='L in the powering trick (max optimised path length)')
    parser.add_argument('--kge_model_config_path', type=str,
                        default='./saved_models/config_compgcn_transe_fb15k237.json',
                        help='Pretraining-time JSON config of the KGC model to be explained')
    parser.add_argument('--regularisation_weight', type=float, default=0.001, help='gamma in gamma * ||M||_2')
    parser.add_argument('--parameterizer_hidden_dim', type=int, default=64, help='Hidden dim of the TES MLP')
    parser.add_argument('--without_path_loss', action='store_true', help='Ablation: drop the path-enforcing loss')
    parser.add_argument('--without_mi', action='store_true', help='Ablation: drop the mutual-information loss')
    parser.add_argument('--hit1', action='store_true', help='Only explain hit@1 samples')
    parser.add_argument('--hit3', action='store_true', help='Only explain hit@3 samples')
    parser.add_argument('--analysis', action='store_true', help='Record running time and GPU memory cost')
    parser.add_argument('--comp_g_size_limit', type=int, default=1000, help='Drop samples whose comp graph exceeds this many nodes')
    parser.add_argument('--path_loss', choices=['power', 'pagelink'], default='power',
                        help='Path-loss variant. "power" is the default Power-Link matrix-power loss; '
                             '"pagelink" is the Dijkstra-based PaGE-Link baseline (WWW \'23).')
    parser.add_argument('--combination_method', default='concat', choices=['concat', 'euclidean'])
    parser.add_argument('--seed', type=int, default=8)
    return parser.parse_args()


def select_device(device_id):
    if torch.cuda.is_available() and device_id >= 0:
        return torch.device('cuda', index=device_id)
    return torch.device('cpu')


def explainable_pred(model, i_subj, i_rel, i_obj, mp_g, hit1, hit3):
    """Return True iff the model considers (i_subj, i_rel, i_obj) factual under the requested rule."""
    model.eval()
    with torch.no_grad():
        score, _, _, score_all = model(i_subj, i_rel, i_obj, mp_g, return_embds=True)
        if hit1 and hit3:
            raise ValueError('Use --hit1 or --hit3, not both.')
        if hit1:
            rankings = (torch.gt(score_all, score) | torch.isclose(score_all, score)).sum()
            return rankings == 1
        if hit3:
            rankings = (torch.gt(score_all, score) | torch.isclose(score_all, score)).sum()
            return rankings <= 3
        return score > 0.5


def make_filename(p, args):
    if args.without_path_loss:
        name = f'explain_results_{p.encoder}_{p.score_func}_{p.dataset}_unpath'
    elif args.path_loss == 'pagelink':
        name = f'explain_results_{p.encoder}_{p.score_func}_{p.dataset}_pl'
    else:
        name = f'explain_results_{p.encoder}_{p.score_func}_{p.dataset}'
    if args.combination_method == 'euclidean':
        name += '_euc'
    if args.power_order != 3:
        name += f'_o{args.power_order}'
    if args.without_mi:
        name += '_no_mi'
    return name


def main():
    args = parse_args()
    set_seed(args.seed)

    device = select_device(args.device_id)
    print_args(args)
    print('*' * 100)

    p = load_config(args.kge_model_config_path)
    kg_data, mp_g, mp_g_meta = load_graph(p, device)
    mp_g_meta['num_ent'] = mp_g.number_of_nodes()
    args.src_ntype = '_N'
    args.tgt_ntype = '_N'
    print('*' * 100)

    filename = make_filename(p, args)
    logging.basicConfig(level=logging.INFO,
                        filename=filename + '.log',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Pretrained KGC model: instantiate, load checkpoint, freeze.
    model = build_kgc_model_from_config(p, device, mp_g_meta)
    model.src_ntype, model.tgt_ntype = args.src_ntype, args.tgt_ntype
    state = torch.load(f'{args.saved_model_dir}/{p.name}.pt', map_location=device)['model']
    state.pop('bias', None)
    model.load_state_dict(state)
    del state
    torch.cuda.empty_cache()
    for param in model.parameters():
        param.requires_grad = False

    explainer = PowerLinkExplainer(
        model,
        lr=args.lr,
        alpha=args.alpha,
        beta=args.beta,
        num_epochs=args.num_epochs,
        log=True,
        embed_dim=p.embed_dim,
        parameterizer_hidden_dim=args.parameterizer_hidden_dim,
    ).to(device)

    subj, rel, obj = get_test_triplets(kg_data, device, shuffle=True, seed=args.seed)

    pred_results = []
    num_explained = 0
    analysis_dict = defaultdict(list) if args.analysis else None

    start_total = time.time()
    use_pagelink_loss = (args.path_loss == 'pagelink')
    for i in tqdm(range(subj.shape[0])):
        i_subj, i_rel, i_obj = subj[i], rel[i], obj[i]

        if not explainable_pred(model, i_subj, i_rel, i_obj, mp_g, args.hit1, args.hit3):
            continue

        print(f'{COLORS.OKGREEN} Explaining sample: {num_explained}, original index: {i} {COLORS.ENDC}')
        start_per = time.time()

        comp_g, comp_g_paths, comp_g_edge_mask_dict = explainer.explain(
            src_nid=i_subj,
            tgt_nid=i_obj,
            rel_id=i_rel,
            ghetero=mp_g,
            num_hops=args.num_hops,
            prune_max_degree=args.prune_max_degree,
            k_core=args.k_core,
            num_paths=args.num_paths,
            max_path_length=args.max_path_length,
            prune_graph=True,
            without_path_loss=args.without_path_loss,
            without_mi=args.without_mi,
            return_mask=True,
            regularisation_weight=args.regularisation_weight,
            power_order=args.power_order,
            comp_g_size_limit=args.comp_g_size_limit,
            pagelink=use_pagelink_loss,
            combination_method=args.combination_method,
            analysis_dict=analysis_dict,
        )
        end_per = time.time()

        if comp_g is None or comp_g_paths is None or comp_g_edge_mask_dict is None:
            continue
        if any(torch.isnan(x).any().item() for x in comp_g_edge_mask_dict.values()):
            continue

        if args.analysis:
            analysis_dict['running_time'].append(round(end_per - start_per, 3))

        comp_g_edge_mask_dict = {k: v.detach().cpu() for k, v in comp_g_edge_mask_dict.items()}
        comp_g = comp_g.cpu()
        comp_g.edata['eweight'] = comp_g.edata['eweight'].detach().cpu()
        torch.cuda.empty_cache()

        pred_results.append({
            'triplet': (i_subj.item(), i_rel.item(), i_obj.item()),
            'comp_g_paths': comp_g_paths,
            'comp_g_mask': comp_g_edge_mask_dict,
            'comp_g': comp_g,
        })

        num_explained += 1
        if args.max_num_samples >= 0 and num_explained >= args.max_num_samples:
            break

    print(f'Total number of samples explained: {num_explained}')
    end_total = time.time()
    if args.analysis:
        analysis_dict['running_time_total'] = round(end_total - start_total, 3)
        print(analysis_dict)

    if args.save_explanation:
        os.makedirs(args.save_explanation_dir, exist_ok=True)
        with open(os.path.join(args.save_explanation_dir, filename), 'wb') as f:
            pickle.dump(pred_results, f)

    if args.analysis:
        with open('./analysis.json', 'w') as f:
            json.dump(dict(analysis_dict), f, indent=3)


if __name__ == '__main__':
    main()
