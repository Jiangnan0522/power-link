import argparse
import json
import logging
import os
import pickle

import dgl
import numpy as np
import torch
from tqdm.auto import tqdm

from power_link.baselines import HeteroPGExplainer
from power_link.data_processing import get_test_triplets
from power_link.kge import build_kgc_model_from_config, load_config, load_graph
from power_link.utils import print_args, set_config_args, set_seed

parser = argparse.ArgumentParser(description='Explain link predictor')
parser.add_argument('--device_id', type=int, default=0)

'''
Dataset args
'''
parser.add_argument('--dataset_dir', type=str, default='datasets')
parser.add_argument('--valid_ratio', type=float, default=0.1) 
parser.add_argument('--test_ratio', type=float, default=0.2)
parser.add_argument('--max_num_samples', type=int, default=-1, 
                    help='maximum number of samples to explain, for fast testing. Use all if -1')

'''
GNN args
'''
parser.add_argument('--emb_dim', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--out_dim', type=int, default=128)
parser.add_argument('--saved_model_dir', type=str, default='saved_models')
parser.add_argument('--saved_model_name', type=str, default='')

'''
Link predictor args
'''
parser.add_argument('--src_ntype', type=str, default='user', help='prediction source node type')
parser.add_argument('--tgt_ntype', type=str, default='item', help='prediction target node type')
parser.add_argument('--pred_etype', type=str, default='likes', help='prediction edge type')
parser.add_argument('--link_pred_op', type=str, default='dot', choices=['dot', 'cos', 'ele', 'cat'],
                   help='operation passed to dgl.EdgePredictor')

'''
Explanation args
'''
parser.add_argument('--alpha1', type=float, default=2e-3, help='explainer sparsity regularizer weight') 
parser.add_argument('--alpha2', type=float, default=1.0, help='explainer entropy regularizer weight') 
parser.add_argument('--lr', type=float, default=0.01, help='explainer learning_rate') 
parser.add_argument('--mask_generator_hidden_dim', type=int, default=64, 
                    help='hidden dimension of mask generator') 
parser.add_argument('--beta', type=float, default=1.0, help='explainer off-path edge regularizer weight') 
parser.add_argument('--num_hops', type=int, default=2, help='computation graph number of hops') 
parser.add_argument('--num_epochs', type=int, default=20, help='How many epochs to learn the mask')
parser.add_argument('--num_paths', type=int, default=40, help='How many paths to generate')
parser.add_argument('--max_path_length', type=int, default=5, help='max lenght of generated paths')
parser.add_argument('--k_core', type=int, default=2, help='k for the k-core graph') 
parser.add_argument('--prune_max_degree', type=int, default=200,
                    help='prune the graph such that all nodes have degree smaller than max_degree. No prune if -1') 
parser.add_argument('--save_explanation', default=False, action='store_true', 
                    help='Whether to save the explanation')
parser.add_argument('--saved_explanation_dir', type=str, default='saved_explanations_experiment',
                    help='directory of saved explanations')
parser.add_argument('--config_path', type=str, default='', help='path of saved configuration args')

# ******** New arg!**************
parser.add_argument('--power_order',dest='power_order', type=int, default=3, help='The order of the power loss. The default is 3, which optimizes explanation paths of max length 3')
parser.add_argument('--kge_model_config_path',dest='kge_model_config_path', type=str, default='./saved_models/config_compgcn_transe_fb15k237.json', help='The path of the config file for the KGE model to be explained')
parser.add_argument('--regularisation_weight', dest='regularisation_weight', type=float, default=0.001, help='Regularisation weight for the explainer.')
parser.add_argument('--parameterizer_hidden_dim',dest='parameterizer_hidden_dim', type=int, default=64, help='Hidden dimension size of the parameterizer MLP')
parser.add_argument('--without_path_loss',dest='without_path_loss', action='store_const', default=False, const=True, help='Whether or not use path loss for explaining')
parser.add_argument('--hit1',dest='hit1', action='store_const', default=False, const=True, help='Only explain the hit1 samples')
parser.add_argument('--hit3',dest='hit3', action='store_const', default=False, const=True, help='Only explain the hit3 samples')
parser.add_argument('--comp_g_size_limit', dest='comp_g_size_limit', type=int, default=1000, help='Limit on the largest node number of the computational graph of the sample. If the exceeded, ignore the sample.')
parser.add_argument('--pagelink_loss',dest='pagelink_loss', action='store_const', default=False, const=True, help='Whether or not to use the pagelink loss for explaining.')
parser.add_argument('--seed',dest='seed', type=int, default=8, help='The seed for replication.')

args = parser.parse_args()
set_seed(args.seed)

if args.config_path:
    args = set_config_args(args, args.config_path, args.dataset_name, 'pagelink')

if torch.cuda.is_available() and args.device_id >= 0:
    device = torch.device('cuda', index=args.device_id)
else:
    device = torch.device('cpu')

if args.link_pred_op in ['cat']:
    pred_kwargs = {"in_feats": args.out_dim, "out_feats": 1}
else:
    pred_kwargs = {}
    
print_args(args)
p = load_config(args.kge_model_config_path)
kg_data, mp_g, mp_g_meta = load_graph(p, device)
num_ent = mp_g.number_of_nodes()
num_rels = mp_g_meta['num_rels'] # * num_rels means the number of relation types instead of the number of edges
edge_type = mp_g_meta['edge_type']
edge_norm = mp_g_meta['edge_norm']
args.src_ntype = '_N'
args.tgt_ntype = '_N'

# set logger
if args.without_path_loss:
    FILENAME = f'pgexp_explain_results_{p.encoder}_{p.score_func}_{p.dataset}_unpath' 
elif not args.pagelink_loss:
    FILENAME = f'pgexp_explain_results_{p.encoder}_{p.score_func}_{p.dataset}'
else:
    FILENAME = f'pgexp_explain_results_{p.encoder}_{p.score_func}_{p.dataset}_pl'

logging.basicConfig(
    level=logging.INFO, 
    filename=FILENAME + '.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)

mp_g_meta['num_ent'] = num_ent
model = build_kgc_model_from_config(p, device, mp_g_meta)
model.src_ntype, model.tgt_ntype, model.link_pred_op = args.src_ntype, args.tgt_ntype, args.link_pred_op
state = torch.load(f'{args.saved_model_dir}/{p.name}.pt', map_location=device)['model']
if 'bias' in state:
    state.pop('bias')
model.load_state_dict(state)
del state
torch.cuda.empty_cache()
model.to(device)

# * Freeze the model
for param in model.parameters():
    param.requires_grad = False

pgexplainer = HeteroPGExplainer(
    model, 
    num_hops=args.num_hops, 
    ghetero=mp_g,
    lr=args.lr,
    alpha1=args.alpha1, 
    alpha2=args.alpha2, 
    embed_dim=p.embed_dim, 
    mask_generator_hidden_dim=args.mask_generator_hidden_dim,
    num_epochs=args.num_epochs
    ).to(device)

# * prepare test data to be explained using data from KG
subj, rel, obj = get_test_triplets(kg_data, device, shuffle=True)

# * max number of samples to test
pred_edge_to_comp_g_edge_mask = []
pred_edge_to_paths = []
pred_results = []
num_explained = 0
text_visualisations = []

for i in tqdm(range(subj.shape[0])):
    i_subj, i_rel, i_obj = subj[i], rel[i], obj[i]

    # * Only explain edges that the model consider existent
    model.eval()
    with torch.no_grad():
        score, node_embds, rel_embds, score_all = model(i_subj, i_rel, i_obj, mp_g, return_embds=True)
        if args.hit1:
            rankings = (torch.gt(score_all, score) | torch.isclose(score_all, score)).sum()
            pred = rankings == 1
        elif args.hit3:
            rankings = (torch.gt(score_all, score) | torch.isclose(score_all, score)).sum()
            pred = rankings <= 3
        else:
            pred = score > 0.5

    if pred:
        # max number of samples to be explained
        print(f'\033[92m Explaining sample: {num_explained}, original index: {i} \033[0m \n')
        src_tgt = ((args.src_ntype, int(i_subj)), (args.tgt_ntype, int(i_obj)))
        comp_g, comp_g_paths, comp_g_edge_mask_dict = \
        pgexplainer.explain(
            src_nid = i_subj, 
            tgt_nid = i_obj,
            rel_id = i_rel, 
            ghetero = mp_g,
            num_hops=args.num_hops,
            comp_g_size_limit=args.comp_g_size_limit,
            )
        
        # When the computational graph is larger than given limit, the sample is ignored.
        if (comp_g is None) or (comp_g_paths is None) or (comp_g_edge_mask_dict is None):
            continue
        
        if np.any([torch.isnan(x).any().item() for x in comp_g_edge_mask_dict.values()]):
            continue
            
        comp_g_edge_mask_dict = {k: v.detach().cpu() for k, v in comp_g_edge_mask_dict.items()}
        comp_g = comp_g.cpu()
        comp_g.edata['eweight'] = comp_g.edata['eweight'].detach().cpu()
        torch.cuda.empty_cache()
        pred_results.append({
            'triplet':(i_subj.item(), i_rel.item(), i_obj.item()),
            'comp_g_paths':comp_g_paths,
            'comp_g_mask':comp_g_edge_mask_dict,
            'comp_g':comp_g
        })

        num_explained += 1
        if num_explained >= args.max_num_samples:
            break
print(f'Total number of samples explained: {num_explained}')

if args.save_explanation:
    if not os.path.exists(args.saved_explanation_dir):
        os.makedirs(args.saved_explanation_dir)

    with open(args.saved_explanation_dir + '/' + FILENAME, 'wb') as f:
        pickle.dump(pred_results, f)
