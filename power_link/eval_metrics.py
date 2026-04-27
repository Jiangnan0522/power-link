import numpy as np
import torch

def cal_fidelity(orig_prob, prob_masked, prob_maskout, format:str):
    if format =='prob':
        fidelity_plus = fidelity_gnn_prob(orig_prob, prob_maskout)
        fidelity_minus = fidelity_gnn_prob_inv(orig_prob, prob_masked)
    elif format == 'acc':
        fidelity_plus = fidelity_gnn_acc(orig_prob, prob_maskout)
        fidelity_minus = fidelity_gnn_acc_inv(orig_prob, prob_masked)
    else:
        raise Exception('Format of fidelity calculation should be either prob or acc.')
    
    return fidelity_plus, fidelity_minus

# Fidelity+  metric  studies  the  prediction  change  by
# removing  important  nodes/edges/node  features.
# Higher fidelity+ value indicates good explanations -->1
def fidelity_gnn_prob(orig_prob, prob_maskout):
    drop_probability = np.abs(orig_prob - prob_maskout)
    return drop_probability.mean().item()


# Fidelity-  metric  studies  the  prediction  change  by
# removing  unimportant  nodes/edges/node  features.
# Lower fidelity- value indicates good explanations -->0
def fidelity_gnn_prob_inv(orig_prob, prob_masked):
    drop_probability = np.abs(orig_prob - prob_masked)
    return drop_probability.mean().item()

def fidelity_gnn_acc(orig_prob, prob_maskout):
    orig_labels = (orig_prob > 0.5).astype(int)
    maskout_labels = (prob_maskout > 0.5).astype(int)
    drop_probability = np.abs(orig_labels - maskout_labels)
    return drop_probability.mean().item()

def fidelity_gnn_acc_inv(orig_prob, prob_masked):
    orig_labels = (orig_prob > 0.5).astype(int)
    maskout_labels = (prob_masked > 0.5).astype(int)
    drop_probability = np.abs(orig_labels - maskout_labels)
    return drop_probability.mean().item()

def cal_sparsity(mask, format:str):
    p = 0
    total = 0
    for _, e_mask in mask.items():
        if format == 'prob':
            p += e_mask.cpu().sum().item()
        elif format == 'acc':
            p += (e_mask > 0.5).cpu().sum().item()    
        total  += torch.numel(e_mask)
    return 1 - p/total

def cal_rankings(score:torch.tensor, score_all:torch.tensor, favor_min=False):
    '''
        require torch tensor as the inputs
    '''
    if not isinstance(score, torch.Tensor):
        score = torch.tensor(score)
    if not isinstance(score_all, torch.Tensor):
        score_all = torch.tensor(score_all)
    if not favor_min:
        ranking = (torch.gt(score_all, score) | torch.isclose(score_all, score)).sum()
    else:
        ranking = (torch.le(score_all, score) | torch.isclose(score_all, score)).sum()
    return ranking.item()

def cal_ranking_diff(a, b):
    '''
        a is considered to have larger ranking than b
    '''
    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)
    return np.mean((a-b))

def cal_ranking_drop_hit(a, b):
    '''
        a is considered to have larger ranking than b
    '''
    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)
    return (a > b).sum() / len(a)
