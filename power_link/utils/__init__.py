"""Utilities split out of the original 1k-line ``utils.py``.

Re-exports all callables so that ``from power_link.utils import X`` keeps
working regardless of which submodule ``X`` actually lives in.
"""

from .seeding import (
    COLORS,
    cuda_usage,
    eids_split,
    get_label,
    idx_split,
    load_pred_paths,
    negative_sampling,
    print_args,
    set_config_args,
    set_seed,
)
from .graph import (
    get_homo_nids_to_hetero_nids,
    get_homo_nids_to_ntype_hetero_nids,
    get_ntype_hetero_nids_to_homo_nids,
    get_ntype_pairs_to_cannonical_etypes,
    get_num_nodes_dict,
    hetero_src_tgt_khop_in_subgraph,
    remove_all_edges_of_etype,
)
from .paths import (
    PathBuffer,
    bidirectional_dijkstra,
    get_neg_path_score_func,
    k_shortest_paths_generator,
    k_shortest_paths_with_max_length,
)
from .eval import (
    eval_edge_mask_auc,
    eval_edge_mask_topk_path_hit,
    eval_hard_edge_mask_path_hit,
    eval_path_explanation_edges_path_hit,
    get_comp_g_edge_labels,
    get_comp_g_path_labels,
)

__all__ = [
    "COLORS", "cuda_usage", "eids_split", "get_label", "idx_split",
    "load_pred_paths", "negative_sampling", "print_args", "set_config_args",
    "set_seed",
    "get_homo_nids_to_hetero_nids", "get_homo_nids_to_ntype_hetero_nids",
    "get_ntype_hetero_nids_to_homo_nids", "get_ntype_pairs_to_cannonical_etypes",
    "get_num_nodes_dict", "hetero_src_tgt_khop_in_subgraph",
    "remove_all_edges_of_etype",
    "PathBuffer", "bidirectional_dijkstra", "get_neg_path_score_func",
    "k_shortest_paths_generator", "k_shortest_paths_with_max_length",
    "eval_edge_mask_auc", "eval_edge_mask_topk_path_hit",
    "eval_hard_edge_mask_path_hit", "eval_path_explanation_edges_path_hit",
    "get_comp_g_edge_labels", "get_comp_g_path_labels",
]
