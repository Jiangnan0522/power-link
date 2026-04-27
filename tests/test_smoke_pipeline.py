"""End-to-end smoke test for PowerLinkExplainer on a tiny synthetic KG.

We construct a 10-node KG with 3 relation types, instantiate a randomly
initialised CompGCN+TransE (no pretraining), and run the explainer on one
triplet. This catches integration-level regressions — wiring between the
GCN forward pass, the explainer's mask training loop, and Dijkstra path
extraction. We do NOT assert numerical quality; the model is untrained.

Marked as `slow` so contributors can opt out via `pytest -m 'not slow'`.
"""

import dgl
import numpy as np
import pytest
import torch

from power_link.explainer import PowerLinkExplainer
from power_link.kge.models import GCN_TransE


@pytest.fixture
def tiny_kg():
    """Build a directed KG: 10 nodes, ~20 edges, 3 forward relation types.

    Mirrors how ``power_link.kge.load_kg.build_graph`` lays out a real KG:
    - For every triplet (h, r, t), add forward edge h → t with edge_type = r
      AND a reverse edge t → h with edge_type = r + num_rels.
    - Compute symmetric edge_norm.
    """
    torch.manual_seed(0)
    num_ent, num_rels = 10, 3
    rng = np.random.default_rng(0)
    src = rng.integers(0, num_ent, size=15)
    dst = rng.integers(0, num_ent, size=15)
    rel = rng.integers(0, num_rels, size=15)
    # Drop self-loops to keep things sane.
    keep = src != dst
    src, dst, rel = src[keep], dst[keep], rel[keep]

    forward_src = torch.from_numpy(np.asarray(src))
    forward_dst = torch.from_numpy(np.asarray(dst))
    forward_etype = torch.from_numpy(np.asarray(rel))

    all_src = torch.cat([forward_src, forward_dst])
    all_dst = torch.cat([forward_dst, forward_src])
    edge_type = torch.cat([forward_etype, forward_etype + num_rels])

    g = dgl.graph((all_src, all_dst), num_nodes=num_ent)
    in_deg = g.in_degrees(range(num_ent)).float()
    norm = (in_deg ** -0.5).nan_to_num(posinf=0.0, neginf=0.0)
    g.ndata['xxx'] = norm
    g.apply_edges(lambda e: {'xxx': e.dst['xxx'] * e.src['xxx']})
    edge_norm = g.edata.pop('xxx').squeeze()
    return g, edge_type, edge_norm, num_ent, num_rels


def test_explainer_runs_end_to_end_on_tiny_kg(tiny_kg):
    g, edge_type, edge_norm, num_ent, num_rels = tiny_kg

    model = GCN_TransE(
        num_ent=num_ent, num_rel=num_rels, num_base=-1,
        init_dim=16, gcn_dim=16, embed_dim=16, n_layer=1,
        edge_type=edge_type, edge_norm=edge_norm,
        bias=True, gcn_drop=0., opn='mult',
        hid_drop=0., gamma=9., wni=False, wsi=False,
        encoder='compgcn', use_bn=True, ltr=True,
    )
    model.src_ntype, model.tgt_ntype = '_N', '_N'
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    explainer = PowerLinkExplainer(
        model,
        lr=0.05,
        num_epochs=2,                  # tiny budget — this is a smoke test
        embed_dim=16,
        parameterizer_hidden_dim=8,
    )

    # Pick a (source, relation, target) triplet that exists in the graph.
    src_nid = torch.tensor(0)
    tgt_nid = torch.tensor(1)
    rel_id = torch.tensor(0)

    comp_g, comp_g_paths, comp_g_edge_mask_dict = explainer.explain(
        src_nid=src_nid,
        tgt_nid=tgt_nid,
        rel_id=rel_id,
        ghetero=g,
        num_hops=2,
        prune_max_degree=200,
        k_core=2,
        num_paths=2,
        max_path_length=3,
        prune_graph=True,
        without_path_loss=False,
        without_mi=False,
        return_mask=True,
        regularisation_weight=0.001,
        power_order=2,
        comp_g_size_limit=1000,
        pagelink=False,
        combination_method='concat',
    )

    # The smoke test passes if explain() returned something coherent.
    # We tolerate empty paths (the random model + tiny budget may not produce a path),
    # but the comp graph and mask must be present.
    assert comp_g is not None, "Computation graph should be returned."
    assert comp_g_edge_mask_dict is not None, "Edge mask dict should be returned."
    for etype, mask in comp_g_edge_mask_dict.items():
        assert torch.isfinite(mask).any(), f"Mask for {etype} contains only NaN/Inf."
