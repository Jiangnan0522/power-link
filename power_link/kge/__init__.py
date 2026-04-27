"""KGE encoders / decoders / KG loaders used by both pretraining and explanation.

This package replaces the historical duplication between ``gcns/`` (used at
explanation time) and ``pretraining/model/`` (used during KGE training). The
layer modules support an optional ``eweight`` argument that the explainer
needs and that pretraining safely ignores (defaults to ``None``).
"""

import torch

from .compgcn_layer import CompGCNCov
from .config import Config, load_config
from .knowledge_graph import load_data
from .load_kg import build_graph, load_graph, process
from .models import GCN_ConvE, GCN_DistMult, GCN_TransE, GCNs
from .rgcn_layer import RelGraphConv
from .wgcn_layer import WGCNLayer

__all__ = [
    "CompGCNCov", "RelGraphConv", "WGCNLayer",
    "GCNs", "GCN_TransE", "GCN_DistMult", "GCN_ConvE",
    "Config", "load_config",
    "load_data", "load_graph", "build_graph", "process",
    "build_kgc_model_from_config",
]


def build_kgc_model_from_config(p, device, mp_g_meta):
    """Instantiate the (encoder, decoder) pair described by ``p``.

    ``p`` is the ``argparse.Namespace`` / ``Config`` produced by
    ``load_config``. ``mp_g_meta`` is the dict returned alongside the
    message-passing graph by ``load_graph`` (carries ``num_ent``, ``num_rels``,
    ``edge_type``, ``edge_norm``). Returns the un-wrapped model with parameters
    on ``device``; the caller is responsible for ``.eval()`` / freezing.
    """
    score_func = p.score_func.lower()
    common_kwargs = dict(
        num_ent=mp_g_meta['num_ent'],
        num_rel=mp_g_meta['num_rels'],
        num_base=getattr(p, 'num_base', -1),
        init_dim=p.init_dim,
        gcn_dim=p.gcn_dim,
        embed_dim=p.embed_dim,
        n_layer=p.n_layer,
        edge_type=mp_g_meta['edge_type'],
        edge_norm=mp_g_meta['edge_norm'],
        bias=getattr(p, 'bias', True),
        gcn_drop=getattr(p, 'gcn_drop', 0.),
        opn=getattr(p, 'opn', 'mult'),
        hid_drop=getattr(p, 'hid_drop', 0.),
        wni=getattr(p, 'wni', False),
        wsi=getattr(p, 'wsi', False),
        encoder=p.encoder,
        use_bn=getattr(p, 'use_bn', True),
        ltr=getattr(p, 'ltr', True),
    )

    if score_func == 'transe':
        model = GCN_TransE(gamma=getattr(p, 'gamma', 9.), **common_kwargs)
    elif score_func == 'distmult':
        model = GCN_DistMult(**common_kwargs)
    elif score_func == 'conve':
        model = GCN_ConvE(
            input_drop=getattr(p, 'input_drop', 0.),
            conve_hid_drop=getattr(p, 'conve_hid_drop', 0.),
            feat_drop=getattr(p, 'feat_drop', 0.),
            num_filt=getattr(p, 'num_filt', None),
            ker_sz=getattr(p, 'ker_sz', None),
            k_h=getattr(p, 'k_h', None),
            k_w=getattr(p, 'k_w', None),
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unknown score function: {score_func!r}. "
                         "Expected one of: 'transe', 'distmult', 'conve'.")

    return model.to(device)
