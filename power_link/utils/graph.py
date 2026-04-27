"""DGL heterogeneous-graph manipulation helpers used by the explainer."""

import dgl
import torch

# ``khop_in_subgraph`` lives at the top level of dgl from 0.9 onward;
# pre-0.9 only exposed it via ``dgl.subgraph.khop_in_subgraph``.
try:
    khop_in_subgraph = dgl.khop_in_subgraph
except AttributeError:  # pragma: no cover - legacy DGL
    from dgl.subgraph import khop_in_subgraph


def get_homo_nids_to_hetero_nids(ghetero):
    """Map homogeneous node ids of ``ghetero`` to its heterogeneous node ids."""
    ghomo = dgl.to_homogeneous(ghetero)
    homo_nids = range(ghomo.num_nodes())
    hetero_nids = ghomo.ndata[dgl.NID].tolist()
    return dict(zip(homo_nids, hetero_nids))


def get_homo_nids_to_ntype_hetero_nids(ghetero):
    """Map homogeneous node ids to ``(ntype, hetero_nid)`` tuples."""
    ghomo = dgl.to_homogeneous(ghetero)
    homo_nids = range(ghomo.num_nodes())
    ntypes = ghetero.ntypes
    ntypes = [ntypes[i] for i in ghomo.ndata[dgl.NTYPE]]
    hetero_nids = ghomo.ndata[dgl.NID].tolist()
    ntypes_hetero_nids = list(zip(ntypes, hetero_nids))
    return dict(zip(homo_nids, ntypes_hetero_nids))


def get_ntype_hetero_nids_to_homo_nids(ghetero):
    """Inverse of :func:`get_homo_nids_to_ntype_hetero_nids`."""
    tmp = get_homo_nids_to_ntype_hetero_nids(ghetero)
    return {v: k for k, v in tmp.items()}


def get_ntype_pairs_to_cannonical_etypes(ghetero, pred_etype='likes'):
    """Map ``(src_ntype, tgt_ntype)`` to canonical edge types, excluding ``pred_etype``."""
    ntype_pairs_to_cannonical_etypes = {}
    for src_ntype, etype, tgt_ntype in ghetero.canonical_etypes:
        if etype != pred_etype:
            ntype_pairs_to_cannonical_etypes[(src_ntype, tgt_ntype)] = (src_ntype, etype, tgt_ntype)
    return ntype_pairs_to_cannonical_etypes


def get_num_nodes_dict(ghetero):
    """Dict from ntype to number of nodes."""
    return {ntype: ghetero.num_nodes(ntype) for ntype in ghetero.ntypes}


def remove_all_edges_of_etype(ghetero, etype):
    """Remove every edge of type ``etype`` from ``ghetero`` (no-op if absent)."""
    etype = ghetero.to_canonical_etype(etype)
    if etype in ghetero.canonical_etypes:
        eids = ghetero.edges('eid', etype=etype)
        return dgl.remove_edges(ghetero, eids, etype=etype)
    return ghetero


def hetero_src_tgt_khop_in_subgraph(src_ntype, src_nid, tgt_ntype, tgt_nid, ghetero, k):
    """Extract the ``k``-hop subgraph centred on the union of ``{src_nid, tgt_nid}``.

    Returns ``(sg_src_nid, sg_tgt_nid, sg, sg_feat_nid)``: the new subgraph node
    ids of the source and target, the subgraph itself, and the original node ids
    so that node features can be looked up after the contraction.
    """
    src_nid = src_nid.item() if torch.is_tensor(src_nid) else src_nid
    tgt_nid = tgt_nid.item() if torch.is_tensor(tgt_nid) else tgt_nid

    device = ghetero.device
    if src_ntype == tgt_ntype:
        pred_dict = {src_ntype: torch.tensor([src_nid, tgt_nid]).to(device)}
        sghetero, inv_dict = khop_in_subgraph(ghetero, pred_dict, k)
        sghetero_src_nid = inv_dict[src_ntype][0]
        sghetero_tgt_nid = inv_dict[tgt_ntype][1]
    else:
        pred_dict = {src_ntype: src_nid, tgt_ntype: tgt_nid}
        sghetero, inv_dict = khop_in_subgraph(ghetero, pred_dict, k)
        sghetero_src_nid = inv_dict[src_ntype]
        sghetero_tgt_nid = inv_dict[tgt_ntype]

    sghetero_feat_nid = sghetero.ndata[dgl.NID]
    return sghetero_src_nid, sghetero_tgt_nid, sghetero, sghetero_feat_nid
