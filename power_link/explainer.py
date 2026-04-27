import logging
from collections import defaultdict
from time import time
from typing import Union

import dgl
import dgl.sparse as dsp
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .eval_metrics import cal_sparsity
from .utils import (
    cuda_usage,
    get_homo_nids_to_ntype_hetero_nids,
    get_neg_path_score_func,
    get_ntype_hetero_nids_to_homo_nids,
    get_ntype_pairs_to_cannonical_etypes,
    hetero_src_tgt_khop_in_subgraph,
    k_shortest_paths_with_max_length,
)

# setup logger
logger = logging.getLogger(__name__)

class DebugPrinter:
    """Lightweight conditional printer used for development tracing.

    Switched OFF by default in the open-source release. Enable per-process by
    calling ``dprint.switch_on()`` (or via a future ``--verbose`` flag)."""

    def __init__(self, enabled: bool = False):
        self.print = enabled

    def switch_off(self):
        self.print = False

    def switch_on(self):
        self.print = True

    def __call__(self, *s):
        if self.print:
            print(s)


dprint = DebugPrinter(enabled=False)


def get_edge_mask_dict(ghetero):
    '''
    Create a dictionary mapping etypes to learnable edge masks 
            
    Parameters
    ----------
    ghetero : heterogeneous dgl graph.

    Return
    ----------
    edge_mask_dict : dictionary
        key=etype, value=torch.nn.Parameter with size number of etype edges
    '''
    device = ghetero.device
    edge_mask_dict = {}
    for etype in ghetero.canonical_etypes:
        num_edges = ghetero.num_edges(etype)
        num_nodes = ghetero.edge_type_subgraph([etype]).num_nodes()
        std = torch.nn.init.calculate_gain('relu') * np.sqrt(2.0 / (2 * num_nodes))
        edge_mask_dict[etype] = torch.nn.Parameter(torch.randn(num_edges, device=device) * std)
    return edge_mask_dict

def get_kgnids(subg, etype):
    '''
        Get the nids in original KG for nodes connected by each edge in the subgraph.
        The number of src nodes or tgt nodes equal to the number of edges in the subgraph.
    '''
    src_nodes, tgt_nodes = subg.edges(etype=etype, form='uv')[0], subg.edges(etype=etype, form='uv')[1]
    kgnids_mapping = subg.nodes[:].data['_ID']
    src_kgnids, tgt_kgnids = kgnids_mapping[src_nodes], kgnids_mapping[tgt_nodes]
    return src_kgnids, tgt_kgnids

def remove_edges_of_high_degree_nodes(ghomo, max_degree=10, always_preserve=[]):
    '''
    For all the nodes with degree higher than `max_degree`, 
    except nodes in `always_preserve`, remove their edges. 
    
    Parameters
    ----------
    ghomo : dgl homogeneous graph
    
    max_degree : int
    
    always_preserve : iterable
        These nodes won't be pruned.
    
    Returns
    -------
    low_degree_ghomo : dgl homogeneous graph
        Pruned graph with edges of high degree nodes removed

    '''
    d = ghomo.in_degrees()
    high_degree_mask = d > max_degree
    
    # preserve nodes
    high_degree_mask[always_preserve] = False    

    high_degree_nids = ghomo.nodes()[high_degree_mask]
    u, v = ghomo.edges()
    high_degree_edge_mask = torch.isin(u, high_degree_nids) | torch.isin(v, high_degree_nids)
    high_degree_u, high_degree_v = u[high_degree_edge_mask], v[high_degree_edge_mask]
    high_degree_eids = ghomo.edge_ids(high_degree_u, high_degree_v)
    low_degree_ghomo = dgl.remove_edges(ghomo, high_degree_eids)
    
    return low_degree_ghomo


def remove_edges_except_k_core_graph(ghomo, k, always_preserve=[]):
    '''
    Find the `k`-core of `ghomo`.
    Only isolate the low degree nodes by removing theirs edges
    instead of removing the nodes, so node ids can be kept.
    
    Parameters
    ----------
    ghomo : dgl homogeneous graph
    
    k : int
    
    always_preserve : iterable
        These nodes won't be pruned.
    
    Returns
    -------
    k_core_ghomo : dgl homogeneous graph
        The k-core graph
    '''
    k_core_ghomo = ghomo
    degrees = k_core_ghomo.in_degrees()
    k_core_mask = (degrees > 0) & (degrees < k)
    k_core_mask[always_preserve] = False
    
    while k_core_mask.any():
        k_core_nids = k_core_ghomo.nodes()[k_core_mask]
        
        u, v = k_core_ghomo.edges()
        k_core_edge_mask = torch.isin(u, k_core_nids) | torch.isin(v, k_core_nids)
        k_core_u, k_core_v = u[k_core_edge_mask], v[k_core_edge_mask]
        k_core_eids = k_core_ghomo.edge_ids(k_core_u, k_core_v)

        k_core_ghomo = dgl.remove_edges(k_core_ghomo, k_core_eids)
        
        degrees = k_core_ghomo.in_degrees()
        k_core_mask = (degrees > 0) & (degrees < k)
        k_core_mask[always_preserve] = False

    return k_core_ghomo

def get_eids_on_paths(paths, ghomo):
    '''
    Collect all edge ids on the paths
    
    Note: The current version is a list version. An edge may be collected multiple times
    A different version is a set version where an edge can only contribute one time 
    even it appears in multiple paths
    
    Parameters
    ----------
    ghomo : dgl homogeneous graph
    
    Returns
    -------
    paths: list of lists
        Each list contains (source node ids, target node ids)
        
    '''
    row, col = ghomo.edges()
    eids = []
    for path in paths:
        for i in range(len(path)-1):
            eids += ((row == path[i]) & (col == path[i+1])).nonzero().squeeze(dim=1).tolist()    
    return torch.LongTensor(eids)

class PowerLinkExplainer(nn.Module):
    """Power-Link: a path-based explainer for GNN-based knowledge-graph completion.

    Implements the path-enforcing learning trick described in:

        Chang, Ye, Lopez-Avila, Du, Li.
        "Path-based Explanation for Knowledge Graph Completion." KDD '24.

    The class also retains the legacy PaGE-Link (WWW '23) Dijkstra-based path
    loss as a baseline; select it via ``path_loss="pagelink"`` (default
    ``"power"``).

    Some lower-level pieces are adapted from the DGL GNNExplainer:
    https://docs.dgl.ai/en/0.8.x/_modules/dgl/nn/pytorch/explain/gnnexplainer.html#GNNExplainer
    
    Parameters
    ----------
    model : nn.Module
        The GNN-based link prediction model to explain.

        * The required arguments of its forward function are source node id, target node id,
          graph, and feature ids. The feature ids are for selecting input node features.
        * It should also optionally take an eweight argument for edge weights
          and multiply the messages by the weights during message passing.
        * The output of its forward function is the logits in (-inf, inf) for the 
          predicted link.
    lr : float, optional
        The learning rate to use, default to 0.01.
    num_epochs : int, optional
        The number of epochs to train.
    alpha1 : float, optional
        A higher value will make the explanation edge masks more sparse by decreasing
        the sum of the edge mask.
    alpha2 : float, optional
        A higher value will make the explanation edge masks more discrete by decreasing
        the entropy of the edge mask.
    alpha : float, optional
        A higher value will make edges on high-quality paths to have higher weights
    beta : float, optional
        A higher value will make edges off high-quality paths to have lower weights
    log : bool, optional
        If True, it will log the computation process, default to True.
    """
    def __init__(self,
                 model,
                 lr=0.001,
                 num_epochs=100,
                 alpha=1.0,
                 beta=1.0,
                 log=False,
                 embed_dim=200,
                 parameterizer_hidden_dim=64):
        super(PowerLinkExplainer, self).__init__()
        self.model = model
        self.embed_dim = embed_dim
        self.parameterizer_hidden_dim = parameterizer_hidden_dim
        self.src_ntype = model.src_ntype
        self.tgt_ntype = model.tgt_ntype
        
        self.lr = lr
        self.num_epochs = num_epochs
        self.alpha = alpha
        self.beta = beta
        self.log = log
        self.all_loss = defaultdict(list)


    def get_mask_parameterizer(self):
        return nn.Sequential(
            nn.Linear(self.parameterizer_input_dim, self.parameterizer_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.parameterizer_hidden_dim, self.parameterizer_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.parameterizer_hidden_dim, 1)
        )


    def _init_masks_pagelink(self, ghetero):
        """Initialize the learnable edge mask.

        Parameters
        ----------
        graph : DGLGraph
            Input graph.

        Returns
        -------
        edge_mask_dict : dict
            key=`etype`, value=torch.nn.Parameter with size being the number of `etype` edges
        """
        ret = get_edge_mask_dict(ghetero)
        return ret

    def combine_embds(self, src_embds, tgt_embds, edge_embds, pred_src_embd, pred_tgt_embd, pred_rel_embd, method='concat'):
        if method == 'concat':
            self.parameterizer_input_dim = self.embed_dim * 6
            combined = torch.cat([src_embds, tgt_embds, edge_embds, pred_src_embd, pred_tgt_embd, pred_rel_embd], dim=1)
        elif method == 'euclidean':
            self.parameterizer_input_dim = 3
            combined = torch.stack(
                [
                    (src_embds - pred_src_embd).norm(dim=-1), 
                    (tgt_embds - pred_tgt_embd).norm(dim=-1), 
                    (edge_embds - pred_rel_embd).norm(dim=-1)
                ], 
                dim=-1
                )
        else:
            raise NotImplementedError(f"The combine method [{method}] is not supported!")
        
        return combined


    
    def get_edge_weight_features(self, node_embds, rel_embds, edge_types, ghetero, src_nid, tgt_nid, rel_id, combination_method='concat'):
        edge_weight_features = {}
        for etype in ghetero.canonical_etypes:
            subg = dgl.edge_type_subgraph(ghetero, [etype])
            src_kgnids, tgt_kgnids = get_kgnids(subg, etype)
            kg_eids = subg.edges[:].data['_ID']
            # * Get etypes of each edge
            # * etypes are used to get relation embeddings, not eids
            kg_etypes = edge_types[kg_eids] 
            pred_src_kgid, pred_tgt_kgid = subg.srcnodes[src_nid].data['_ID'], subg.dstnodes[tgt_nid].data['_ID']

            src_embds = node_embds[src_kgnids,:]
            tgt_embds = node_embds[tgt_kgnids,:]
            edge_embds = rel_embds[kg_etypes,:]

            pred_src_embd = node_embds[pred_src_kgid,:].repeat(src_embds.shape[0], 1)
            pred_tgt_embd = node_embds[pred_tgt_kgid,:].repeat(src_embds.shape[0], 1)
            pred_rel_embd = rel_embds[rel_id,:].repeat(src_embds.shape[0], 1)
            weight_features = self.combine_embds(
                src_embds, 
                tgt_embds, 
                edge_embds, 
                pred_src_embd, 
                pred_tgt_embd, 
                pred_rel_embd, 
                method=combination_method
                )
            edge_weight_features[etype] = weight_features
        return edge_weight_features

    def update_masks(self, mask_parameterizer, edge_weight_features):
        edge_mask_dict = {}
        for etype in edge_weight_features.keys():
            edge_mask_dict[etype] = mask_parameterizer(edge_weight_features[etype]).squeeze()
        return edge_mask_dict
        
    def _prune_graph(self, ghetero, prune_max_degree=-1, k_core=2, always_preserve=[]):
        # Prune edges by (optionally) removing edges of high degree nodes and extracting k-core
        # The pruning is computed on the homogeneous graph, i.e., ignoring node/edge types
        ghomo = dgl.to_homogeneous(ghetero)
        device = ghetero.device
        ghomo.edata['eid_before_prune'] = torch.arange(ghomo.num_edges()).to(device)
        
        if prune_max_degree > 0:
            max_degree_pruned_ghomo = remove_edges_of_high_degree_nodes(ghomo, prune_max_degree, always_preserve)
            k_core_ghomo = remove_edges_except_k_core_graph(max_degree_pruned_ghomo, k_core, always_preserve)
            
            if k_core_ghomo.num_edges() <= 0: # no k-core found
                pruned_ghomo = max_degree_pruned_ghomo
            else:
                pruned_ghomo = k_core_ghomo
        else:
            k_core_ghomo = remove_edges_except_k_core_graph(ghomo, k_core, always_preserve)
            if k_core_ghomo.num_edges() <= 0: # no k-core found
                pruned_ghomo = ghomo
            else:
                pruned_ghomo = k_core_ghomo
        
        pruned_ghomo_eids = pruned_ghomo.edata['eid_before_prune']
        pruned_ghomo_eid_mask = torch.zeros(ghomo.num_edges()).bool()
        pruned_ghomo_eid_mask[pruned_ghomo_eids] = True

        # Apply the pruning result on the heterogeneous graph
        etypes_to_pruned_ghetero_eid_masks = {}
        pruned_ghetero = ghetero
        cum_num_edges = 0
        for etype in ghetero.canonical_etypes:
            num_edges = ghetero.num_edges(etype=etype)
            pruned_ghetero_eid_mask = pruned_ghomo_eid_mask[cum_num_edges:cum_num_edges+num_edges]
            etypes_to_pruned_ghetero_eid_masks[etype] = pruned_ghetero_eid_mask

            remove_ghetero_eids = (~ pruned_ghetero_eid_mask).nonzero().view(-1).to(device)
            pruned_ghetero = dgl.remove_edges(pruned_ghetero, eids=remove_ghetero_eids, etype=etype)

            cum_num_edges += num_edges
                
        return pruned_ghetero, etypes_to_pruned_ghetero_eid_masks

    def ele_div(self, a:torch.tensor, b:torch.tensor):
        assert torch.all(a.indices().eq(b.indices())), \
        'a and b should share the same sparsity!'

        values = a.values()/b.values()
        ret = torch.sparse_coo_tensor(
            indices=a.indices(),
            values=values,
            size=a.size(),
            device=a.device
        )
        return ret
    
    def powerlink_path_loss(self, homo_src_nid, homo_tgt_nid, ml_ghomo, ml_ghomo_eweights, power_order=3):
            '''
                Implementation of the path loss based on the power of adjacency matrix. 
                BCE loss.
                The adj-related matrices are all represented as sparse COO matrices.
            '''

            # * Create a weighted sparse adjacency matrix representing M, which contains the learnable parameters
            # * The row/column index of the adj matrix correponds to the node id in the homogeneous graph (e.g. adj[i][j] -> node_i to node_j)
            
            def sparse_index_select(m:torch.Tensor, index:Union[tuple, list, torch.Tensor]):
                '''Get the values of the sparse matrix at the given position.
                Args:
                    m: torch sparse COO matrix
                    position: tuple/list of 2 elements
                Returns:
                    torch.Tensor: the value at the given position of the matrix.
                '''
                if not isinstance(index, torch.Tensor):
                    index = torch.tensor(index, device=m.device, requires_grad=False)
                m = m.coalesce()
                indices = m.indices()
                values = m.values()
                bools = indices.eq(index.unsqueeze(-1)).all(dim=0)
                if bools.sum() == 0.0:
                    # when the position is not in the sparse matrix but in the dense adj matrix
                    ret = torch.tensor([0.0], device=m.device)
                else:
                    ret = values[bools]
                return ret
            
            def get_sparse_row(m:torch.Tensor, row:int, reset_row_indices=True):
                '''Get the indices and values of a given row in the sparse matrix
                '''
                m = m.coalesce()
                indices = m.indices()
                values = m.values()

                col_bools = indices[0,:].eq(row)
                row_indices = indices[:, col_bools]
                row_values = values[col_bools]

                if reset_row_indices:
                    row_indices[0,:] = torch.zeros(row_values.shape, device = m.device)
                return row_indices, row_values

            device = ml_ghomo.device
            # Build the (indices, values) pair directly from the graph's edge list
            # rather than ``ml_ghomo.adj()``. DGL 1.x pre-coalesces ``g.adj()``,
            # which loses 1-to-1 alignment with ``ml_ghomo.edata['eweight']``
            # (the source of ``ml_ghomo_eweights``). Using ``g.edges()`` keeps
            # the per-edge alignment intact; the ``.coalesce()`` below sums
            # duplicate edges' weights, which is the correct semantics for the
            # weighted adjacency.
            src_nodes, dst_nodes = ml_ghomo.edges()
            adj_indices = torch.stack([src_nodes, dst_nodes])
            num_nodes = ml_ghomo.number_of_nodes()
            adj_shape = (num_nodes, num_nodes)
            adj_values = torch.ones(adj_indices.shape[1], device=device)

            # * Create the weighted adj matrix
            weighted_adj = torch.sparse_coo_tensor(
                indices=adj_indices,
                values=ml_ghomo_eweights,
                size=adj_shape,
                device=device
            ).coalesce()
            # * Get the source row in the weighted adj matrix
            row_indices, row_values = get_sparse_row(weighted_adj, homo_src_nid, reset_row_indices=True)
            row_weighted = torch.sparse_coo_tensor(
                indices=row_indices,
                values=row_values,
                size=[1, adj_shape[1]],
                device=device
            ).coalesce()

            # * Coalesce the adj matrix using torch_sparse (dgl.sparse.coalesce does not support torch.grad!)
            # * The coalescing process will accumulate the non-zero elements of the same indices by summation.
            # * Thus the weights will be summed up for the edges on the same path
            plain_adj = torch.sparse_coo_tensor(
                indices=adj_indices,
                values=adj_values,
                size=adj_shape,
                device=device
            ).coalesce()
            row_plain = torch.sparse_coo_tensor(
                indices=row_indices,
                values=torch.ones(row_values.shape, device=device),
                size=[1, adj_shape[1]],
                device=device
            ).coalesce()
            
            power_row_weighted = row_weighted
            power_row_padj = row_plain
            p_on_path = 0

            for i in range(power_order-1):
                power_row_weighted = torch.sparse.mm(power_row_weighted, weighted_adj).coalesce()
                power_row_padj = torch.sparse.mm(power_row_padj, plain_adj).coalesce()
                
                dprint(f'Number of total paths:{power_row_padj[0, homo_tgt_nid]}')
                # * Normalize by the number of paths: elementwise division
                # * Sparse matrix division only acts on non-zero values. So we don't need to care division-by-zeros
                power_normalized = self.ele_div(power_row_weighted, power_row_padj)

                at = sparse_index_select(power_normalized, [0, homo_tgt_nid])
                path_sum = torch.pow(at, 1.0/(i+2))
                p_on_path += path_sum #

            p_on_path = p_on_path/(i+1)
            loss = -p_on_path.log()
            self.all_loss['loss_on_path'] += [p_on_path]
            return loss


    def pagelink_path_loss(self, src_nid, tgt_nid, g, eweights, num_paths=4):
        """Compute the path loss.

        Parameters
        ----------
        src_nid : int
            source node id

        tgt_nid : int
            target node id

        g : dgl graph

        eweights : Tensor
            Edge weights with shape equals the number of edges.
            
        num_paths : int
            Number of paths to compute path loss on

        Returns
        -------
        loss : Tensor
            The path loss
        """
        neg_path_score_func = get_neg_path_score_func(g, 'eweight', [src_nid, tgt_nid])
        paths = k_shortest_paths_with_max_length(g, 
                                                 src_nid, 
                                                 tgt_nid, 
                                                 weight=neg_path_score_func, 
                                                 k=num_paths)

        eids_on_path = get_eids_on_paths(paths, g)

        if eids_on_path.nelement() > 0:
            loss_on_path = - eweights[eids_on_path].mean()
        else:
            loss_on_path = 0

        eids_off_path_mask = ~torch.isin(torch.arange(eweights.shape[0]), eids_on_path)
        if eids_off_path_mask.any():
            loss_off_path = eweights[eids_off_path_mask].mean()
        else:
            loss_off_path = 0

        loss = self.alpha * loss_on_path + self.beta * loss_off_path

        self.all_loss['loss_on_path'] += [float(loss_on_path)]
        self.all_loss['loss_off_path'] += [float(loss_off_path)]
        return loss   

    def get_edge_mask(self, 
                      src_nid, 
                      tgt_nid, 
                      rel_id,
                      ghetero, 
                      feat_nids, 
                      ghetero_full,
                      prune_max_degree=-1,
                      k_core=2, 
                      prune_graph=True,
                      without_path_loss=False,
                      without_mi=False,
                      regularisation_weight=0.001,
                      power_order=3,
                      pagelink=False,
                      combination_method='concat',
                      **kwargs,
                      ):

        """Learning the edge mask dict.   
        
        Parameters
        ----------
        see the `explain` method.
        
        Returns
        -------
        edge_mask_dict : dict
            key=`etype`, value=torch.nn.Parameter with size being the number of `etype` edges
        """

        self.model.eval()
        device = ghetero.device
        analysis_dict = kwargs.get('analysis_dict')
        analysis = analysis_dict is not None

        ntype_hetero_nids_to_homo_nids = get_ntype_hetero_nids_to_homo_nids(ghetero)    
        homo_src_nid = ntype_hetero_nids_to_homo_nids[(self.src_ntype, int(src_nid))]
        homo_tgt_nid = ntype_hetero_nids_to_homo_nids[(self.tgt_ntype, int(tgt_nid))]

        # Get the initial prediction.
        with torch.no_grad():
            orig_score = self.model(src_nid, rel_id, tgt_nid ,ghetero)
            # pred = (orig_score > 0.5).int().item() # * score has passed sigmoid, so the threshold should be 0.5
            embed_only_score = self.model(src_nid, rel_id, tgt_nid ,ghetero, message_passing=False)
        if prune_graph:
            # The pruned graph for mask learning  
            ml_ghetero, etypes_to_pruned_ghetero_eid_masks = self._prune_graph(ghetero, 
                                                                               prune_max_degree,
                                                                               k_core,
                                                                               [homo_src_nid, homo_tgt_nid])
        else:
            # The original graph for mask learning  
            ml_ghetero = ghetero

        # initialize the mask M
        if not pagelink:
            # * Run the GNN on the original graph to get the embeddings (we can use any src_nid, rel_id, tgt_id this time)
            _, node_embds, rel_embds, _ = self.model(src_nid, rel_id, tgt_nid, ghetero_full, return_embds=True)
            # * Prepare the edge embedding features
            edge_weight_features = self.get_edge_weight_features(node_embds,
                                                rel_embds,
                                                self.model.edge_type,
                                                ml_ghetero,
                                                src_nid, 
                                                tgt_nid, 
                                                rel_id,
                                                combination_method = combination_method
                                                )
            
            # * Initialize the parameterizer
            mask_parameterizer = self.get_mask_parameterizer().to(ghetero.device)
            optimizer = torch.optim.Adam(mask_parameterizer.parameters(), lr=self.lr)
            mask_parameterizer.train()
        else:
            ml_edge_mask_dict = self._init_masks_pagelink(ml_ghetero)
            optimizer = torch.optim.Adam(ml_edge_mask_dict.values(), lr=self.lr)
        
        if self.log:
            pbar = tqdm(total=self.num_epochs)
        
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device=device)
        # * Training
        for e in range(self.num_epochs):
            # Apply sigmoid to edge_mask to get eweight
            if not pagelink:
                ml_edge_mask_dict = self.update_masks(mask_parameterizer, edge_weight_features)
            ml_eweight_dict = {etype: ml_edge_mask_dict[etype].sigmoid() for etype in ml_edge_mask_dict}
            ml_counter_dict = {etype: 1.0 - ml_edge_mask_dict[etype].sigmoid() for etype in ml_edge_mask_dict}

            score = self.model(src_nid, rel_id, tgt_nid, ml_ghetero, ml_eweight_dict)

            # * Logging for observations
            score_counter = self.model(src_nid, rel_id, tgt_nid, ml_ghetero, ml_counter_dict)
            sparsity = cal_sparsity(ml_eweight_dict, 'prob')

            if e == 0:
                initial_score = score
                initial_score_counter = score_counter
                initial_sparsity = sparsity

            dprint('-'*100)
            dprint(f'Graph used during training: {ml_ghetero}')
            dprint(f'Embedding-only score:', embed_only_score)
            dprint(f'original score: ', orig_score)
            dprint('score: ', score)
            dprint('counter_score: ', score_counter)
            dprint(f'sparsity: {sparsity}')
            dprint(f'num_nodes: {ml_ghetero.number_of_nodes()}')
            dprint(f'num_edges: {ml_ghetero.number_of_edges()}')

            # * 'score' has already passed sigmoid, inside the 'sum' is the BCE loss of one single h+r-t.
            pred_loss = (-1) * score.log()
            self.all_loss['pred_loss'] += [pred_loss.item()]

            # * if the graph is homogeneous
            if len(list(ml_eweight_dict.keys())) == 1:
                ml_eweight_dict = list(ml_eweight_dict.values())[0]
            ml_ghetero.edata['eweight'] = ml_eweight_dict
            ml_ghomo = dgl.to_homogeneous(ml_ghetero, edata=['eweight'])
            ml_ghomo_eweights = ml_ghomo.edata['eweight']
            
            w = 0.8
            if without_path_loss:
                path_loss = torch.tensor(0.0).float()
                w = 1.0
            else: 
                if not pagelink:
                    path_loss = self.powerlink_path_loss(homo_src_nid, homo_tgt_nid, ml_ghomo, ml_ghomo_eweights, power_order)
                else:
                    path_loss = self.pagelink_path_loss(homo_src_nid, homo_tgt_nid, ml_ghomo, ml_ghomo_eweights)

            if without_mi:
                pred_loss = torch.tensor(0.0).float()
            
            loss = w * pred_loss + path_loss + regularisation_weight * ml_ghomo_eweights.norm()

            if device.type == 'cuda':
                cuda_mem_explain = torch.cuda.max_memory_allocated(device=device) // (1024**2)
                cuda_mem = cuda_usage()
                dprint(f'current total cuda usage: {cuda_mem}')
                dprint(f'cuda usage for explaining: {cuda_mem_explain}')

            dprint(f'Prediction loss: {pred_loss.item()}')
            dprint(f'Path loss: {path_loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.all_loss['total_loss'] += [loss.item()]

            if self.log:
                pbar.update(1)
        
        if analysis:
            assert torch.cuda.is_available(), 'Please make sure GPU is available for analysis.'
            analysis_dict['cuda_memory'].append(cuda_mem_explain)
            analysis_dict['num_nodes'].append(ml_ghetero.number_of_nodes())
            analysis_dict['num_edges'].append(ml_ghetero.number_of_edges())

        info = f'node_num:{ml_ghetero.number_of_nodes()}, edge_num:{ml_ghetero.number_of_edges()}, score_embedding_only:{embed_only_score.item() :.4f}, score_original:{orig_score.item() :.4f}, score_diff:{score.item() - initial_score.item() :.4f}, counter_score_diff:{score_counter.item() - initial_score_counter.item() :.4f}, sparsity_diff:{sparsity - initial_sparsity :.4f}, initial_score:{initial_score.item() :.4f}, final_score:{score.item() :.4f}, initial_counter_score:{initial_score_counter.item() :.4f}, final_counter_score:{score_counter.item() :.4f}, initial_sparsity:{initial_sparsity :.4f}, final_sparsity:{sparsity :.4f}'
        logger.info(info)
        if self.log:
            pbar.close()

        edge_mask_dict_placeholder = {etype:torch.zeros(ghetero.num_edges(etype), device=ghetero.device) for etype in ghetero.canonical_etypes}
        edge_mask_dict = {}
        
        if prune_graph:
            # remove pruned edges
            for etype in ghetero.canonical_etypes:
                edge_mask = edge_mask_dict_placeholder[etype].data + float('-inf')    
                pruned_ghetero_eid_mask = etypes_to_pruned_ghetero_eid_masks[etype]
                edge_mask[pruned_ghetero_eid_mask] = ml_edge_mask_dict[etype]
                edge_mask_dict[etype] = edge_mask
                
        else:
            edge_mask_dict = ml_edge_mask_dict
    
        edge_mask_dict = {k : v.detach() for k, v in edge_mask_dict.items()}
        return edge_mask_dict    

    def get_paths(self,
                  src_nid, 
                  tgt_nid, 
                  ghetero,
                  edge_mask_dict,
                  num_paths=1, 
                  max_path_length=3):

        """A postprocessing step that turns the `edge_mask_dict` into actual paths.
        
        Parameters
        ----------
        edge_mask_dict : dict
            key=`etype`, value=torch.nn.Parameter with size being the number of `etype` edges

        Others: see the `explain` method.
        
        Returns
        -------
        paths: list of lists
            each list contains (cannonical edge type, source node ids, target node ids)
        """
        ntype_pairs_to_cannonical_etypes = get_ntype_pairs_to_cannonical_etypes(ghetero)
        eweight_dict = {etype: edge_mask_dict[etype].sigmoid() for etype in edge_mask_dict}
        if len(list(eweight_dict.keys())) == 1:
            eweight_dict = list(eweight_dict.values())[0]
        ghetero.edata['eweight'] = eweight_dict

        # convert ghetero to ghomo and find paths
        ghomo = dgl.to_homogeneous(ghetero, edata=['eweight'])
        ntype_hetero_nids_to_homo_nids = get_ntype_hetero_nids_to_homo_nids(ghetero)    
        homo_src_nid = ntype_hetero_nids_to_homo_nids[(self.src_ntype, int(src_nid))]
        homo_tgt_nid = ntype_hetero_nids_to_homo_nids[(self.tgt_ntype, int(tgt_nid))]

        neg_path_score_func = get_neg_path_score_func(ghomo, 'eweight', [src_nid.item(), tgt_nid.item()])
        homo_paths = k_shortest_paths_with_max_length(ghomo, 
                                                       homo_src_nid, 
                                                       homo_tgt_nid,
                                                       weight=neg_path_score_func,
                                                       k=num_paths,
                                                       max_length=max_path_length)
        

        paths = []
        homo_nids_to_ntype_hetero_nids = get_homo_nids_to_ntype_hetero_nids(ghetero)
    
        if len(homo_paths) > 0:
            for homo_path in homo_paths:
                hetero_path = []
                for i in range(1, len(homo_path)):
                    homo_u, homo_v = homo_path[i-1], homo_path[i]
                    hetero_u_ntype, hetero_u_nid = homo_nids_to_ntype_hetero_nids[homo_u] 
                    hetero_v_ntype, hetero_v_nid = homo_nids_to_ntype_hetero_nids[homo_v] 
                    can_etype = ntype_pairs_to_cannonical_etypes[(hetero_u_ntype, hetero_v_ntype)]    
                    hetero_path += [(can_etype, hetero_u_nid, hetero_v_nid)]
                paths += [hetero_path]

        else:
            # A rare case, no paths found, take the top edges
            cat_edge_mask = torch.cat([v for v in edge_mask_dict.values()])
            M = len(cat_edge_mask)
            k = min(num_paths * max_path_length, M)
            threshold = cat_edge_mask.topk(k)[0][-1].item()
            path = []
            for etype in edge_mask_dict:
                u, v = ghetero.edges(etype=etype)  
                topk_edge_mask = edge_mask_dict[etype] >= threshold
                path += list(zip([etype] * topk_edge_mask.sum().item(), u[topk_edge_mask].tolist(), v[topk_edge_mask].tolist()))                
            paths = [path]
        return paths
    
    def explain(self,  
                src_nid, 
                tgt_nid,
                rel_id, 
                ghetero,
                num_hops=2,
                prune_max_degree=-1,
                k_core=2, 
                num_paths=1, 
                max_path_length=3,
                prune_graph=True,
                without_path_loss=False,
                without_mi=False,
                return_mask=False,
                regularisation_weight=0.001,
                power_order=3,
                comp_g_size_limit=1000,
                pagelink=False,
                combination_method='concat',
                **kwargs
                ):
        
        """Return a path explanation of a predicted link
        
        Parameters
        ----------
        src_nid : int
            source node id

        tgt_nid : int
            target node id

        ghetero : dgl graph

        num_hops : int
            Number of hops to extract the computation graph, i.e. GNN # layers
            
        prune_max_degree : int
            If positive, prune the edges of graph nodes with degree larger than `prune_max_degree`
            If  -1, do nothing
            
        k_core : int 
            k for the the k-core graph extraction
            
        num_paths : int
            Number of paths for the postprocessing path extraction
            
        max_path_length : int
            Maximum length of paths for the postprocessing path extraction
        
        prune_graph : bool
            If true apply the max_degree and/or k-core pruning. For ablation. Default True.
            
        without_path_loss : bool
            If true include the path loss. For ablation. Default True.
            
        return_mask : bool
            If true return the edge mask in addition to the path. For AUC evaluation. Default False
        
        Returns
        -------
        paths: list of lists
            each list contains (cannonical edge type, source node ids, target node ids)

        (optional) edge_mask_dict : dict
            key=`etype`, value=torch.nn.Parameter with size being the number of `etype` edges
        """
        # Extract the computation graph (k-hop subgraph)
        (comp_g_src_nid, 
         comp_g_tgt_nid, 
         comp_g, 
         comp_g_feat_nids) = hetero_src_tgt_khop_in_subgraph(self.src_ntype, 
                                                             src_nid, 
                                                             self.tgt_ntype, 
                                                             tgt_nid,
                                                             ghetero, 
                                                             num_hops
                                                             )

        if comp_g_size_limit == -1:
            # no size limit is imposed
            comp_g_size_limit = comp_g.number_of_nodes()
        if comp_g.number_of_nodes() > comp_g_size_limit:
            return None, None, None
        # * comp_g.ndata['_ID'], comp_g.edata['_ID']
        
        # Learn the edge mask on the computation graph
        comp_g_edge_mask_dict = self.get_edge_mask(comp_g_src_nid, 
                                                   comp_g_tgt_nid,
                                                   rel_id,
                                                   comp_g, 
                                                   comp_g_feat_nids,
                                                   ghetero,
                                                   prune_max_degree,
                                                   k_core,
                                                   prune_graph,
                                                   without_path_loss,
                                                   without_mi,
                                                   regularisation_weight,
                                                   power_order,
                                                   pagelink,
                                                   combination_method=combination_method,
                                                   **kwargs
                                                   )

        # Extract paths 
        comp_g_paths = self.get_paths(comp_g_src_nid,
                                      comp_g_tgt_nid, 
                                      comp_g, 
                                      comp_g_edge_mask_dict, 
                                      num_paths, 
                                      max_path_length)    
        
        return comp_g, comp_g_paths, comp_g_edge_mask_dict



