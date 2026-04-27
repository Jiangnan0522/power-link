import os
import random
import logging
import argparse
import math
import numpy as np
import time
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from dgl import function as fn


class WGCNLayer(nn.Module):
    def __init__(self, in_features, out_features, num_relations, bias=False):
        super(WGCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.alpha = torch.nn.Embedding(num_relations, 1, padding_idx=0)
        self.bn = torch.nn.BatchNorm1d(out_features)

        self.num_relations = num_relations
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def message_func(self, edges):
        msg = edges.src['ft'] * edges.data['a']
        # apply mask to messages
        if '_edge_weight' in edges.data:
            if isinstance(edges.data['_edge_weight'], dict):
                # * We only have one type of nodes and edges in the graph for KG
                mask = list(edges.data['_edge_weight'].values())[0] 
            else:
                mask = edges.data['_edge_weight']
            msg = msg * mask.unsqueeze(-1)
        return {'m':msg}


    def forward(self, g, all_edge_type, input, eweight=None, message_passing=True):
        if eweight is not None:
        # * if the KG is in the homogeneous form
            if len(list(eweight.keys())) == 1:
                eweight = list(eweight.values())[0]
            g.edata['_edge_weight'] = eweight


        with g.local_scope():
            feats = torch.mm(input, self.weight)
            g.srcdata['ft'] = feats
            
            # Get the relation_ids of the inverse directions. 
            # The id of the inverse relation of relation id a is: a + total_num_relations
            num_relations_single_direction = int(self.num_relations / 2)
            transposed_all_edge_type = torch.where(
                all_edge_type >= num_relations_single_direction, \
                all_edge_type - num_relations_single_direction, \
                all_edge_type + num_relations_single_direction
                )
            alp = self.alpha(all_edge_type) + self.alpha(transposed_all_edge_type)
            g.edata['a'] = alp

            # g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            if message_passing:
                reduce_func = fn.sum('m', 'ft')
            else:
                reduce_func = lambda nodes: {'ft': nodes.data['ft']}
            g.update_all(self.message_func, reduce_func)

            output = g.dstdata['ft']
            output = self.bn(output) # Batchnorm
            # no dropout in the GCN layer since it is added outside
            if self.bias is not None:
                return output + self.bias
            else:
                return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'
