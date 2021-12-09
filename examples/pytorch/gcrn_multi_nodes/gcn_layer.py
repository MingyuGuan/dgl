import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, h, agg=None):
        if agg is None:
            with g.local_scope():
                g.ndata['h'] = h
                g.update_all(message_func=fn.copy_u('h', 'm'), reduce_func=fn.sum('m', 'h_N'))
                agg = g.ndata['h_N']
        
        return self.linear(h)
