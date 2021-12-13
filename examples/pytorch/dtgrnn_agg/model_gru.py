import numpy as np
import scipy.sparse as sparse
import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn
from dgl.base import DGLError
import dgl.function as fn
from dgl.nn.functional import edge_softmax

class GraphGRUCell(nn.Module):
    '''Graph GRU unit which can use any message passing
    net to replace the linear layer in the original GRU
    Parameter
    ==========
    in_feats : int
        number of input features

    out_feats : int
        number of output features

    net : torch.nn.Module
        message passing network
    '''

    def __init__(self, in_feats, out_feats, net, reuse_msg_passing):
        super(GraphGRUCell, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.reuse_msg_passing = reuse_msg_passing

        # net can be any GNN model
        self.r_x_net = net(in_feats, out_feats)
        self.r_h_net = net(out_feats, out_feats)

        self.u_x_net = net(in_feats, out_feats)
        self.u_h_net = net(out_feats, out_feats)

        self.c_x_net = net(in_feats, out_feats)
        self.c_h_net = net(out_feats, out_feats)

    def forward(self, g, x, h, x_agg=None):
        h_agg = None
        if self.reuse_msg_passing:
            # message passing
            with g.local_scope():
                g.ndata['x'] = h
                # update_all is a message passing API.
                g.update_all(message_func=fn.copy_u('x', 'm'), reduce_func=fn.sum('m', 'h_N'))
                h_agg = g.ndata['h_N']

            # even merge these two?
            if x_agg is None:
                # message passing
                with g.local_scope():
                    g.ndata['x'] = x
                    # update_all is a message passing API.
                    g.update_all(message_func=fn.copy_u('x', 'm'), reduce_func=fn.sum('m', 'h_N'))
                    x_agg = g.ndata['h_N']

        r = torch.sigmoid(self.r_x_net(g, x, agg=x_agg) + self.r_h_net(g, h, agg=h_agg))
        u = torch.sigmoid(self.u_x_net(g, x, agg=x_agg) + self.u_h_net(g, h, agg=h_agg))
        h_ = r*h
        c = torch.tanh(self.c_x_net(g, x, agg=x_agg) + self.c_h_net(g, h_))
        new_h = u*h + (1-u)*c
        return new_h


class StackedEncoder(nn.Module):
    '''One step encoder unit for hidden representation generation
    it can stack multiple vertical layers to increase the depth.

    Parameter
    ==========
    in_feats : int
        number if input features

    out_feats : int
        number of output features

    num_layers : int
        vertical depth of one step encoding unit

    net : torch.nn.Module
        message passing network for graph computation
    '''

    def __init__(self, in_feats, out_feats, num_layers, net, seq_len, merge_time_steps, reuse_msg_passing):
        super(StackedEncoder, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.net = net
        self.seq_len = seq_len
        self.merge_time_steps = merge_time_steps
        self.layers = nn.ModuleList()
        if self.num_layers <= 0:
            raise DGLError("Layer Number must be greater than 0! ")
        self.layers.append(GraphGRUCell(
            self.in_feats, self.out_feats, self.net, reuse_msg_passing))
        for _ in range(self.num_layers-1):
            self.layers.append(GraphGRUCell(
                self.out_feats, self.out_feats, self.net, reuse_msg_passing))

    # hidden_states should be a list which for different layer
    def forward(self, g, x, hidden_states):
        x_aggs = None
        if self.merge_time_steps:
            x_split = torch.split(x, 1)
            x_cat = torch.squeeze(torch.cat(x_split, dim=-1))

            # message passing
            with g.local_scope():
                g.ndata['x'] = x_cat
                # update_all is a message passing API.
                g.update_all(message_func=fn.copy_u('x', 'm'), reduce_func=fn.sum('m', 'h_N'))
                x_agg = g.ndata['h_N']

            x_aggs = torch.chunk(x_agg, self.seq_len, dim=-1)

        for i in range(self.seq_len):
            input_ = x[i]
            if self.merge_time_steps:
                x_agg = x_aggs[i]
            else:
                x_agg = None
            hiddens = []
            for j, layer in enumerate(self.layers):
                input_ = layer(g, input_, hidden_states[j], x_agg=x_agg)
                hiddens.append(input_)
                x_agg = None # merging time steps can only be applied to first layer
            hidden_states = hiddens
        return x, hidden_states


class StackedDecoder(nn.Module):
    '''One step decoder unit for hidden representation generation
    it can stack multiple vertical layers to increase the depth.

    Parameter
    ==========
    in_feats : int
        number if input features

    hid_feats : int
        number of feature before the linear output layer

    out_feats : int
        number of output features

    num_layers : int
        vertical depth of one step encoding unit

    net : torch.nn.Module
        message passing network for graph computation
    '''

    def __init__(self, in_feats, hid_feats, out_feats, num_layers, net, seq_len, merge_time_steps, reuse_msg_passing):
        super(StackedDecoder, self).__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.net = net
        self.seq_len = seq_len
        self.out_layer = nn.Linear(self.hid_feats, self.out_feats)
        self.merge_time_steps = merge_time_steps
        self.layers = nn.ModuleList()
        if self.num_layers <= 0:
            raise DGLError("Layer Number must be greater than 0!")
        self.layers.append(GraphGRUCell(self.in_feats, self.hid_feats, net, reuse_msg_passing))
        for _ in range(self.num_layers-1):
            self.layers.append(GraphGRUCell(
                self.hid_feats, self.hid_feats, net, reuse_msg_passing))

    def forward(self, g, x, hidden_states):
        x_aggs = None
        if self.merge_time_steps:
            x_split = torch.split(x, 1)
            x_cat = torch.squeeze(torch.cat(x_split, dim=-1))

            # message passing
            with g.local_scope():
                g.ndata['x'] = x_cat
                # update_all is a message passing API.
                g.update_all(message_func=fn.copy_u('x', 'm'), reduce_func=fn.sum('m', 'h_N'))
                x_agg = g.ndata['h_N']

            x_aggs = torch.chunk(x_agg, self.seq_len, dim=-1)

        outputs = []
        for i in range(self.seq_len):
            input_ = x[i]
            if self.merge_time_steps:
                x_agg = x_aggs[i]
            else:
                x_agg = None
            hiddens = []
            for j, layer in enumerate(self.layers):
                input_ = layer(g, input_, hidden_states[j], x_agg=x_agg)
                hiddens.append(input_)
                x_agg = None
            outputs.append(self.out_layer(input_))
            hidden_states = hiddens
        return outputs, hidden_states

class GraphGRU(nn.Module):
    '''Graph Sequence to sequence prediction framework
    Support multiple backbone GNN. Mainly used for traffic prediction.

    Parameter
    ==========
    in_feats : int
        number of input features

    out_feats : int
        number of prediction output features

    seq_len : int
        input and predicted sequence length

    num_layers : int
        vertical number of layers in encoder and decoder unit

    net : torch.nn.Module
        Message passing GNN as backbone

    decay_steps : int
        number of steps for the teacher forcing probability to decay
    '''

    def __init__(self,
                 in_feats,
                 out_feats,
                 seq_len,
                 num_layers,
                 net,
                 decay_steps,
                 merge_time_steps,
                 reuse_msg_passing):
        super(GraphGRU, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.net = net
        self.decay_steps = decay_steps

        self.encoder = StackedEncoder(self.in_feats,
                                      self.out_feats,
                                      self.num_layers,
                                      self.net,
                                      self.seq_len,
                                      merge_time_steps,
                                      reuse_msg_passing)

        self.decoder = StackedDecoder(self.in_feats,
                                      self.out_feats,
                                      self.in_feats,
                                      self.num_layers,
                                      self.net,
                                      self.seq_len,
                                      merge_time_steps,
                                      reuse_msg_passing)
    # Threshold For Teacher Forcing

    def compute_thresh(self, batch_cnt):
        return self.decay_steps/(self.decay_steps + np.exp(batch_cnt / self.decay_steps))

    def encode(self, g, inputs, device):
        hidden_states = [torch.zeros(g.num_nodes(), self.out_feats).to(
            device) for _ in range(self.num_layers)]

        _, hidden_states = self.encoder(g, inputs, hidden_states)

        return hidden_states

    def decode(self, g, teacher_states, hidden_states, batch_cnt, device):
        # outputs = []
        inputs = torch.zeros(self.seq_len, g.num_nodes(), self.in_feats).to(device)

        outputs, hidden_states = self.decoder(g, inputs, hidden_states)

        outputs = torch.stack(outputs)
        return outputs

    def forward(self, g, inputs, teacher_states, batch_cnt, device):
        hidden = self.encode(g, inputs, device)
        outputs = self.decode(g, teacher_states, hidden, batch_cnt, device)
        return outputs