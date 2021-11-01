import numpy as np
import scipy.sparse as sparse
import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn
from dgl.base import DGLError
import dgl.function as fn
from dgl.nn.functional import edge_softmax

class GraphLSTMCell(nn.Module):
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

    def __init__(self, in_feats, out_feats, net, reuse_gate_rz):
        super(GraphLSTMCell, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.reuse_gate_rz = reuse_gate_rz

        # net can be any GNN model
        self.i_x_net = net(in_feats, out_feats)
        self.i_h_net = net(out_feats, out_feats)

        self.f_x_net = net(in_feats, out_feats)
        self.f_h_net = net(out_feats, out_feats)

        self.c_x_net = net(in_feats, out_feats)
        self.c_h_net = net(out_feats, out_feats)

        self.o_x_net = net(in_feats, out_feats)
        self.o_h_net = net(out_feats, out_feats)

        # c weights / TODO: try nn.Parameter(torch.Tensor(1, out_feats))
        self.weights_c_i = nn.Parameter(torch.rand(out_feats))
        self.weights_c_f = nn.Parameter(torch.rand(out_feats))
        self.weights_c_o = nn.Parameter(torch.rand(out_feats))

        # bias
        self.bias_i = nn.Parameter(torch.rand(out_feats))
        self.bias_f = nn.Parameter(torch.rand(out_feats))
        self.bias_c = nn.Parameter(torch.rand(out_feats))
        self.bias_o = nn.Parameter(torch.rand(out_feats))

    def forward(self, g, x, h, c, h_N=None):
        h_agg = None
        if self.reuse_gate_rz:
            # message passing
            with g.local_scope():
                g.ndata['x'] = h
                # update_all is a message passing API.
                g.update_all(message_func=fn.copy_u('x', 'm'), reduce_func=fn.mean('m', 'h_N'))
                h_agg = g.ndata['h_N']

            # even merge these two?
            if h_N is None:
                # message passing
                with g.local_scope():
                    g.ndata['x'] = x
                    # update_all is a message passing API.
                    g.update_all(message_func=fn.copy_u('x', 'm'), reduce_func=fn.mean('m', 'h_N'))
                    h_N = g.ndata['h_N']

        i = torch.sigmoid(
                self.i_x_net(g, x, h_N=h_N) 
                + self.i_h_net(g, h, h_N=h_agg)
                + self.weights_c_i * c 
                + self.bias_i)
        f = torch.sigmoid(
                self.f_x_net(g, x, h_N=h_N) 
                + self.f_h_net(g, h, h_N=h_agg)
                + self.weights_c_f * c 
                + self.bias_f) 
        c_ = torch.tanh(
                self.c_x_net(g, x, h_N=h_N) 
                + self.c_h_net(g, h, h_N=h_agg)
                + self.bias_c) 
        new_c = f * c + i * c_
        o = torch.sigmoid(
                self.o_x_net(g, x, h_N=h_N) 
                + self.o_h_net(g, h, h_N=h_agg)
                + self.weights_c_o * new_c 
                + self.bias_o) 
        new_h = o * torch.tanh(new_c)
        return new_h, new_c


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

    def __init__(self, in_feats, out_feats, num_layers, net, seq_len, agg_seq, reuse_gate_rz):
        super(StackedEncoder, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.net = net
        self.seq_len = seq_len
        self.agg_seq = agg_seq
        self.layers = nn.ModuleList()
        if self.num_layers <= 0:
            raise DGLError("Layer Number must be greater than 0! ")
        self.layers.append(GraphGRUCell(
            self.in_feats, self.out_feats, self.net, reuse_gate_rz))
        for _ in range(self.num_layers-1):
            self.layers.append(GraphGRUCell(
                self.out_feats, self.out_feats, self.net, reuse_gate_rz))

    # hidden_states should be a list which for different layer
    def forward(self, g, x, hidden_states, cell_states):
        h_Ns = None
        if self.agg_seq:
            x_split = torch.split(x, 1)
            x_cat = torch.squeeze(torch.cat(x_split, dim=-1))

            # message passing
            with g.local_scope():
                g.ndata['x'] = x_cat
                # update_all is a message passing API.
                g.update_all(message_func=fn.copy_u('x', 'm'), reduce_func=fn.mean('m', 'h_N'))
                h_N = g.ndata['h_N']

            h_Ns = torch.tensor_split(h_N, self.seq_len, dim=-1)

        for i in range(self.seq_len):
            input_ = x[i]
            if self.agg_seq:
                h_N = h_Ns[i]
            else:
                h_N = None
            hiddens = []
            c_states = []
            for j, layer in enumerate(self.layers):
                input_, c_state = layer(g, input_, hidden_states[j], cell_states[j], h_N=h_N)
                hiddens.append(input_)
                c_states.append(c_state)
                h_N = None
            hidden_states = hiddens
            cell_states = c_states
        return x, hidden_states, cell_states


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

    def __init__(self, in_feats, hid_feats, out_feats, num_layers, net, seq_len, agg_seq, reuse_gate_rz):
        super(StackedDecoder, self).__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.net = net
        self.seq_len = seq_len
        self.out_layer = nn.Linear(self.hid_feats, self.out_feats)
        self.agg_seq = agg_seq
        self.layers = nn.ModuleList()
        if self.num_layers <= 0:
            raise DGLError("Layer Number must be greater than 0!")
        self.layers.append(GraphGRUCell(self.in_feats, self.hid_feats, net, reuse_gate_rz))
        for _ in range(self.num_layers-1):
            self.layers.append(GraphGRUCell(
                self.hid_feats, self.hid_feats, net, reuse_gate_rz))

    def forward(self, g, x, hidden_states, cell_states):
        h_Ns = None
        if self.agg_seq:
            x_split = torch.split(x, 1)
            x_cat = torch.squeeze(torch.cat(x_split, dim=-1))

            # message passing
            with g.local_scope():
                g.ndata['x'] = x_cat
                # update_all is a message passing API.
                g.update_all(message_func=fn.copy_u('x', 'm'), reduce_func=fn.mean('m', 'h_N'))
                h_N = g.ndata['h_N']

            h_Ns = torch.tensor_split(h_N, self.seq_len, dim=-1)

        outputs = []
        for i in range(self.seq_len):
            input_ = x[i]
            if self.agg_seq:
                h_N = h_Ns[i]
            else:
                h_N = None
            hiddens = []
            c_states = []
            for j, layer in enumerate(self.layers):
                input_, c_state = layer(g, input_, hidden_states[j], h_N=h_N)
                hiddens.append(input_)
                c_states.append(c_state)
                h_N = None
            outputs.append(self.out_layer(input_))
            hidden_states = hiddens
            cell_states = c_states
        return outputs, hidden_states, cell_states

class GraphRNN(nn.Module):
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
                 agg_seq,
                 reuse_gate_rz):
        super(GraphRNN, self).__init__()
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
                                      agg_seq,
                                      reuse_gate_rz)

        self.decoder = StackedDecoder(self.in_feats,
                                      self.out_feats,
                                      self.in_feats,
                                      self.num_layers,
                                      self.net,
                                      self.seq_len,
                                      agg_seq,
                                      reuse_gate_rz)
    # Threshold For Teacher Forcing

    def compute_thresh(self, batch_cnt):
        return self.decay_steps/(self.decay_steps + np.exp(batch_cnt / self.decay_steps))

    def encode(self, g, inputs, device):
        hidden_states = [torch.zeros(g.num_nodes(), self.out_feats).to(
            device) for _ in range(self.num_layers)]

        cell_states = [torch.zeros(g.num_nodes(), self.out_feats).to(
            device) for _ in range(self.num_layers)]
        # for i in range(self.seq_len):
        #     _, hidden_states = self.encoder(g, inputs[i], hidden_states)

        _, hidden_states, cell_states = self.encoder(g, inputs, hidden_states, cell_states)

        return hidden_states, cell_states

    def decode(self, g, teacher_states, hidden_states, cell_states, batch_cnt, device):
        # outputs = []
        inputs = torch.zeros(self.seq_len, g.num_nodes(), self.in_feats).to(device)

        # if np.random.random() < self.compute_thresh(batch_cnt) and self.training:
        #     outputs, hidden_states = self.decoder(g, teacher_states, hidden_states)
        # else:
        outputs, _, _ = self.decoder(g, inputs, hidden_states, cell_states)

        outputs = torch.stack(outputs)
        return outputs

    def forward(self, g, inputs, teacher_states, batch_cnt, device):
        hidden_states, cell_states = self.encode(g, inputs, device)
        outputs = self.decode(g, teacher_states, hidden_states, cell_states, batch_cnt, device)
        return outputs
