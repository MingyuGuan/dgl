from functools import partial
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import dgl
# from model import GraphRNN
# from dcrnn import DiffConv
# from dcrnn import DiffConvAgg
# from gaan import GatedGAT
from model_gru import GraphGRU
from model_lstm import GraphLSTM
from sage_conv import SageConv
from gcn_layer import GCNLayer
from dataloading import METR_LAGraphDataset, METR_LATrainDataset,\
    METR_LATestDataset, METR_LAValidDataset,\
    PEMS_BAYGraphDataset, PEMS_BAYTrainDataset,\
    PEMS_BAYValidDataset, PEMS_BAYTestDataset
from utils import NormalizationLayer, masked_mae_loss, get_learning_rate

import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import os

batch_cnt = [0]


def train(model, graph, dataloader, optimizer, scheduler, normalizer, loss_fn, device, args):
    total_loss = []
    graph = graph.to(device)
    model.train()
    batch_size = args.batch_size
    total_batch = 0
    start = time.time()
    for i, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        # Padding: Since the diffusion graph is precmputed we need to pad the batch so that
        # each batch have same batch size
        if x.shape[0] != batch_size:
            x_buff = torch.zeros(
                batch_size, x.shape[1], x.shape[2], x.shape[3])
            y_buff = torch.zeros(
                batch_size, x.shape[1], x.shape[2], x.shape[3])
            x_buff[:x.shape[0], :, :, :] = x
            x_buff[x.shape[0]:, :, :,
                   :] = x[-1].repeat(batch_size-x.shape[0], 1, 1, 1)
            y_buff[:x.shape[0], :, :, :] = y
            y_buff[x.shape[0]:, :, :,
                   :] = y[-1].repeat(batch_size-x.shape[0], 1, 1, 1)
            x = x_buff
            y = y_buff
        # Permute the dimension for shaping
        # before: [batch_size, seq_len, num_nodes, in_feats]
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        # after: [seq_len, batch_size, num_nodes, in_feats]

        # replicate feats / nodes, if applicable
        if args.in_feats > 2:
            x_ = tuple(x for _ in range(int(args.in_feats / x.shape[3])))
            x = torch.cat(x_, dim=3)
            y_ = tuple(y for _ in range(int(args.in_feats / y.shape[3])))
            y = torch.cat(y_, dim=3)

        if args.replicate_nodes > 1:
            x_ = tuple(x for _ in range(args.replicate_nodes))
            x = torch.cat(x_, dim=2)
            y_ = tuple(y for _ in range(args.replicate_nodes))
            y = torch.cat(y_, dim=2)

        x_norm = normalizer.normalize(x).reshape(
            x.shape[0], -1, x.shape[3]).float().to(device)
        y_norm = normalizer.normalize(y).reshape(
            x.shape[0], -1, x.shape[3]).float().to(device)
        y = y.reshape(y.shape[0], -1, y.shape[3]).float().to(device)

        # after: [seq_len, batch_size * num_nodes, in_feats]

        batch_graph = dgl.batch([graph]*batch_size)
        output = model(batch_graph, x_norm, y_norm, batch_cnt[0], device)
        # Denormalization for loss compute
        y_pred = normalizer.denormalize(output)
        loss = loss_fn(y_pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        if get_learning_rate(optimizer) > args.minimum_lr:
            scheduler.step()
        total_loss.append(float(loss))
        batch_cnt[0] += 1
        total_batch = i
    end = time.time()
    snapshot_cnt = (total_batch + 1) * batch_size * 12
    throughput = snapshot_cnt / (end - start)
    print("Training throughput:", throughput, "snapshot/sec") 
    return np.mean(total_loss)


def eval(model, graph, dataloader, normalizer, loss_fn, device, args):
    total_loss = []
    graph = graph.to(device)
    model.eval()
    batch_size = args.batch_size
    for i, (x, y) in enumerate(dataloader):
        # Padding: Since the diffusion graph is precmputed we need to pad the batch so that
        # each batch have same batch size
        if x.shape[0] != batch_size:
            x_buff = torch.zeros(
                batch_size, x.shape[1], x.shape[2], x.shape[3])
            y_buff = torch.zeros(
                batch_size, x.shape[1], x.shape[2], x.shape[3])
            x_buff[:x.shape[0], :, :, :] = x
            x_buff[x.shape[0]:, :, :,
                   :] = x[-1].repeat(batch_size-x.shape[0], 1, 1, 1)
            y_buff[:x.shape[0], :, :, :] = y
            y_buff[x.shape[0]:, :, :,
                   :] = y[-1].repeat(batch_size-x.shape[0], 1, 1, 1)
            x = x_buff
            y = y_buff
        # Permute the order of dimension
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)

        # replicate feats / nodes, if applicable
        if args.in_feats > 2:
            x_ = tuple(x for _ in range(int(args.in_feats / x.shape[3])))
            x = torch.cat(x_, dim=3)
            y_ = tuple(y for _ in range(int(args.in_feats / y.shape[3])))
            y = torch.cat(y_, dim=3)

        if args.replicate_nodes > 1:
            x_ = tuple(x for _ in range(args.replicate_nodes))
            x = torch.cat(x_, dim=2)
            y_ = tuple(y for _ in range(args.replicate_nodes))
            y = torch.cat(y_, dim=2)

        x_norm = normalizer.normalize(x).reshape(
            x.shape[0], -1, x.shape[3]).float().to(device)
        y_norm = normalizer.normalize(y).reshape(
            x.shape[0], -1, x.shape[3]).float().to(device)
        y = y.reshape(x.shape[0], -1, x.shape[3]).to(device)

        batch_graph = dgl.batch([graph]*batch_size)
        output = model(batch_graph, x_norm, y_norm, i, device)
        y_pred = normalizer.denormalize(output)
        loss = loss_fn(y_pred, y)
        total_loss.append(float(loss))
    return np.mean(total_loss)

def spmd_main(local_rank, g, train_data, val_data, test_data, args, seed=0):
    #dist.init_process_group(
    #    backend='nccl', init_method='env://')
        #init_method='tcp://'+args.master_addr+':1234',
        #rank=args.rank,
        #world_size=args.world_size)
    
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    dist.init_process_group(backend="nccl")

    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print("Hostname:", hostname, "/ Hostip:", local_ip)

    device = local_rank
    torch.cuda.set_device(device)
    
    print("CUDA device:", torch.cuda.get_device_name(device))
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, num_workers=0, shuffle=False, pin_memory=True, sampler=train_sampler)
    valid_loader = DataLoader(
        valid_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    # Any problem?
    normalizer = NormalizationLayer(train_data.mean, train_data.std)
     
    # init model 
    torch.manual_seed(seed)
    
    if args.model == 'sage':
        net = SageConv
    elif args.model == 'gcn':
        net = GCNLayer

    if args.rnn == 'gru':
        graph_rnn = GraphGRU(in_feats=args.in_feats,
                         out_feats=64,
                         seq_len=12,
                         num_layers=args.num_layers,
                         net=net,
                         decay_steps=args.decay_steps,
                         merge_time_steps=args.merge_time_steps,
                         reuse_msg_passing=args.reuse_msg_passing).to(device)
    elif args.rnn == 'lstm':
        graph_rnn = GraphLSTM(in_feats=args.in_feats,
                         out_feats=64,
                         seq_len=12,
                         num_layers=args.num_layers,
                         net=net,
                         decay_steps=args.decay_steps,
                         merge_time_steps=args.merge_time_steps,
                         reuse_msg_passing=args.reuse_msg_passing).to(device)
    
    graph_rnn = DistributedDataParallel(graph_rnn, device_ids=[device], output_device=device)
    
    optimizer = torch.optim.Adam(graph_rnn.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    loss_fn = masked_mae_loss

    for e in range(args.epochs):
        train_sampler.set_epoch(e)
        start = time.time()
        train_loss = train(graph_rnn, g, train_loader, optimizer, scheduler,
                           normalizer, loss_fn, device, args)
        valid_loss = eval(graph_rnn, g, valid_loader,
                          normalizer, loss_fn, device, args)
        test_loss = eval(graph_rnn, g, test_loader,
                         normalizer, loss_fn, device, args)
        end = time.time()
        print("Epoch {}: Time: {} Train Loss: {} Valid Loss: {} Test Loss: {}".format(e,
                                                                             end-start,
                                                                             train_loss,
                                                                             valid_loss,
                                                                             test_loss))
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Define the arguments
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Size of batch for minibatch Training")
    parser.add_argument('--num_workers', type=int, default=0,
                        help="Number of workers for parallel dataloading")
    parser.add_argument('--model', type=str, default='sage',
                        help="WHich model to use SAGE vs GCN vs DCRNN vs GaAN")
    parser.add_argument('--gpu', type=int, default=-1,
                        help="GPU indexm -1 for CPU training")
    parser.add_argument('--diffsteps', type=int, default=2,
                        help="Step of constructing the diffusiob matrix")
    parser.add_argument('--num_heads', type=int, default=2,
                        help="Number of multiattention head")
    parser.add_argument('--decay_steps', type=int, default=2000,
                        help="Teacher forcing probability decay ratio")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="Initial learning rate")
    parser.add_argument('--minimum_lr', type=float, default=2e-6,
                        help="Lower bound of learning rate")
    parser.add_argument('--dataset', type=str, default='LA',
                        help="dataset LA for METR_LA; BAY for PEMS_BAY")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of epoches for training")
    parser.add_argument('--max_grad_norm', type=float, default=5.0,
                        help="Maximum gradient norm for update parameters")

    parser.add_argument('--in-feats', type=int, default=2,
                        help="num of node features increased by replication; 2 is # feats of the original dataset")
    parser.add_argument('--replicate-nodes', type=int, default=1,
                        help="Relicate nodes; 1 for original dataset")
    parser.add_argument('--num-layers', type=int, default=2,
                        help="Number of layers of the encoder/decoder")
    parser.add_argument('--rnn', type=str, default='gru',
                        help="rnn model: gru or lstm")
    parser.add_argument('--merge-time-steps', action='store_true',
                        help="enable optimization of merging time steps")
    parser.add_argument('--reuse-msg-passing', action='store_true',
                        help="enable optimization of resusing message passing")
    
    # pytorch dist training
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--workspace", type=str)
    # parser.add_argument("--master_addr", type=str)
    # parser.add_argument("--local_world_size", type=int, default=1)
    
    args = parser.parse_args()
    # Load the datasets
    if args.dataset == 'LA':
        g = METR_LAGraphDataset(args.workspace)
        train_data = METR_LATrainDataset(args.workspace)
        test_data = METR_LATestDataset(args.workspace)
        valid_data = METR_LAValidDataset(args.workspace)
    elif args.dataset == 'BAY':
        g = PEMS_BAYGraphDataset(args.workspace)
        train_data = PEMS_BAYTrainDataset(args.workspace)
        test_data = PEMS_BAYTestDataset(args.workspace)
        valid_data = PEMS_BAYValidDataset(args.workspace)

    if args.replicate_nodes > 1:
        g = dgl.batch([g]*args.replicate_nodes)
    
    spmd_main(args.local_rank, g, train_data, valid_data, test_data, args)
#    import torch.multiprocessing as mp
#    num_gpus = 2
#    procs = []
#    for rank in range(num_gpus):
#        p = mp.Process(target=main, args=(rank, num_gpus, g, train_data, valid_data, test_data, args))
#        p.start()
#        procs.append(p)
#    for p in procs:
#        p.join()    

