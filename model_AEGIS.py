import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import MLP

import numpy as np

def neighList_to_edgeList(adj):
    edge_list = []
    for i in range(adj.shape[0]):
        for j in torch.argwhere(adj[i, :] >0):
            edge_list.append([int(i), int(j)])
    return edge_list

def neighList_to_edgeList_train(adj, idx_train):
    edge_list = []
    for i in idx_train:
        for j in torch.argwhere(adj[i, :] >0):
            edge_list.append([int(i), int(j)])
    return edge_list

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.mm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq, 1).values


class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values


class WSReadout(nn.Module):
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq, sim)
        out = torch.sum(out, 1)
        return out


class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl):
        scs = []
        # positive
        scs.append(self.f_k(h_pl, c))

        # negative
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1, :], c_mi[:-1, :]), 0)
            scs.append(self.f_k(h_pl, c_mi))

        logits = torch.cat(tuple(scs))

        return logits


class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout):
        super(Model, self).__init__()
        self.noise_dim = 16
        self.hid_dim = 64
        self.read_mode = readout

        self.act = nn.ReLU()
        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

        self.disc = Discriminator(n_h, negsamp_round)

        noise_dim = 16
        hid_dim = 64
        num_layers = 4
        dropout = 0.
        in_dim = n_in
        generator_layers = math.floor(num_layers / 2)
        encoder_layers = math.ceil(num_layers / 2)
        act = torch.nn.functional.relu
        self.gcn_enc1 = GCN(n_in, n_h, activation)
        self.gcn_enc2 = GCN(n_h, n_h, activation)
        self.gcn_dec1 = GCN(n_h, n_h, activation)
        self.gcn_dec2 = GCN(n_h, n_in, activation)
        self.generator = MLP(in_channels=noise_dim,
                             hidden_channels=hid_dim,
                             out_channels=in_dim,
                             num_layers=generator_layers,
                             dropout=dropout,
                             act=act)

        self.discriminator = MLP(in_channels=in_dim,
                                 hidden_channels=hid_dim,
                                 out_channels=hid_dim,
                                 num_layers=encoder_layers,
                                 dropout=dropout,
                                 act=act
                                 )
        self.discriminator2 = MLP(in_channels=n_h,
                                 hidden_channels=hid_dim,
                                 out_channels=1,
                                 num_layers=encoder_layers,
                                 dropout=dropout,
                                 act=torch.sigmoid
                                 )



    def model_enc(self, x, adj,  noise, idx_train):
        """
        Forward computation.

        Parameters
        ----------
        x : torch.Tensor
            Input attribute embeddings.
        noise : torch.Tensor
            Input noise.

        Returns
        -------
        x_ : torch.Tensor
            Reconstructed node features.
        a : torch.Tensor
            Reconstructed adjacency matrix from real samples.
        a_ : torch.Tensor
            Reconstructed adjacency matrix from fake samples.
        """
        x_gen = self.generator(noise.cuda())
        # x_gen = self.generator(noise)
        z_gen = self.gcn_enc1(x_gen, adj)
        z_gen = self.gcn_enc2(z_gen, adj)

        z = self.gcn_enc1(x, adj)
        z = self.gcn_enc2(z, adj)

        z_gen_dec = self.gcn_dec1(z, adj)
        z_gen_dec = self.gcn_dec2(z_gen_dec, adj)

        z_dec = self.gcn_dec1(z, adj)
        z_dec = self.gcn_dec2(z_dec, adj)

        emb_all = torch.cat([z, z_gen], 0)
        label = torch.cat([torch.zeros(len(z)), torch.ones(len(z_gen))])
        logits = self.discriminator2(emb_all)
        logits_gen = self.discriminator2(z_gen)
        logits = torch.sigmoid(logits)
        logits_gen = torch.sigmoid(logits_gen)

        idx_train = idx_train + [len(z)+i for i in range(len(z))]
        loss_dis = F.binary_cross_entropy(logits[idx_train, 0], label[idx_train].cuda())
        # loss_dis = F.binary_cross_entropy(logits[idx_train, 0], label[idx_train])
        loss_g = F.binary_cross_entropy(logits_gen[:, 0], torch.zeros_like(logits_gen[:, 0]))
        return z_dec, loss_dis, loss_g, logits, emb_all


    def forward(self, seq1, adj, idx_train, idx_test, sparse=False):
        seq1 = torch.squeeze(seq1)
        adj = torch.squeeze(adj)
        noise = torch.randn(seq1.shape[0], self.noise_dim)
        z_dec, loss_dis, loss_g, logits, emb_all = self.model_enc(seq1, adj, noise, idx_train)

        diff_attr = torch.pow(seq1[idx_train, :] - z_dec[idx_train, :], 2)
        # diff_attr = torch.pow(seq1[:, :] - z_gen_dec[:, :], 2)
        loss_ae = torch.mean(torch.sqrt(torch.sum(diff_attr, 1)), 0)

        score = logits[idx_test, :]
        return loss_ae, loss_g, loss_ae,  score, emb_all
