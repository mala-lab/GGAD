import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import MLP

import numpy as np


def neighList_to_edgeList(adj):
    edge_list = []
    for i in range(adj.shape[0]):
        for j in torch.argwhere(adj[i, :] > 0):
            edge_list.append([int(i), int(j)])
    return edge_list


def neighList_to_edgeList_train(adj, idx_train):
    edge_list = []
    for i in idx_train:
        for j in torch.argwhere(adj[i, :] > 0):
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
            out = torch.bmm(adj, seq_fts)
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

    def double_recon_loss(self,
                          x,
                          x_,
                          s,
                          s_,
                          weight=0.5,
                          pos_weight_a=0.5,
                          pos_weight_s=0.5,
                          bce_s=False):
        r"""
        Double reconstruction loss function for feature and structure.
        The loss function is defined as :math:`\alpha \symbf{E_a} +
        (1-\alpha) \symbf{E_s}`, where :math:`\alpha` is the weight between
        0 and 1 inclusive, and :math:`\symbf{E_a}` and :math:`\symbf{E_s}`
        are the reconstruction loss for feature and structure, respectively.
        The first dimension is kept for outlier scores of each node.

        For feature reconstruction, we use mean squared error loss:
        :math:`\symbf{E_a} = \|\symbf{X}-\symbf{X}'\odot H\|`,
        where :math:`H=\begin{cases}1 - \eta &
        \text{if }x_{ij}=0\\ \eta & \text{if }x_{ij}>0\end{cases}`, and
        :math:`\eta` is the positive weight for feature.

        For structure reconstruction, we use mean squared error loss by
        default: :math:`\symbf{E_s} = \|\symbf{S}-\symbf{S}'\odot
        \Theta\|`, where :math:`\Theta=\begin{cases}1 -
        \theta & \text{if }s_{ij}=0\\ \theta & \text{if }s_{ij}>0
        \end{cases}`, and :math:`\theta` is the positive weight for
        structure. Alternatively, we can use binary cross entropy loss
        for structure reconstruction: :math:`\symbf{E_s} =
        \text{BCE}(\symbf{S}, \symbf{S}' \odot \Theta)`.

        Parameters
        ----------
        x : torch.Tensor
            Ground truth node feature
        x_ : torch.Tensor
            Reconstructed node feature
        s : torch.Tensor
            Ground truth node structure
        s_ : torch.Tensor
            Reconstructed node structure
        weight : float, optional
            Balancing weight :math:`\alpha` between 0 and 1 inclusive between node feature
            and graph structure. Default: ``0.5``.
        pos_weight_a : float, optional
            Positive weight for feature :math:`\eta`. Default: ``0.5``.
        pos_weight_s : float, optional
            Positive weight for structure :math:`\theta`. Default: ``0.5``.
        bce_s : bool, optional
            Use binary cross entropy for structure reconstruction loss.

        Returns
        -------
        score : torch.tensor
            Outlier scores of shape :math:`N` with gradients.
        """

        assert 0 <= weight <= 1, "weight must be a float between 0 and 1."
        assert 0 <= pos_weight_a <= 1 and 0 <= pos_weight_s <= 1, \
            "positive weight must be a float between 0 and 1."

        # attribute reconstruction loss
        diff_attr = torch.pow(x - x_, 2)

        if pos_weight_a != 0.5:
            diff_attr = torch.where(x > 0,
                                    diff_attr * pos_weight_a,
                                    diff_attr * (1 - pos_weight_a))

        attr_error = torch.sqrt(torch.sum(diff_attr, 1))

        # structure reconstruction loss
        if bce_s:
            diff_stru = F.binary_cross_entropy(s_, s, reduction='none')
        else:
            diff_stru = torch.pow(s - s_, 2)

        if pos_weight_s != 0.5:
            diff_stru = torch.where(s > 0,
                                    diff_stru * pos_weight_s,
                                    diff_stru * (1 - pos_weight_s))

        stru_error = torch.sqrt(torch.sum(diff_stru, 1))

        score = weight * attr_error + (1 - weight) * stru_error

        return score

    @staticmethod
    def loss_func_g(a_):
        loss_g = F.binary_cross_entropy(a_, torch.ones_like(a_))
        return loss_g

    @staticmethod
    def loss_func_ed(a, a_):
        loss_r = F.binary_cross_entropy(a, torch.ones_like(a))
        loss_f = F.binary_cross_entropy(a_, torch.zeros_like(a_))
        return (loss_f + loss_r) / 2

    def model_enc(self, x, noise):
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
        # if torch.cuda.is_available():
        #     x_ = self.generator(noise).cuda()
        # else:
        #     x_ = self.generator(noise)
        x_ = self.generator(noise)
        self.emb = self.discriminator(x)
        z_ = self.discriminator(x_)

        a = torch.sigmoid((self.emb @ self.emb.T))
        a_ = torch.sigmoid((z_ @ z_.T))

        return x_, a, a_

    def forward(self, seq1, adj, idx_train, idx_test, sparse=False):
        seq1 = torch.squeeze(seq1)
        adj = torch.squeeze(adj)
        # if torch.cuda.is_available():
        #     noise = torch.randn(seq1.shape[0], self.noise_dim).cuda()
        # else:
        #     noise = torch.randn(seq1.shape[0], self.noise_dim)
        noise = torch.randn(seq1.shape[0], self.noise_dim)
        # if torch.cuda.is_available():
        #     noise = noise.cuda()

        x_, a, a_ = self.model_enc(seq1, noise)
        # edge_index = np.array(neighList_to_edgeList_train(adj, idx_train))
        edge_index = np.array(neighList_to_edgeList_train(adj, idx_train))
        # loss_g = self.loss_func_g(a_[edge_index[:, 0], edge_index[:, 1]])
        # mar = a[edge_index[:, 0], edge_index[:, 1]]
        loss = self.loss_func_ed(a[edge_index[:, 0], edge_index[:, 1]],
                                 a_[edge_index[:, 0], edge_index[:, 1]].detach())
        # seq1_mean = torch.mean(seq1[idx_train], 0)
        diff_attr = torch.pow(seq1[idx_train, :] - x_[idx_train, :], 2)
        loss_g = torch.mean(torch.sqrt(torch.sum(diff_attr, 1)), 0)
        # loss = 0.5 * loss + loss_g
        self.weight = 1
        score = self.double_recon_loss(x=seq1[idx_test, :],
                                       x_=x_[idx_test, :],
                                       s=a[idx_test, :],
                                       s_=a[idx_test, :],
                                       weight=self.weight,
                                       pos_weight_s=1,
                                       bce_s=True)

        return loss, loss_g, score
