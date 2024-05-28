import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN

def neighList_to_edgeList(adj):
    edge_list = []
    for i in range(adj.shape[0]):
        for j in torch.argwhere(adj[i, :] >0):
            edge_list.append((int(i), int(j)))
    return edge_list


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
        self.read_mode = readout
        self.dense_stru = nn.Linear(n_in, n_h)
        self.gat_layer = GCN(n_h, n_in, num_layers=2)
        self.dense_attr_1 = nn.Linear(n_in, n_h)
        self.dense_attr_2 = nn.Linear(n_h, n_in)
        self.dropout = 0.
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

    def double_recon_loss(self, x,
                          x_,
                          s,
                          s_,
                          weight=1,
                          ):
        # attribute reconstruction loss
        diff_attr = torch.pow(x - x_, 2)

        attr_error = torch.sqrt(torch.sum(diff_attr, 1))

        # diff_stru = torch.pow(s - s_, 2)

        # stru_error = torch.sqrt(torch.sum(diff_stru, 1))

        # score = weight * attr_error + (1 - weight) * stru_error
        score = weight * attr_error

        return score

    def model_enc(self, x, edge_index):
        """
         Forward computation.

         Parameters
         ----------
         x : torch.Tensor
             Input attribute embeddings.
         edge_index : torch.Tensor
             Edge index.
         batch_size : int
             Batch size.

         Returns
         -------
         x_ : torch.Tensor
             Reconstructed attribute embeddings.
         s_ : torch.Tensor
             Reconstructed adjacency matrix.
         """
        h = self.dense_stru(x)
        if self.act is not None:
            h = self.act(h)
        h = F.dropout(h, self.dropout)
        self.emb = self.gat_layer(h, edge_index)

        # s_ = torch.sigmoid(self.emb @ self.emb.T)

        x = self.dense_attr_1(x)
        if self.act is not None:
            x = self.act(x)
        x = F.dropout(x, self.dropout)
        x = self.dense_attr_2(x)
        x_ = F.dropout(x, self.dropout)
        # x_ = self.emb @ x.T
        # return x_, s_
        return x_, x_

    def forward(self, seq1, adj, idx_train, idx_test, sparse=False):
        adj = torch.squeeze(adj)
        seq1 = torch.squeeze(seq1)
        edge_index = neighList_to_edgeList(adj)
        edge_index = torch.tensor(np.array(edge_index)).T.cuda()
        # edge_index = torch.tensor(np.array(edge_index)).T
        x_, s_ = self.model_enc(seq1, edge_index)

        score = self.double_recon_loss(seq1[idx_train, :],
                                     x_[idx_train, :],
                                     adj[idx_train, :],
                                     s_[idx_train, :],
                                     )

        loss = torch.mean(score)
        score_test = self.double_recon_loss(seq1[idx_test, :],
                                     x_[idx_test, :],
                                     adj[idx_test, :],
                                     s_[idx_test, :],
                                     )

        return loss, score_test

