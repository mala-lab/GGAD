import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score, \
    confusion_matrix

"""
	GraphSAGE implementations
	Paper: Inductive Representation Learning on Large Graphs
	Source: https://github.com/williamleif/graphsage-simple/
"""


class GraphSage(nn.Module):
    """
	Vanilla GraphSAGE Model
	Code partially from https://github.com/williamleif/graphsage-simple/
	"""

    def __init__(self, num_classes, enc):
        super(GraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def to_prob(self, nodes):
        pos_scores = torch.sigmoid(self.forward(nodes))
        return pos_scores

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())


class MeanAggregator(nn.Module):
    """
	Aggregates a node's embeddings using mean of neighbors' embeddings
	"""

    def __init__(self, features, cuda=False, gcn=False):
        """
		Initializes the aggregator for a specific graph.

		features -- function mapping LongTensor of node ids to FloatTensor of feature values.
		cuda -- whether to use GPU
		gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
		"""

        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn

    def forward(self, nodes, to_neighs, num_sample=10):
        """
		nodes --- list of nodes in a batch
		to_neighs --- list of sets, each set is the set of neighbors for node in batch
		num_sample --- number of neighbors to sample. No sampling if None.
		"""
        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh,
                                        num_sample,
                                        )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh.union(set([int(nodes[i])])) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        to_feats = mask.mm(embed_matrix)
        return to_feats


class Encoder(nn.Module):
    """
	Vanilla GraphSAGE Encoder Module
	Encodes a node's using 'convolutional' GraphSage approach
	"""

    def __init__(self, features, feature_dim,
                 embed_dim, adj_lists, aggregator,
                 num_sample=10,
                 base_model=None, gcn=False, cuda=False,
                 feature_transform=False):
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        self.weight = nn.Parameter(
            torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        """
		Generates embeddings for a batch of nodes.

		nodes     -- list of nodes
		"""
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes],
                                              self.num_sample)

        if isinstance(nodes, list):
            index = torch.LongTensor(nodes)
        else:
            index = nodes

        if not self.gcn:
            if self.cuda:
                self_feats = self.features(index).cuda()
            else:
                self_feats = self.features(index)
            combined = torch.cat((self_feats, neigh_feats), dim=1)
        else:
            combined = neigh_feats
        combined = F.relu(self.weight.mm(combined.t()))
        return combined


class GCN(nn.Module):
    """
	Vanilla GCN Model
	Code partially from https://github.com/williamleif/graphsage-simple/
	"""

    def __init__(self, num_classes, enc):
        super(GCN, self).__init__()
        self.enc = enc
        # self.xent = nn.CrossEntropyLoss()
        self.xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([1]))
        self.weight = nn.Parameter(torch.FloatTensor(1, enc.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes, label, train_flag):
        embeds, to_feats_neigh, anomaly_feat, anomaly_feat_new = self.enc(nodes, label, train_flag)
        # print(True in torch.isnan(embeds))
        scores = self.weight.mm(embeds)
        # scores = torch.sigmoid(scores)
        return scores.t(), to_feats_neigh, embeds, anomaly_feat, anomaly_feat_new

    def to_prob(self, nodes, label):
        pos_scores, affinity, embeds, anomaly_feat, anomaly_feat_new = self.forward(nodes, label, train_flag=False)
        pos_scores = torch.sigmoid(pos_scores)
        return pos_scores

    def to_prob_reconstruction(self, nodes, label):
        pos_scores, affinity, embeds, anomaly_feat, anomaly_feat_new = self.forward(nodes, label, train_flag=False)
        return pos_scores

    def to_prob_one_class_svm(self, nodes, label):
        pos_scores, affinity, embeds = self.forward(nodes, label, train_flag=False)
        return pos_scores


    def recon2(self, anomaly_feat, anomaly_feat_new):
        std = 0.01
        mean = 0.02
        # noise = torch.randn(anomaly_feat.size()) * std + mean
        # anomaly_feat = anomaly_feat + noise.cuda()
        diff_attribute = torch.pow(anomaly_feat - anomaly_feat_new, 2)
        loss_rec = torch.mean(torch.sqrt(torch.sum(diff_attribute, 0)))
        return loss_rec

    def normalize(self, emb):
        emb_inf = torch.norm(emb, dim=-1, keepdim=True)
        emb_inf = torch.pow(emb_inf, -1)
        emb_inf[torch.isinf(emb_inf)] = 0.
        emb_norm = emb * emb_inf
        return emb_norm

    # Add loss function on affinity
    def affinity(self, combined_all, labels, to_feats_neigh):
        # Caculate node affinity after normalization
        # combined_norm = self.normalize(combined_all)
        # to_feats_neigh_norm = self.normalize(to_feats_neigh)
        # if True in torch.isnan(combined_norm):
        #     print('error')
        # print(True in torch.isnan(combined_norm))
        # print(True in torch.isnan(to_feats_neigh_norm))
        # combined_norm[torch.isnan(combined_norm)] = 0
        # to_feats_neigh_norm[torch.isnan(to_feats_neigh_norm)] = 0
        # affinity = torch.diag(torch.mm(to_feats_neigh, combined_all))
        # affinity = torch.sum(torch.mul(to_feats_neigh_norm, combined_norm.t()), 1)
        # combined_all[torch.isnan(combined_all)] = 0
        # to_feats_neigh[torch.isnan(to_feats_neigh)] = 0

        # emb_combined_all = torch.norm(combined_all, dim=-1, keepdim=True)
        # emb_combined_all = torch.pow(emb_combined_all, -1)
        # emb_combined_all[torch.isinf(emb_combined_all)] = 0.
        # combined_all = combined_all * emb_combined_all
        #
        # emb_to_feats_neigh = torch.norm(to_feats_neigh, dim=-1, keepdim=True)
        # emb_to_feats_neigh = torch.pow(emb_to_feats_neigh, -1)
        # emb_to_feats_neigh[torch.isinf(emb_to_feats_neigh)] = 0.
        # to_feats_neigh = to_feats_neigh * emb_to_feats_neigh

        affinity = torch.cosine_similarity(combined_all, to_feats_neigh.t(), dim=0)
        # affinity = torch.diag(torch.mm(to_feats_neigh, combined_all))
        confidence_margin = 1
        affinity_normal_mean = torch.mean(affinity[torch.argwhere(labels == 0)], 0)
        affinity_abnormal_mean = torch.mean(affinity[torch.argwhere(labels == 1)], 0)
        # marigin loss function
        loss_margin = (confidence_margin - (affinity_normal_mean - affinity_abnormal_mean)).clamp_min(min=0)

        return loss_margin

    def loss(self, nodes, labels):
        scores, to_feats_neigh, embeds, anomaly_feat, anomaly_feat_new = self.forward(nodes, labels, train_flag=True)
        loss_cls = torch.mean(self.xent(scores.squeeze(), torch.tensor(labels, dtype=torch.float32)))

        loss_constraint = self.affinity(embeds, labels, to_feats_neigh)

        # print("loss_constraint", loss_constraint)
        loss_rec = self.recon2(anomaly_feat, anomaly_feat_new)
        # print('loss_cls', loss_cls)
        #
        # print('loss_constraint', loss_constraint)
        # print('loss_rec', loss_rec)
        # return loss_cls, loss_cls, loss_cls, loss_cls
        #
        return 1 * loss_cls + 1 * loss_constraint + 0.1 * loss_rec, loss_cls, loss_constraint, loss_rec

    def reconstruction_loss(self, nodes, labels):
        scores, to_feats_neigh, embeds = self.forward(nodes, labels, train_flag=True)

        loss_rec = self.recon(embeds, labels)

        return 1 * loss_rec

    def one_class_svm(self, nodes, labels):
        scores, to_feats_neigh, embeds = self.forward(nodes, labels, train_flag=True)

        loss_rec = self.recon(embeds, labels)

        return 1 * loss_rec


class GCNAggregator(nn.Module):
    """ where is insert
	Aggregates a node's embeddings using normalized mean of neighbors' embeddings
	"""

    def __init__(self, features, cuda=False, gcn=False):
        """
		Initializes the aggregator for a specific graph.

		features -- function mapping LongTensor of node ids to FloatTensor of feature values.
		cuda -- whether to use GPU
		gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
		"""

        super(GCNAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn

    def forward(self, nodes, to_neighs, adj_list, train_flag):
        """
		nodes --- list of nodes in a batch
		to_neighs --- list of sets, each set is the set of neighbors for node in batch
		"""
        # Local pointers to functions (speed hack)

        samp_neighs = to_neighs

        #  Add self to neighs
        samp_neighs = [samp_neigh.union(set([int(nodes[i])])) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1.0  # Adjacency matrix for the sub-graph
        if self.cuda:
            mask = mask.cuda()
        row_normalized = mask.sum(1, keepdim=True).sqrt()
        col_normalized = mask.sum(0, keepdim=True).sqrt()
        row_normalized_avg = mask.sum(1, keepdim=True)
        mask_row = mask.div(row_normalized_avg)
        mask = mask.div(row_normalized).div(col_normalized)
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
            to_feats = mask.mm(embed_matrix)
            # to_feats = to_feats + self.features(torch.LongTensor(nodes).cuda())
            to_feats = to_feats + self.features(torch.LongTensor(nodes).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
            to_feats = mask.mm(embed_matrix)
            # to_feats = to_feats + self.features(torch.LongTensor(nodes))


        to_feats_neigh = None
        if train_flag == True:
            # Embedding of Neighbors for Affinity
            # samp_neighs = [adj_list[int(node)] for node in unique_nodes_list]
            ## adj_list get neighor from key
            samp_neighs = [adj_list.get(node) for node in unique_nodes_list]
            unique_nodes_list_expand = list(set.union(*samp_neighs))
            unique_nodes_expand = {n: i for i, n in enumerate(unique_nodes_list_expand)}
            mask_neigh = Variable(torch.zeros(len(samp_neighs), len(unique_nodes_expand)))
            column_indices = [unique_nodes_expand[n] for samp_neigh in samp_neighs for n in samp_neigh]
            row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]

            mask_neigh[row_indices, column_indices] = 1.0  # Adjacency matrix for the sub-graph

            if self.cuda:
                mask_neigh = mask_neigh.cuda()
            row_normalized = mask_neigh.sum(1, keepdim=True).sqrt()
            col_normalized = mask_neigh.sum(0, keepdim=True).sqrt()
            mask_neigh = mask_neigh.div(row_normalized).div(col_normalized)
            if self.cuda:
                embed_matrix_expand = self.features(torch.LongTensor(unique_nodes_list_expand).cuda())
                to_feats_neigh = mask_neigh.mm(embed_matrix_expand)
                to_feats_neigh = to_feats_neigh + self.features(torch.LongTensor(unique_nodes_list).cuda())
            else:
                embed_matrix_expand = self.features(torch.LongTensor(unique_nodes_list_expand))
                to_feats_neigh = mask_neigh.mm(embed_matrix_expand)
                # to_feats_neigh = to_feats_neigh + self.features(torch.LongTensor(unique_nodes_list))

            # to_feats_neigh = self.features(torch.LongTensor(unique_nodes_list).cuda())

        return to_feats, to_feats_neigh, mask_row


class GCNEncoder(nn.Module):
    """
	GCN Encoder Module
	"""

    def __init__(self, features, feature_dim,
                 embed_dim, adj_lists, aggregator,
                 num_sample=10,
                 base_model=None, gcn=False, cuda=False,
                 feature_transform=False):
        super(GCNEncoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        self.weight = nn.Parameter(
            torch.FloatTensor(embed_dim, self.feat_dim))
        init.xavier_uniform_(self.weight)
        self.fc = nn.Linear(embed_dim, embed_dim, bias=False)
        # self.weight_generate = nn.Parameter(
        #     torch.FloatTensor(embed_dim, embed_dim))
        # init.xavier_uniform_(self.weight_generate)

    def forward(self, nodes, label, train_flag):
        """
		Generates embeddings for a batch of nodes.
		Input:
			nodes -- list of nodes
		Output:
		    embed_dim*len(nodes)
		"""
        neigh_feats, neigh_feats_expand, mask = self.aggregator.forward(nodes,
                                                                        [self.adj_lists[int(node)] for node in nodes],
                                                                        self.adj_lists, train_flag)

        if isinstance(nodes, list):
            index = torch.LongTensor(nodes)
        else:
            index = nodes

        combined = F.relu(self.weight.mm(neigh_feats.t()))

        to_feats_neigh = None
        anomaly_feat = None
        anomaly_feat_new = None
        combined_all = combined
        if train_flag == True:
            combined_expand = F.relu(self.weight.mm(neigh_feats_expand.t()))

            to_feats_neigh = mask.mm(combined_expand.t())
            # generate anomaly samples from neighs
            # batch_anomaly_id = [i for i in self.smaple_anomaly_id if i in nodes]
            # arg_batch_ano_id = [torch.argwhere(nodes == i) for i in batch_anomaly_id]


            anomaly_feat = combined[:, label == 1]
            anomaly_feat2 = to_feats_neigh.T[:, label == 1]

            anomaly_feat_new = F.relu(self.fc(anomaly_feat2.t()))

            # TODO ablation study add noise on the selected nodes

            # std = 0.01
            # mean = 0.02
            # noise = torch.randn(anomaly_feat2.size()) * std + mean
            # combined_all = torch.cat((combined[:, label == 0], anomaly_feat2 + noise.cuda()), 1)

            # TODO ablation study generate outlier from random noise
            # std = 0.01
            # mean = 0.02
            # noise = torch.randn(anomaly_feat_new.size()) * std + mean
            # emb_con = F.relu(self.fc(noise.cuda()))
            # combined_all = torch.cat((combined[:, label == 0], emb_con.t()), 1)


            # anomaly_feat_new = anomaly_feat2.t()
            # anomaly_feat_new = F.relu(self.fc(anomaly_feat.t()))

            combined_all = torch.cat((combined[:, label == 0], anomaly_feat_new.t()), 1)

            # combined_all = torch.cat((combined[:, label == 0], anomaly_feat), 1)
            anomaly_feat_new = anomaly_feat_new.t()
        return combined_all, to_feats_neigh, anomaly_feat, anomaly_feat_new
