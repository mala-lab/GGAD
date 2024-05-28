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
from torch_geometric.nn import MLP


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

    def forward(self, nodes):
        logits, logits_gen, label = self.enc(nodes)
        return logits, logits_gen, label

    def to_prob(self, nodes):
        logits, logits_gen, label = self.forward(nodes)
        pos_scores = logits[:int(len(logits)/2)]
        return pos_scores

    def to_prob_reconstruction(self, nodes, label):
        pos_scores, affinity, embeds, anomaly_feat, anomaly_feat_new = self.forward(nodes, label, train_flag=False)
        return pos_scores

    def to_prob_one_class_svm(self, nodes, label):
        pos_scores, affinity, embeds = self.forward(nodes, label, train_flag=False)
        return pos_scores

    def reconstruction(self, anomaly_feat, anomaly_feat_new):
        diff_attribute = torch.pow(anomaly_feat - anomaly_feat_new, 2)
        loss_rec = torch.mean(torch.sqrt(torch.sum(diff_attribute, 0)))
        return loss_rec

    def normalize(self, emb):
        emb_inf = torch.norm(emb, dim=-1, keepdim=True)
        emb_inf = torch.pow(emb_inf, -1)
        emb_inf[torch.isinf(emb_inf)] = 0.
        emb_norm = emb * emb_inf
        return emb_norm

    def loss(self, nodes):
        logits, logits_noise, label = self.forward(nodes)
        loss_dis = F.binary_cross_entropy(logits[:, 0].cuda(), label)
        loss_g = F.binary_cross_entropy(logits_noise[:, 0], torch.zeros_like(logits_noise[:, 0]))
        return loss_dis, loss_g


class GCNAggregator(nn.Module):
    """ where is insert
	Aggregates a node's embeddings using normalized mean of neighbors' embeddings
	"""

    def __init__(self, features, features_data,  cuda=False, gcn=False):
        """
		Initializes the aggregator for a specific graph.

		features -- function mapping LongTensor of node ids to FloatTensor of feature values.
		cuda -- whether to use GPU
		gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
		"""

        super(GCNAggregator, self).__init__()

        self.features = features
        self.noise_dim = features_data.shape[1]
        self.noise = torch.randn(features_data.shape[0], self.noise_dim)
        self.cuda = cuda
        self.gcn = gcn

    def forward(self, nodes, to_neighs):
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
            embed_noise_matrix = self.noise[unique_nodes_list, :].cuda()
            to_noise_feats = mask.mm(embed_noise_matrix)
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
            to_feats = mask.mm(embed_matrix)
            embed_noise_matrix = self.noise[unique_nodes_list, :]
            to_noise_feats = mask.mm(embed_noise_matrix)

        return to_feats, to_noise_feats


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
        # self.weight_generate = nn.Parameter(
        #     torch.FloatTensor(embed_dim, embed_dim))
        # init.xavier_uniform_(self.weight_generate)
        self.fc = nn.Linear(embed_dim, feature_dim, bias=False)
        noise_dim = 16
        hid_dim = 64
        num_layers = 4
        dropout = 0.
        act = torch.nn.functional.relu
        in_dim = 64
        import math
        generator_layers = math.floor(num_layers / 2)
        encoder_layers = math.ceil(num_layers / 2)

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
        self.discriminator2 = MLP(in_channels=in_dim,
                                  hidden_channels=hid_dim,
                                  out_channels=1,
                                  num_layers=encoder_layers,
                                  dropout=dropout,
                                  act=torch.sigmoid
                                  )

    def forward(self, nodes):
        """
		Generates embeddings for a batch of nodes.
		Input:
			nodes -- list of nodes
		Output:
		    embed_dim*len(nodes)
		"""
        neigh_feats, to_noise_feats= self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes])

        if isinstance(nodes, list):
            index = torch.LongTensor(nodes)
        else:
            index = nodes
        combined = F.relu(self.weight.mm(neigh_feats.t()))
        combined_noise = F.relu(self.weight.mm(to_noise_feats.t()))

        emb_all = torch.cat([combined, combined_noise], 1)
        label = torch.cat([torch.zeros(len(combined.t())), torch.ones(len(combined_noise.t()))]).cuda()

        logits_all = self.discriminator2(emb_all.t())
        logits_gen = self.discriminator2(combined_noise.t())
        logits_all = torch.sigmoid(logits_all)
        logits_gen = torch.sigmoid(logits_gen)

        return logits_all, logits_gen, label
