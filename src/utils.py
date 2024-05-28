import pickle
import random
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
import copy as cp
import torch
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score, \
    confusion_matrix

from collections import defaultdict
import dgl


def load_data(data, prefix='data/'):
    """
	Load graph, feature, and label given dataset name
	:returns: home and single-relation graphs, feature, label
	"""

    if data == 'dgraphfin':
        f = np.load('../data/dgraphfin.npz')
        labels = torch.tensor(f['y']).float()
        labels = (labels == 1).int()
        labels = np.array(labels)
        feat_data = np.array(torch.tensor(f['x']).float())
        with open(prefix + 'dgraphfin_adj_list', 'rb') as file:
            homo = pickle.load(file)
        file.close()

    return homo, feat_data, labels

    # return [homo, relation1, relation2, relation3], feat_data, labels


def trans_sparse(adj):
    row = adj._indices()[0]
    col = adj._indices()[1]
    data = adj._values()
    shape = adj.size()
    adj = sp.csr_matrix((data, (row, col)), shape=shape)
    return adj


def sparse_to_adjlist(sp_matrix, filename):
    """
	Transfer sparse matrix to adjacenc-y list
	:param sp_matrix: the sparse matrix
	:param filename: the filename of adjlist
	"""
    # add self loop
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    # create adj_list
    adj_lists = defaultdict(set)
    edges = homo_adj.nonzero()
    for index, node in enumerate(edges[0]):
        adj_lists[node].add(edges[1][index])
        adj_lists[edges[1][index]].add(node)
    with open(filename, 'wb') as file:
        pickle.dump(adj_lists, file)
    file.close()


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize(mx):
    """
		Row-normalize sparse matrix
		Code from https://github.com/williamleif/graphsage-simple/
	"""
    rowsum = np.array(mx.sum(1)) + 0.01
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


# def normalize(features):
#     """Row-normalize feature matrix and convert to tuple representation"""
#     rowsum = np.array(features.sum(1))
#     r_inv = np.power(rowsum, -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0.
#     r_mat_inv = sp.diags(r_inv)
#     features = r_mat_inv.dot(features)
#     return features

def sparse_to_adjlist(sp_matrix, filename):
    """
	Transfer sparse matrix to adjacency list
	:param sp_matrix: the sparse matrix
	:param filename: the filename of adjlist
	"""
    # add self loop
    # homo_adj = sp_matrix + torch.eye(sp_matrix.shape[0])
    # create adj_list
    adj_lists = defaultdict(set)
    edges = sp_matrix.nonzero()
    for index, node in enumerate(edges[0]):
        adj_lists[node].add(edges[1][index])
        adj_lists[edges[1][index]].add(node)
    with open(filename, 'wb') as file:
        pickle.dump(adj_lists, file)
    file.close()


def pos_neg_split(nodes, labels):
    """
	Find positive and negative nodes given a list of nodes and their labels
	:param nodes: a list of nodes
	:param labels: a list of node labels
	:returns: the spited positive and negative nodes
	"""
    pos_nodes = []
    neg_nodes = cp.deepcopy(nodes)
    aux_nodes = cp.deepcopy(nodes)
    for idx, label in enumerate(labels):
        if label == 1:
            pos_nodes.append(aux_nodes[idx])
            neg_nodes.remove(aux_nodes[idx])

    return pos_nodes, neg_nodes


def pick_step(idx_train, y_train, adj_list, size):
    degree_train = [len(adj_list[node]) for node in idx_train]
    lf_train = (y_train.sum() - len(y_train)) * y_train + len(y_train)
    smp_prob = np.array(degree_train) / lf_train
    return random.choices(idx_train, weights=smp_prob, k=size)


def test_recon(test_cases, labels, model, batch_size, test_attr, thres=0.5):
    """
	Test the performance of GraphSAGE
	:param test_cases: a list of testing node
	:param labels: a list of testing node labels
	:param model: the GNN model
	:param batch_size: number nodes in a batch
	"""

    test_batch_num = int(len(test_cases) / batch_size) + 1
    gnn_pred_list = []
    gnn_prob_list = []
    for iteration in range(test_batch_num):
        i_start = iteration * batch_size
        i_end = min((iteration + 1) * batch_size, len(test_cases))
        batch_nodes = test_cases[i_start:i_end]
        batch_label = labels[i_start:i_end]
        emb = model.to_prob(batch_nodes, None)

        diff_attribute = torch.pow(emb.cuda() - test_attr[batch_nodes], 2)
        gnn_prob = torch.sqrt(torch.sum(diff_attribute, 1))

        gnn_prob_arr = gnn_prob.data.cpu().numpy().squeeze()
        # gnn_pred = prob2pred(gnn_prob_arr, thres)

        # gnn_pred_list.extend(gnn_pred.tolist())
        gnn_prob_list.extend(gnn_prob_arr.tolist())

    auc_gnn = roc_auc_score(labels, np.array(gnn_prob_list))
    AP = average_precision_score(labels, np.array(gnn_prob_list), average='macro', pos_label=1, sample_weight=None)

    print('Testing AUC', auc_gnn)
    print('Testing AP:', AP)


def test_aegis(test_cases, labels, model, batch_size, thres=0.5):
    """
	Test the performance of GraphSAGE
	:param test_cases: a list of testing node
	:param labels: a list of testing node labels
	:param model: the GNN model
	:param batch_size: number nodes in a batch
	"""

    test_batch_num = int(len(test_cases) / batch_size) + 1
    gnn_pred_list = []
    gnn_prob_list = []
    for iteration in range(test_batch_num):
        i_start = iteration * batch_size
        i_end = min((iteration + 1) * batch_size, len(test_cases))
        batch_nodes = test_cases[i_start:i_end]
        batch_label = labels[i_start:i_end]
        gnn_prob = model.to_prob(batch_nodes)

        gnn_prob_arr = gnn_prob.data.cpu().numpy().squeeze()
        gnn_pred = prob2pred(gnn_prob_arr, thres)

        gnn_pred_list.extend(gnn_pred.tolist())
        gnn_prob_list.extend(gnn_prob_arr.tolist())

    auc_gnn = roc_auc_score(labels, np.array(gnn_prob_list))
    AP = average_precision_score(labels, np.array(gnn_prob_list), average='macro', pos_label=1, sample_weight=None)

    print('Testing AP:', AP)
    print('Testing AUC', auc_gnn)


def test_sage(test_cases, labels, model, batch_size, thres=0.5):
    """
	Test the performance of GraphSAGE
	:param test_cases: a list of testing node
	:param labels: a list of testing node labels
	:param model: the GNN model
	:param batch_size: number nodes in a batch
	"""

    test_batch_num = int(len(test_cases) / batch_size) + 1
    gnn_pred_list = []
    gnn_prob_list = []
    for iteration in range(test_batch_num):
        i_start = iteration * batch_size
        i_end = min((iteration + 1) * batch_size, len(test_cases))
        batch_nodes = test_cases[i_start:i_end]
        batch_label = labels[i_start:i_end]
        gnn_prob = model.to_prob(batch_nodes, None)

        gnn_prob_arr = gnn_prob.data.cpu().numpy().squeeze()
        gnn_pred = prob2pred(gnn_prob_arr, thres)

        gnn_pred_list.extend(gnn_pred.tolist())
        gnn_prob_list.extend(gnn_prob_arr.tolist())

    auc_gnn = roc_auc_score(labels, np.array(gnn_prob_list))
    AP = average_precision_score(labels, np.array(gnn_prob_list), average='macro', pos_label=1, sample_weight=None)

    f1_binary_1_gnn = f1_score(labels, np.array(gnn_pred_list), pos_label=1, average='binary')
    f1_binary_0_gnn = f1_score(labels, np.array(gnn_pred_list), pos_label=0, average='binary')
    f1_micro_gnn = f1_score(labels, np.array(gnn_pred_list), average='micro')
    f1_macro_gnn = f1_score(labels, np.array(gnn_pred_list), average='macro')
    conf_gnn = confusion_matrix(labels, np.array(gnn_pred_list))
    tn, fp, fn, tp = conf_gnn.ravel()
    gmean_gnn = conf_gmean(conf_gnn)

    print(f"   GNN F1-binary-1: {f1_binary_1_gnn:.4f}\tF1-binary-0: {f1_binary_0_gnn:.4f}" +
          f"\tF1-macro: {f1_macro_gnn:.4f}\tG-Mean: {gmean_gnn:.4f}\tAUC: {auc_gnn:.4f}")
    print('Testing AP:', AP)
    print(f"   GNN TP: {tp}\tTN: {tn}\tFN: {fn}\tFP: {fp}")
    return f1_macro_gnn, f1_binary_1_gnn, f1_binary_0_gnn, auc_gnn, gmean_gnn


def prob2pred(y_prob, thres=0.5):
    """
	Convert probability to predicted results according to given threshold
	:param y_prob: numpy array of probability in [0, 1]
	:param thres: binary classification threshold, default 0.5
	:returns: the predicted result with the same shape as y_prob
	"""
    y_pred = np.zeros_like(y_prob, dtype=np.int32)
    y_pred[y_prob >= thres] = 1
    y_pred[y_prob < thres] = 0
    return y_pred


def test_ggad(test_cases, labels, model, batch_size, thres=0.5):
    """
	Test the performance of PC-GNN and its variants
	:param test_cases: a list of testing node
	:param labels: a list of testing node labels
	:param model: the GNN model
	:param batch_size: number nodes in a batch
	:returns: the AUC and Recall of GNN and Simi modules
	"""

    test_batch_num = int(len(test_cases) / batch_size) + 1
    f1_gnn = 0.0
    acc_gnn = 0.0
    recall_gnn = 0.0
    f1_label1 = 0.0
    acc_label1 = 0.00
    recall_label1 = 0.0
    gnn_pred_list = []
    gnn_prob_list = []
    label_list1 = []

    for iteration in range(test_batch_num):
        i_start = iteration * batch_size
        i_end = min((iteration + 1) * batch_size, len(test_cases))
        batch_nodes = test_cases[i_start:i_end]
        batch_label = labels[i_start:i_end]
        label_prob1, affinitiy = model.to_prob(batch_nodes, batch_label, train_flag=False)

        gnn_prob_arr = label_prob1.data.cpu().numpy()[:, 1]
        gnn_pred = prob2pred(gnn_prob_arr, thres)

        f1_label1 += f1_score(batch_label, label_prob1.data.cpu().numpy().argmax(axis=1), average="macro")
        acc_label1 += accuracy_score(batch_label, label_prob1.data.cpu().numpy().argmax(axis=1))
        recall_label1 += recall_score(batch_label, label_prob1.data.cpu().numpy().argmax(axis=1), average="macro")

        gnn_pred_list.extend(gnn_pred.tolist())
        gnn_prob_list.extend(gnn_prob_arr.tolist())
        label_list1.extend(label_prob1.data.cpu().numpy()[:, 1].tolist())

    auc_gnn = roc_auc_score(labels, np.array(gnn_prob_list))
    ap_gnn = average_precision_score(labels, np.array(gnn_prob_list))
    auc_label1 = roc_auc_score(labels, np.array(label_list1))
    ap_label1 = average_precision_score(labels, np.array(label_list1))

    f1_binary_1_gnn = f1_score(labels, np.array(gnn_pred_list), pos_label=1, average='binary')
    f1_binary_0_gnn = f1_score(labels, np.array(gnn_pred_list), pos_label=0, average='binary')
    f1_micro_gnn = f1_score(labels, np.array(gnn_pred_list), average='micro')
    f1_macro_gnn = f1_score(labels, np.array(gnn_pred_list), average='macro')
    conf_gnn = confusion_matrix(labels, np.array(gnn_pred_list))
    tn, fp, fn, tp = conf_gnn.ravel()
    gmean_gnn = conf_gmean(conf_gnn)

    print(f"   GNN F1-binary-1: {f1_binary_1_gnn:.4f}\tF1-binary-0: {f1_binary_0_gnn:.4f}" +
          f"\tF1-macro: {f1_macro_gnn:.4f}\tG-Mean: {gmean_gnn:.4f}\tAUC: {auc_gnn:.4f}")
    print(f"   GNN TP: {tp}\tTN: {tn}\tFN: {fn}\tFP: {fp}")
    print(f"Label1 F1: {f1_label1 / test_batch_num:.4f}\tAccuracy: {acc_label1 / test_batch_num:.4f}" +
          f"\tRecall: {recall_label1 / test_batch_num:.4f}\tAUC: {auc_label1:.4f}\tAP: {ap_label1:.4f}")

    return f1_macro_gnn, f1_binary_1_gnn, f1_binary_0_gnn, auc_gnn, gmean_gnn


def conf_gmean(conf):
    tn, fp, fn, tp = conf.ravel()
    return (tp * tn / ((tp + fn) * (tn + fp))) ** 0.5
