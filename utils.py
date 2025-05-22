import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
import dgl
from collections import Counter


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def load_mat(dataset, train_rate=0.3, val_rate=0.1):

    """Load .mat dataset."""
    data = sio.loadmat("./dataset/{}.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']

    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)

    # labels = np.squeeze(np.array(data['Class'], dtype=np.int64) - 1)
    # num_classes = np.max(labels) + 1
    # labels = dense_to_one_hot(labels, num_classes)

    ano_labels = np.squeeze(np.array(label))
    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[: num_train]
    idx_val = all_idx[num_train: num_train + num_val]
    idx_test = all_idx[num_train + num_val:]
    # idx_test = all_idx[num_train:]
    print('Training', Counter(np.squeeze(ano_labels[idx_train])))
    print('Test', Counter(np.squeeze(ano_labels[idx_test])))
    # Sample some labeled normal nodes
    all_normal_label_idx = [i for i in idx_train if ano_labels[i] == 0]
    rate = 0.5  #  change train_rate to 0.3 0.5 0.6  0.8
    normal_label_idx = all_normal_label_idx[: int(len(all_normal_label_idx) * rate)]
    print('Training rate', rate)

    # normal_label_idx = all_normal_label_idx[: int(len(all_normal_label_idx) * 0.2)]
    # normal_label_idx = all_normal_label_idx[: int(len(all_normal_label_idx) * 0.25)]
    # normal_label_idx = all_normal_label_idx[: int(len(all_normal_label_idx) * 0.15)]
    # normal_label_idx = all_normal_label_idx[: int(len(all_normal_label_idx) * 0.10)]

    # contamination
    # real_abnormal_id = np.array(all_idx)[np.argwhere(ano_labels == 1).squeeze()].tolist()
    # add_rate = 0.1 * len(real_abnormal_id)
    # random.shuffle(real_abnormal_id)
    # add_abnormal_id = real_abnormal_id[:int(add_rate)]
    # normal_label_idx = normal_label_idx + add_abnormal_id
    # idx_test = np.setdiff1d(idx_test, add_abnormal_id, False)

    # contamination 
    # real_abnormal_id = np.array(all_idx)[np.argwhere(ano_labels == 1).squeeze()].tolist()
    # add_rate = 0.05 * len(real_abnormal_id)  #0.05 0.1  0.15
    # remove_rate = 0.15 * len(real_abnormal_id)
    # random.shuffle(real_abnormal_id)
    # add_abnormal_id = real_abnormal_id[:int(add_rate)]
    # remove_abnormal_id = real_abnormal_id[:int(remove_rate)]
    # normal_label_idx = normal_label_idx + add_abnormal_id
    # idx_test = np.setdiff1d(idx_test, remove_abnormal_id, False)

    # camouflage
    # real_abnormal_id = np.array(all_idx)[np.argwhere(ano_labels == 1).squeeze()].tolist()
    # normal_feat = np.mean(feat[normal_label_idx], 0)
    # replace_rate = 0.05 * normal_feat.shape[1]
    # feat[real_abnormal_id, :int(replace_rate)] = normal_feat[:, :int(replace_rate)]

    random.shuffle(normal_label_idx)
    # 0.05 for Amazon and 0.15 for other datasets
    if dataset in ['Amazon']:
        abnormal_label_idx = normal_label_idx[: int(len(normal_label_idx) * 0.05)]  
    else:
        abnormal_label_idx = normal_label_idx[: int(len(normal_label_idx) * 0.15)]  
    return adj, feat, ano_labels, all_idx, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels, normal_label_idx, abnormal_label_idx


def adj_to_dgl_graph(adj):
    """Convert adjacency matrix to dgl format."""
    nx_graph = nx.from_scipy_sparse_matrix(adj)
    dgl_graph = dgl.DGLGraph(nx_graph)
    return dgl_graph


def generate_rwr_subgraph(dgl_graph, subgraph_size):
    """Generate subgraph with RWR algorithm."""
    all_idx = list(range(dgl_graph.number_of_nodes()))
    reduced_size = subgraph_size - 1
    traces = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, all_idx, restart_prob=1,
                                                           max_nodes_per_seed=subgraph_size * 3)
    subv = []

    for i, trace in enumerate(traces):
        subv.append(torch.unique(torch.cat(trace), sorted=False).tolist())
        retry_time = 0
        while len(subv[i]) < reduced_size:
            cur_trace = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, [i], restart_prob=0.9,
                                                                      max_nodes_per_seed=subgraph_size * 5)
            subv[i] = torch.unique(torch.cat(cur_trace[0]), sorted=False).tolist()
            retry_time += 1
            if (len(subv[i]) <= 2) and (retry_time > 10):
                subv[i] = (subv[i] * reduced_size)
        subv[i] = subv[i][:reduced_size * 3]
        subv[i].append(i)

    return subv


import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib

matplotlib.use('Agg')
plt.rcParams['figure.dpi'] = 300  # 图片像素
plt.rcParams['figure.figsize'] = (8.5, 7.5)
# plt.rcParams['figure.figsize'] = (10.5, 9.5)
from matplotlib.backends.backend_pdf import PdfPages


def draw_pdf(message_normal, message_abnormal, message_real_abnormal, dataset, epoch):
    message_all = [np.squeeze(message_normal), np.squeeze(message_abnormal), np.squeeze(message_real_abnormal)]
    mu_0 = np.mean(message_all[0])  # 计算均值
    sigma_0 = np.std(message_all[0])
    # print('The mean of normal {}'.format(mu_0))
    # print('The std of normal {}'.format(sigma_0))
    mu_1 = np.mean(message_all[1])  # 计算均值
    sigma_1 = np.std(message_all[1])
    # print('The mean of abnormal {}'.format(mu_1))
    # print('The std of abnormal {}'.format(sigma_1))
    mu_2 = np.mean(message_all[2])  # 计算均值
    sigma_2 = np.std(message_all[2])
    # print('The mean of abnormal {}'.format(mu_2))
    # print('The std of abnormal {}'.format(sigma_2))
    n, bins, patches = plt.hist(message_all, bins=30, normed=1, label=['Normal', 'Outlier', 'Abnormal'])
    y_0 = mlab.normpdf(bins, mu_0, sigma_0)  # 拟合一条最佳正态分布曲线y
    y_1 = mlab.normpdf(bins, mu_1, sigma_1)  # 拟合一条最佳正态分布曲线y
    y_2 = mlab.normpdf(bins, mu_2, sigma_2)  # 拟合一条最佳正态分布曲线y
    # plt.plot(bins, y_0, 'g--', linewidth=3.5)  # 绘制y的曲线
    # plt.plot(bins, y_1, 'r--', linewidth=3.5)  # 绘制y的曲线
    plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=7.5)  # 绘制y的曲线
    plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=7.5)  # 绘制y的曲线
    plt.plot(bins, y_2, color='green', linestyle='--', linewidth=7.5)  # 绘制y的曲线
    plt.ylim(0, 20)

    # plt.xlabel('RAW-based Affinity', fontsize=25)
    # plt.xlabel('TAM-based Affinity', fontsize=25)
    # plt.ylabel('Number of Samples', size=25)
    plt.yticks(fontsize=30)
    plt.xticks(fontsize=30)
    # from matplotlib.pyplot import MultipleLocator
    # x_major_locator = MultipleLocator(0.02)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    # plt.legend(loc='upper left', fontsize=30)
    # plt.title('Amazon'.format(dataset), fontsize=25)
    # plt.title('BlogCatalog', fontsize=50)
    plt.savefig('fig/{}/{}_{}.pdf'.format(dataset, dataset, epoch))
    plt.close()


def draw_pdf_methods(method, message_normal, message_abnormal, message_real_abnormal, dataset, epoch):
    message_all = [np.squeeze(message_normal), np.squeeze(message_abnormal), np.squeeze(message_real_abnormal)]
    mu_0 = np.mean(message_all[0])  # 计算均值
    sigma_0 = np.std(message_all[0])
    # print('The mean of normal {}'.format(mu_0))
    # print('The std of normal {}'.format(sigma_0))
    mu_1 = np.mean(message_all[1])  # 计算均值
    sigma_1 = np.std(message_all[1])
    # print('The mean of abnormal {}'.format(mu_1))
    # print('The std of abnormal {}'.format(sigma_1))
    mu_2 = np.mean(message_all[2])  # 计算均值
    sigma_2 = np.std(message_all[2])
    # print('The mean of abnormal {}'.format(mu_2))
    # print('The std of abnormal {}'.format(sigma_2))

    n, bins, patches = plt.hist(message_all, bins=30, normed=1, label=['Normal', 'Outlier', 'Abnormal'])
    y_0 = mlab.normpdf(bins, mu_0, sigma_0)  # 拟合一条最佳正态分布曲线y
    y_1 = mlab.normpdf(bins, mu_1, sigma_1)  # 拟合一条最佳正态分布曲线y
    y_2 = mlab.normpdf(bins, mu_2, sigma_2)  # 拟合一条最佳正态分布曲线y
    # plt.plot(bins, y_0, 'g--', linewidth=3.5)  # 绘制y的曲线
    # plt.plot(bins, y_1, 'r--', linewidth=3.5)  # 绘制y的曲线
    plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=7.5)  # 绘制y的曲线
    plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=7.5)  # 绘制y的曲线
    plt.plot(bins, y_2, color='green', linestyle='--', linewidth=7.5)  # 绘制y的曲线
    plt.ylim(0, 8)

    # plt.xlabel('RAW-based Affinity', fontsize=25)
    # plt.xlabel('TAM-based Affinity', fontsize=25)
    # plt.ylabel('Number of Samples', size=25)

    plt.yticks(fontsize=30)
    plt.xticks(fontsize=30)
    # plt.legend(loc='upper left', fontsize=30)
    # plt.title('Amazon'.format(dataset), fontsize=25)
    # plt.title('BlogCatalog', fontsize=50)
    plt.savefig('fig/{}/{}2/{}_{}.svg'.format(method, dataset, dataset, epoch))
    plt.close()
