import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
import dgl

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


def normalize_adj_tensor(raw_adj):
    adj = raw_adj[0, :, :]
    row_sum = torch.sum(adj, 0)
    r_inv = torch.pow(row_sum, -0.5).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    adj = torch.mm(adj, torch.diag_embed(r_inv))
    adj = torch.mm(torch.diag_embed(r_inv), adj)
    adj = adj.unsqueeze(0)
    return adj


def normalize_score(ano_score):
    ano_score = ((ano_score - np.min(ano_score)) / (
            np.max(ano_score) - np.min(ano_score)))
    return ano_score


def process_dis(init_value, cutting_dis_array):
    r_inv = np.power(init_value, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    cutting_dis_array = cutting_dis_array.dot(sp.diags(r_inv))
    cutting_dis_array = sp.diags(r_inv).dot(cutting_dis_array)
    return cutting_dis_array


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

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score


def draw_roc(ano_label, message):
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr[0], tpr[0], _ = roc_curve(ano_label, message)
    roc_auc[0] = auc(fpr[0], tpr[0])
    lw = 2
    plt.yticks(fontsize=25)
    plt.xticks(fontsize=25)
    # plt.plot(fpr[0], tpr[0],
    #          lw=lw, label=' (AUROC = %0.2f)' % roc_auc[0], linewidth=2.5)
    plt.plot(fpr[0], tpr[0],
             lw=lw, label=' (AUROC = 0.564)', linewidth=2.5)
    plt.legend(loc="lower right", prop={'size': 20})
    # plt.show()
    plt.savefig('roc1.svg', dpi=500, bbox_inches='tight')
    plt.cla()
    print("save suceess")


from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score


def draw_pr(ano_label, message):
    lr_precision, lr_recall, _ = precision_recall_curve(ano_label, message)
    #   plt.plot([0,1], [no_skill, no_skill], linestyle='--')
    # plt.plot(lr_recall, lr_precision, lw=2,
    #          label=' (AUPRC = %0.2f)' % average_precision_score(ano_label, message), linewidth=2.5)
    plt.plot(lr_recall, lr_precision, lw=2,
             label=' (AUPRC = 0.0764)', linewidth=2.5)

    fontsize = 14
    # plt.xlabel('Recall', fontsize=fontsize)
    # plt.ylabel('Precision', fontsize=fontsize)
    # plt.title('Precision Recall Curve')
    plt.yticks(fontsize=25)
    plt.xticks(fontsize=25)
    plt.legend(prop={'size': 20})
    # plt.show()
    plt.savefig('pr1.svg', dpi=500, bbox_inches='tight')
    print("save suceess")
def load_mat(dataset):
    """Load .mat dataset."""

    data = sio.loadmat("./data/{}.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']

    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)

    ano_labels = np.squeeze(np.array(label))
    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None
    num_node = adj.shape[0]
    train_rate = 0.3
    val_rate = 0.1
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[: num_train]
    idx_val = all_idx[num_train: num_train + num_val]
    idx_test = all_idx[num_train + num_val:]
    all_normal_label_idx = [i for i in idx_train if ano_labels[i] == 0]
    normal_label_idx = all_normal_label_idx[: int(len(all_normal_label_idx) * 0.8)]

    # contamination
    real_abnormal_id = np.array(all_idx)[np.argwhere(ano_labels == 1).squeeze()].tolist()
    add_rate = 0.15 * len(real_abnormal_id)
    random.shuffle(real_abnormal_id)
    add_abnormal_id = real_abnormal_id[:int(add_rate)]
    normal_label_idx = normal_label_idx + add_abnormal_id
    idx_test = np.setdiff1d(idx_test, add_abnormal_id, False)

    return adj, feat, ano_labels, str_ano_labels, attr_ano_labels, normal_label_idx, idx_test


def adj_to_dgl_graph(adj):
    """Convert adjacency matrix to dgl format."""
    nx_graph = nx.from_scipy_sparse_matrix(adj)
    dgl_graph = dgl.DGLGraph(nx_graph)
    return dgl_graph


# compute the distance between each node
def calc_distance(adj, seq):
    dis_array = torch.zeros((adj.shape[0], adj.shape[1]))
    row = adj.shape[0]
    for i in range(row):
        print(i)
        node_index = torch.argwhere(adj[i, :] > 0)
        for j in node_index:
            dis = torch.sqrt(torch.sum((seq[i] - seq[j]) * (seq[i] - seq[j])))
            dis_array[i][j] = dis
    return dis_array


def get_cos_similar(v1: list, v2: list):
    num = float(np.dot(v1, v2))  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0


def calc_sim(adj_matrix, attr_matrix):
    row = adj_matrix.shape[0]
    col = adj_matrix.shape[1]
    dis_array = np.zeros((row, col))
    for i in range(row):
        # print(i)
        node_index = np.argwhere(adj_matrix[i, :] > 0)[:, 0]
        for j in node_index:
            dis = get_cos_similar(attr_matrix[i].tolist(), attr_matrix[j].tolist())
            dis_array[i][j] = dis

    return dis_array


def graph_nsgt(dis_array, adj):
    # dis_array = dis_array.cuda()
    row = dis_array.shape[0]
    dis_array_u = dis_array * adj
    mean_dis = dis_array_u[dis_array_u != 0].mean()
    for i in range(row):
        node_index = torch.argwhere(adj[i, :] > 0)
        if node_index.shape[0] != 0:
            max_dis = dis_array[i, node_index].max()
            min_dis = mean_dis
            if max_dis > min_dis:
                random_value = (max_dis - min_dis) * np.random.random_sample() + min_dis
                cutting_edge = torch.argwhere(dis_array[i, node_index[:, 0]] > random_value)
                if cutting_edge.shape[0] != 0:
                    adj[i, node_index[cutting_edge[:, 0]]] = 0

    adj = adj + adj.T
    adj[adj > 1] = 1
    return adj


import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = (8.5, 7.5)
from matplotlib.backends.backend_pdf import PdfPages

def draw_pdf(message, ano_label, dataset):
    with PdfPages('{}-TAM.pdf'.format(dataset)) as pdf:
        normal_message_all = message[ano_label == 0]
        abnormal_message_all = message[ano_label == 1]
        message_all = [normal_message_all, abnormal_message_all]
        mu_0 = np.mean(message_all[0])
        sigma_0 = np.std(message_all[0])
        print('The mean of normal {}'.format(mu_0))
        print('The std of normal {}'.format(sigma_0))
        mu_1 = np.mean(message_all[1])
        sigma_1 = np.std(message_all[1])
        print('The mean of abnormal {}'.format(mu_1))
        print('The std of abnormal {}'.format(sigma_1))
        n, bins, patches = plt.hist(message_all, bins=30, normed=1, label=['Normal', 'Abnormal'])
        y_0 = mlab.normpdf(bins, mu_0, sigma_0)
        y_1 = mlab.normpdf(bins, mu_1, sigma_1)
        plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=7.5)
        plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=7.5)
        plt.yticks(fontsize=30)
        plt.xticks(fontsize=30)
        plt.legend(loc='upper left', fontsize=30)
        plt.title(''.format(dataset), fontsize=25)
        plt.show()


def draw_pdf_str_attr(message, ano_label, str_ano_label, attr_ano_label, dataset):
    with PdfPages('{}-TAM.pdf'.format(dataset)) as pdf:
        normal_message_all = message[ano_label == 0]
        str_abnormal_message_all = message[str_ano_label == 1]
        attr_abnormal_message_all = message[attr_ano_label == 1]
        message_all = [normal_message_all, str_abnormal_message_all, attr_abnormal_message_all]

        mu_0 = np.mean(message_all[0])
        sigma_0 = np.std(message_all[0])
        print('The mean of normal {}'.format(mu_0))
        print('The std of normal {}'.format(sigma_0))
        mu_1 = np.mean(message_all[1])
        sigma_1 = np.std(message_all[1])
        print('The mean of str_abnormal {}'.format(mu_1))
        print('The std of str_abnormal {}'.format(sigma_1))
        mu_2 = np.mean(message_all[2])
        sigma_2 = np.std(message_all[2])
        print('The mean of attt_abnormal {}'.format(mu_2))
        print('The std of attt_abnormal {}'.format(sigma_2))
        n, bins, patches = plt.hist(message_all, bins=30, normed=1, label=['Normal', 'Structural Abnormal', 'Contextual Abnormal'])
        y_0 = mlab.normpdf(bins, mu_0, sigma_0)
        y_1 = mlab.normpdf(bins, mu_1, sigma_1)
        y_2= mlab.normpdf(bins, mu_2, sigma_2)  #

        plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=3.5)
        plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=3.5)
        plt.plot(bins, y_2, color='green', linestyle='--', linewidth=3.5)

        plt.xlabel('TAM-based Affinity', fontsize=25)
        plt.ylabel('Number of Samples', size=25)
        plt.yticks(fontsize=25)
        plt.xticks(fontsize=25)
        plt.legend(loc='upper left', fontsize=18)
        plt.title('{}'.format(dataset), fontsize=25)
        plt.show()



