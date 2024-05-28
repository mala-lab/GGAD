import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from model_AEGIS import Model
from utils import *

from sklearn.metrics import roc_auc_score
import random
import os
import dgl
from sklearn.metrics import precision_recall_curve, average_precision_score
import argparse
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [2]))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Set argument
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str,
                    default='reddit')  # ' tolokers_no_isolated 'BlogCatalog'  'Flickr'  'ACM'  'cora'  'citeseer'  'pubmed'
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--recon_num_epoch', type=int, default=10)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--readout', type=str, default='avg')  # max min avg  weighted_sum
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--negsamp_ratio', type=int, default=1)

args = parser.parse_args()

if args.lr is None:
    if args.dataset in ['Amazon']:
        args.lr = 1e-3
    elif args.dataset in ['tf_finace']:
        args.lr = 5e-4
    elif args.dataset in ['reddit']:
        args.lr = 1e-3

if args.num_epoch is None:

    if args.dataset in ['reddit']:
        args.num_epoch = 500
    elif args.dataset in ['tf_finace']:
        args.num_epoch = 1500
    elif args.dataset in ['Amazon']:
        args.num_epoch = 800

batch_size = args.batch_size
subgraph_size = args.subgraph_size

print('Dataset: ', args.dataset)

# Set random seed
dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)
# torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)

# Load and preprocess data
adj, features, labels, all_idx, idx_train, idx_val, \
idx_test, ano_label, str_ano_label, attr_ano_label, normal_label_idx, abnormal_label_idx = load_mat(args.dataset)
if args.dataset in ['Amazon', 'tf_finace', 'reddit', 'elliptic']:
    features, _ = preprocess_features(features)
else:
    features = features.todense()
dgl_graph = adj_to_dgl_graph(adj)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
raw_adj = adj
adj = normalize_adj(adj)


raw_adj = (raw_adj + sp.eye(raw_adj.shape[0])).todense()
adj = (adj + sp.eye(adj.shape[0])).todense()

features = torch.FloatTensor(features[np.newaxis])
adj = torch.FloatTensor(adj[np.newaxis])
raw_adj = torch.FloatTensor(raw_adj[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])
# idx_train = torch.LongTensor(idx_train)
# idx_val = torch.LongTensor(idx_val)
# idx_test = torch.LongTensor(idx_test)

# Initialize model and optimiser
model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
optimiser_ae = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.weight_decay)
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimiser_gen = torch.optim.Adam(model.generator.parameters(),
                                 lr=args.lr)
if torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()

    # idx_train = idx_train.cuda()
    # idx_val = idx_val.cuda()
    # idx_test = idx_test.cuda()

cnt_wait = 0
best = 1e9
best_t = 0
batch_num = nb_nodes // batch_size + 1
import time

# Train model
with tqdm(total=args.num_epoch) as pbar:
    pbar.set_description('Training')

    for epoch in range(args.recon_num_epoch):
        loss_dis, loss_g, loss_ae, score_test, emb_all = model(features, adj, idx_train, idx_test)
        # loss_dis, loss_g, loss_ae, score_test, emb_all = model(features, adj, all_idx, idx_test)
        loss_ae.backward()
        optimiser_ae.step()
        print("Epoch:", '%04d' % (epoch), "ae_loss=", "{:.5f}".format(loss_ae.item()))
    total_time = 0
    for epoch in range(args.num_epoch):
        start_time = time.time()
        model.train()
        optimiser.zero_grad()
        optimiser_gen.zero_grad()
        # Train model
        # loss_dis, loss_g, loss_ae, score_test, emb_all = model(features, adj, idx_train, idx_test)
        loss_dis, loss_g, loss_ae, score_test, emb_all = model(features, adj, all_idx, idx_test)
        loss_g.backward(retain_graph=True)
        loss_dis.backward(retain_graph=True)
        # loss = loss_dis + loss_g
        optimiser.step()
        optimiser_gen.step()
        score_test = np.array(score_test.detach().cpu())

        emb_inf = torch.norm(emb_all, dim=-1, keepdim=True)
        emb_inf = torch.pow(emb_inf, -1)
        emb_inf[torch.isinf(emb_inf)] = 0.
        emb_norm = emb_all * emb_inf

        sim_matrix = torch.mm(emb_norm, emb_norm.T)
        raw_adj = torch.squeeze(raw_adj).cuda()
        similar_matrix1 = sim_matrix[:int(raw_adj.shape[0]), :int(raw_adj.shape[1])] * raw_adj
        similar_matrix2 = sim_matrix[int(raw_adj.shape[0]):, int(raw_adj.shape[1]):] * raw_adj

        r_inv = torch.pow(torch.sum(raw_adj, 0), -1)
        r_inv[torch.isinf(r_inv)] = 0.
        affinity1 = torch.sum(similar_matrix1, 0) * r_inv
        affinity2 = torch.sum(similar_matrix2, 0) * r_inv

        if epoch % 20 == 0:
            # save data for tsne
            import scipy.io as io

            # tsne_data_path = 'draw/AEGIS2_tfinance/tsne_data_{}.mat'.format(str(epoch))
            # # io.savemat(tsne_data_path, {'emb': np.array(emb_all.cpu().detach()), 'ano_label': ano_label,
            # #                             'abnormal_label_idx': np.array(abnormal_label_idx),
            # #                             'normal_label_idx': np.array(normal_label_idx)})
            real_abnormal_label_idx = np.array(all_idx)[np.argwhere(ano_label == 1).squeeze()].tolist()
            real_normal_label_idx = np.array(all_idx)[np.argwhere(ano_label == 0).squeeze()].tolist()

            # real_abnormal_label_idx = real_abnormal_label_idx[:50]

            real_affinity, index = torch.sort(affinity1[real_abnormal_label_idx])
            real_affinity = real_affinity[:50]


            draw_pdf_methods('AEGIS', np.array(affinity1[real_normal_label_idx].detach().cpu()),
                             np.array(affinity2[:500].detach().cpu()),
                             np.array(real_affinity.detach().cpu()), args.dataset, epoch)

        # if epoch % 10 == 0:
        #     real_abnormal_label_idx = np.array(all_idx)[np.argwhere(ano_label == 1).squeeze()].tolist()

        # extend_label = torch.zeros(emb_combine.size(0), 1)
        # extend_label[abnormal_label_idx] = 1
        # extend_label[real_abnormal_label_idx] = 2

        # data_dict = dict([('embedding', emb_combine), ('Label', extend_label)])
        #
        # scio.savemat('embedding/{}_{}.mat'.format(args.dataset, epoch), data_dict)
        #
        # draw_pdf(np.array(affinity[normal_label_idx].detach()),
        #          np.array(affinity[abnormal_label_idx].detach()),
        #          np.array(affinity[real_abnormal_label_idx].detach()), args.dataset, epoch)

        if epoch % 5 == 0:
            print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(loss_dis.item()))
            model.eval()
            auc = roc_auc_score(ano_label[idx_test], score_test)
            print('Testing {} AUC:{:.4f}'.format(args.dataset, auc))
            AP = average_precision_score(ano_label[idx_test], score_test, average='macro', pos_label=1,
                                         sample_weight=None)
            print('Testing AP:', AP)
            print('Total time is', total_time)

        end_time = time.time()
        total_time += end_time - start_time
