import numpy as np
import scipy.sparse as sp
import torch

from model_AnomalyDAE import Model
from utils import *

from sklearn.metrics import roc_auc_score
import random
import dgl
from sklearn.metrics import  average_precision_score
import argparse
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [2]))
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Set argument
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str,
                    default='t_finance')  # ' tolokers_no_isolated 'BlogCatalog'  'Flickr'  'ACM'  'cora'  'citeseer'  'pubmed'
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--num_epoch', type=int)
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
    elif args.dataset in ['t_finance']:
        args.lr = 5e-4
    elif args.dataset in ['reddit']:
        args.lr = 1e-3
    elif args.dataset in ['photo']:
        args.lr = 3e-3
    elif args.dataset in ['elliptic']:
        args.lr = 3e-3

if args.num_epoch is None:

    if args.dataset in ['reddit']:
        args.num_epoch = 500
    elif args.dataset in ['t_finance']:
        args.num_epoch = 1500
    elif args.dataset in ['Amazon']:
        args.num_epoch = 800
    elif args.dataset in ['photo']:
        args.num_epoch = 500
    elif args.dataset in ['elliptic']:
        args.num_epoch = 500

batch_size = args.batch_size
subgraph_size = args.subgraph_size

print('Dataset: ', args.dataset)

# Set random seed
dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



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
# nb_classes = labels.shape[1]
raw_adj = adj
adj = normalize_adj(adj)
adj = (adj + sp.eye(adj.shape[0])).todense()
raw_adj = (raw_adj + sp.eye(raw_adj.shape[0])).todense()

features = torch.FloatTensor(features[np.newaxis])
adj = torch.FloatTensor(adj[np.newaxis])
raw_adj = torch.FloatTensor(raw_adj[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])

# idx_train = torch.LongTensor(idx_train)
# idx_val = torch.LongTensor(idx_val)
# idx_test = torch.LongTensor(idx_test)

# Initialize model and optimiser
model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# if torch.cuda.is_available():
#     print('Using CUDA')
#     model.cuda()
#     features = features.cuda()
#     adj = adj.cuda()
#     labels = labels.cuda()

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
    total_time = 0
    pbar.set_description('Training')
    for epoch in range(args.num_epoch):
        start_time = time.time()
        model.train()
        optimiser.zero_grad()

        # Train model
        loss, score = model(features, adj, normal_label_idx, idx_test)
        # loss, score = model(features, adj, all_idx, idx_test)
        loss.backward()
        optimiser.step()

        if epoch % 2 == 0:
            print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(loss.item()))

        if epoch % 5 == 0:
             model.eval()
             # loss, score = model(features, adj, normal_label_idx, idx_test)
             score = np.array(score.detach().cpu())
             auc = roc_auc_score(ano_label[idx_test], score)
             print('Testing {} AUC:{:.4f}'.format(args.dataset, auc))
             AP = average_precision_score(ano_label[idx_test], score, average='macro', pos_label=1, sample_weight=None)
             print('Testing AP:', AP)
             print('Total time is', total_time)

        end_time = time.time()
        total_time += end_time - start_time

