import time, datetime
import os
import random
import argparse
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import test_ggad, test_sage, load_data, pos_neg_split, normalize
from src.model import PCALayer
from src.layers import InterAgg, IntraAgg
from src.graphsage import *



import os


class ModelHandler(object):

    def __init__(self, config):
        args = argparse.Namespace(**config)
        # load graph, feature, and label
        # [homo, relation1, relation2, relation3], feat_data, labels = load_data(args.data_name, prefix=args.data_dir)
        homo, feat_data, labels = load_data(args.data_name, prefix=args.data_dir)
        # train_test split
        np.random.seed(args.seed)
        random.seed(args.seed)

        if args.data_name == 'dgraphfin':
            index = list(range(len(labels)))
            idx_normal = [i for i in index if labels[i] == 0]
            random.shuffle(idx_normal)
            idx_labeled = idx_normal[0: int(len(idx_normal) * 0.3)]
            idx_train = idx_labeled

            y_train = labels[idx_labeled]
            idx_rest = list(set(index).difference(set(idx_labeled)))
            y_rest = labels[idx_rest]
            idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                                    test_size=args.test_ratio,
                                                                    random_state=2, shuffle=True)

        print(f'Run on {args.data_name}, postive/total num: {np.sum(labels)}/{len(labels)}, train num {len(y_train)},' +
              f'valid num {len(y_valid)}, valid positive num {np.sum(y_valid)} , test num {len(y_test)}, test positive num {np.sum(y_test)}')
        print(f"Classification threshold: {args.thres}")
        print(f"Feature dimension: {feat_data.shape[1]}")

        # split pos neg sets for under-sampling
        train_pos, train_neg = pos_neg_split(idx_train, y_train)

        # if args.data == 'amazon':
        feat_data = normalize(feat_data)
        # train_feats = feat_data[np.array(idx_train)]
        # scaler = StandardScaler()
        # scaler.fit(train_feats)
        # feat_data = scaler.transform(feat_data)
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id

        # set input graph
        if args.model == 'SAGE' or args.model == 'GCN':
            adj_lists = homo
        else:
            adj_lists = homo
            # adj_lists = [relation1, relation2, relation3]

        print(f'Model: {args.model}, multi-relation aggregator: {args.multi_relation}, emb_size: {args.emb_size}.')

        self.args = args
        self.dataset = {'feat_data': feat_data, 'labels': labels, 'adj_lists': adj_lists, 'homo': homo,
                        'idx_train': idx_train, 'idx_valid': idx_valid, 'idx_test': idx_test,
                        'y_train': y_train, 'y_valid': y_valid, 'y_test': y_test,
                        'train_pos': train_pos, 'train_neg': train_neg, 'idx_labeled': idx_labeled,
                        }

    def train(self):
        args = self.args
        feat_data, adj_lists = self.dataset['feat_data'], self.dataset['adj_lists']
        print(type(adj_lists))

        print(type(adj_lists[0]))

        idx_train, y_train = self.dataset['idx_train'], self.dataset['y_train']
        idx_label = self.dataset['idx_labeled']
        # idx_anomaly = self.dataset['idx_anomaly']
        idx_valid, y_valid, idx_test, y_test = self.dataset['idx_valid'], self.dataset['y_valid'], self.dataset[
            'idx_test'], self.dataset['y_test']
        # initialize model input
        features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
        features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
        if args.cuda:
            features.cuda()

        # build one-layer models

        if args.model == 'SAGE':
            agg_sage = MeanAggregator(features, cuda=args.cuda)
            enc_sage = Encoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg_sage, gcn=False,
                               cuda=args.cuda)
        elif args.model == 'GCN':
            agg_gcn = GCNAggregator(features, cuda=args.cuda)
            enc_gcn = GCNEncoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg_gcn, gcn=True,
                                 cuda=args.cuda)

        if args.model == 'SAGE':
            # the vanilla GraphSAGE model as baseline
            enc_sage.num_samples = 5
            gnn_model = GraphSage(2, enc_sage)
        elif args.model == 'GCN':
            gnn_model = GCN(2, enc_gcn)

        if args.cuda:
            gnn_model.cuda()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gnn_model.parameters()), lr=args.lr,
                                     weight_decay=args.weight_decay)

        timestamp = time.time()
        timestamp = datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H-%M-%S')
        dir_saver = args.save_dir + timestamp
        path_saver = os.path.join(dir_saver, '{}_{}.pkl'.format(args.data_name, args.model))
        f1_mac_best, auc_best, ep_best = 0, 0, -1


        # train the model
        for epoch in range(args.num_epochs):
            # sampled_idx_train = pick_step(idx_train, y_train, self.dataset['homo'], size=len(self.dataset['train_pos'])*2)
            sampled_idx_train = idx_train
            # random.shuffle(sampled_idx_train)

            num_batches = int(len(sampled_idx_train) / args.batch_size) + 1
            num_batches = 20
            loss = 0.0
            epoch_time = 0
            loss_constraint_sum = 0.0
            from tqdm import tqdm
            # mini-batch training num_batches
            for batch in tqdm(range(num_batches)):

                start_time = time.time()
                i_start = batch * args.batch_size
                i_end = min((batch + 1) * args.batch_size, len(sampled_idx_train))
                batch_nodes = sampled_idx_train[i_start:i_end]
                #Sample some anomaly nodes from batch
                ratio = int(len(batch_nodes) * 0.05)
                idx_samle_abnormal = batch_nodes[-ratio:]

                #Concate existing and generation
                batch_nodes = batch_nodes + idx_samle_abnormal
                batch_label = self.dataset['labels'][np.array(batch_nodes)]
                print('batch positive num', np.sum(batch_label))

                #Construct the label
                batch_label[-ratio:] = 1
                # generated_label = np.ones(len(idx_samle_abnormal), dtype=int)
                # batch_label = np.hstack((batch_label, generated_label))

                optimizer.zero_grad()
                if args.cuda:
                    loss, loss_cls, loss_constraint, loss_rec = gnn_model.loss(batch_nodes, Variable(
                        torch.cuda.LongTensor(batch_label)))
                else:
                    loss, loss_cls, loss_constraint, loss_rec = gnn_model.loss(batch_nodes,
                                                                               Variable(torch.LongTensor(batch_label)))
                loss.backward()
                optimizer.step()
                end_time = time.time()
                epoch_time += end_time - start_time
                loss += loss.item()
                loss_constraint_sum += loss_constraint.item()

            print(f'Epoch: {epoch}, loss: {loss / num_batches}, marigin_loss: {loss_constraint_sum / num_batches}, time: {epoch_time}s')

            # Valid the model for every $valid_epoch$ epoch
            if epoch % args.valid_epochs == 0:
                if args.model == 'SAGE' or args.model == 'GCN':
                    print("Valid at epoch {}".format(epoch))
                    f1_mac_val, f1_1_val, f1_0_val, auc_val, gmean_val = test_sage(idx_valid, y_valid, gnn_model,
                                                                                   args.batch_size, args.thres)
                    if auc_val > auc_best:
                        f1_mac_best, auc_best, ep_best = f1_mac_val, auc_val, epoch
                        if not os.path.exists(dir_saver):
                            os.makedirs(dir_saver)
                        print('  Saving model ...')
                        torch.save(gnn_model.state_dict(), path_saver)
                else:
                    print("Valid at epoch {}".format(epoch))
                    f1_mac_val, f1_1_val, f1_0_val, auc_val, gmean_val = test_ggad(idx_valid, y_valid, gnn_model,
                                                                                    args.batch_size, args.thres)
                    if auc_val > auc_best:
                        f1_mac_best, auc_best, ep_best = f1_mac_val, auc_val, epoch
                        if not os.path.exists(dir_saver):
                            os.makedirs(dir_saver)
                        print('  Saving model ...')
                        torch.save(gnn_model.state_dict(), path_saver)

        print("Restore model from epoch {}".format(ep_best))
        print("Model path: {}".format(path_saver))
        gnn_model.load_state_dict(torch.load(path_saver))
        if args.model == 'SAGE' or args.model == 'GCN':
            f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = test_sage(idx_test, y_test, gnn_model,
                                                                                args.batch_size, args.thres)
        else:
            f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = test_ggad(idx_test, y_test, gnn_model,
                                                                                 args.batch_size, args.thres)
        return f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test
