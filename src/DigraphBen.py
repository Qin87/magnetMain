# external files
import numpy as np
import pickle as pk
import torch.optim as optim
from datetime import datetime
import os, time, argparse, csv
from collections import Counter
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.datasets import WebKB, WikipediaNetwork, WikiCS
import tqdm

# internal files
from gens_GraphSHA import sampling_idx_individual_dst, sampling_node_source, neighbor_sampling
# from gens import sampling_idx_individual_dst, sampling_node_source, neighbor_sampling
from layer.DiGCN import *
from src.data_utils import make_longtailed_data_remove, get_idx_info, CrossEntropy, generate_masks, keep_all_data
from src.gens_GraphSHA import neighbor_sampling_bidegree, saliency_mixup, duplicate_neighbor, test_directed
from src.neighbor_dist import get_PPR_adj, get_heat_adj, get_ins_neighbor_dist
from utils.Citation import *
from layer.geometric_baselines import *
from torch_geometric.utils import to_undirected
from utils.preprocess import geometric_dataset, load_syn
from utils.save_settings import write_log
from utils.hermitian import hermitian_decomp
from utils.edge_data import get_appr_directed_adj, get_second_directed_adj

# select cuda device if available
cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="baseline--Digraph")

    parser.add_argument('--MakeImbalance', type=bool, default=True, help='True for turn dataset into imbalanced')
    parser.add_argument('--CustomizeMask', type=bool, default=False,
                        help='True for generate train,val,test splits by me')

    parser.add_argument('--log_root', type=str, default='../logs/',
                        help='the path saving model.t7 and the training process')
    parser.add_argument('--log_path', type=str, default='test',
                        help='the path saving model.t7 and the training process, the name of folder will be log/(current time)')
    parser.add_argument('--data_path', type=str, default='../dataset/data/tmp/',
                        help='data set folder, for default format see dataset/cora/cora.edges and cora.node_labels')
    parser.add_argument('--dataset', type=str, default='WebKB/Cornell', help='data set selection')

    parser.add_argument('--epochs', type=int, default=1500, help='training epochs')
    parser.add_argument('--num_filter', type=int, default=2, help='num of filters')
    parser.add_argument('--p_q', type=float, default=0.95, help='direction strength, from 0.5 to 1.')
    parser.add_argument('--p_inter', type=float, default=0.1, help='inter_cluster edge probabilities.')
    parser.add_argument('--method_name', type=str, default='DiG', help='method name')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for training testing split/random graph generation')

    parser.add_argument('--dropout', type=float, default=0.0, help='dropout prob')
    parser.add_argument('--debug', '-D', action='store_true', help='debug mode')
    parser.add_argument('--new_setting', '-NS', action='store_true', help='whether not to load best settings')

    parser.add_argument('--layer', type=int, default=2, help='number of layers (2 or 3), default: 2')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--l2', type=float, default=5e-4, help='l2 regularizer')

    parser.add_argument('--alpha', type=float, default=0.1, help='alpha teleport prob')
    parser.add_argument('--randomseed', type=int, default=-1, help='if set random seed in training')
    parser.add_argument('--withAug', type=bool, default=True, help='with Aug or not')
    parser.add_argument('--AugDirect', type=int, default=1, help='1 for one direction, 2 for bidirection aug edges')
    parser.add_argument('--imb_ratio', type=float, default=100, help='imbalance ratio')
    parser.add_argument('--net', type=str, choices=['GCN', 'GAT', 'SAGE'], default='GCN', help='GNN bachbone')
    parser.add_argument('--n_layer', type=int, default=2, help='the number of layers')
    parser.add_argument('--feat_dim', type=int, default=64, help='feature dimension')
    parser.add_argument('--warmup', type=int, default=5, help='warmup epoch')
    parser.add_argument('--epoch', type=int, default=900, help='epoch')
    # parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--tau', type=int, default=2,
                        help='temperature in the sofax function when calculating confidence-based node hardness')
    parser.add_argument('--max', action="store_true",
                        help='synthesizing to max or mean num of training set. default is mean')
    parser.add_argument('--no_mask', action="store_true",
                        help='whether to mask the self class in sampling neighbor classes. default is mask')
    parser.add_argument('--gdc', type=str, choices=['ppr', 'hk', 'none'], default='ppr',
                        help='how to convert to weighted graph')
    return parser.parse_args()


def acc(pred, label, mask):
    correct = int(pred[mask].eq(label[mask]).sum().item())
    acc = correct / int(mask.sum())
    return acc


def main(args):
    if args.randomseed > 0:
        torch.manual_seed(args.randomseed)

    date_time = datetime.now().strftime('%m-%d-%H:%M:%S')
    log_path = os.path.join(args.log_root, args.log_path, args.save_name, date_time)
    if os.path.isdir(log_path) is False:
        try:
            os.makedirs(log_path)
        except FileExistsError:
            print('Folder exists!')
    load_func, subset = args.dataset.split('/')[0], args.dataset.split('/')[1]
    if load_func == 'WebKB':
        load_func = WebKB
        dataset = load_func(root=args.data_path, name=subset)
    elif load_func == 'WikipediaNetwork':
        load_func = WikipediaNetwork
        dataset = load_func(root=args.data_path, name=subset)
    elif load_func == 'WikiCS':
        load_func = WikiCS
        dataset = load_func(root=args.data_path)
    elif load_func == 'cora_ml':
        dataset = citation_datasets(root='../dataset/data/tmp/cora_ml/cora_ml.npz')
    elif load_func == 'citeseer_npz':
        dataset = citation_datasets(root='../dataset/data/tmp/citeseer_npz/citeseer_npz.npz')
    else:
        dataset = load_syn(args.data_path + args.dataset, None)

    if os.path.isdir(log_path) is False:
        os.makedirs(log_path)

    data = dataset[0]
    # results = np.zeros((1, 4))

    global class_num_list, idx_info, prev_out, sample_times
    global data_train_mask, data_val_mask, data_test_mask  # data split: train, validation, test
    if not data.__contains__('edge_weight'):
        data.edge_weight = None
    else:
        data.edge_weight = torch.FloatTensor(data.edge_weight)
    data.y = data.y.long()
    num_classes = (data.y.max() - data.y.min() + 1).detach().numpy()

    # copy GraphSHA
    if args.dataset.split('/')[0].startswith('dgl'):
        edges = torch.cat((data.edges()[0].unsqueeze(0), data.edges()[1].unsqueeze(0)), dim=0)
        data_y = data.ndata['label']
        data_train_mask, data_val_mask, data_test_mask = (
        data.ndata['train_mask'].clone(), data.ndata['val_mask'].clone(), data.ndata['test_mask'].clone())
        data_x = data.ndata['feat']
        # print(data_x.shape, data.num_nodes)  # torch.Size([3327, 3703])
        dataset_num_features = data_x.shape[1]
    else:
        edges = data.edge_index  # for torch_geometric librar
        data_y = data.y
        # data_train_mask, data_val_mask, data_test_mask = (data.train_mask[:,0].clone(), data.val_mask[:,0].clone(),
        #                                                   data.test_mask[:,0].clone())
        data_train_mask, data_val_mask, data_test_mask = (data.train_mask.clone(), data.val_mask.clone(),
                                                          data.test_mask.clone())
        # print("how many val,,", data_val_mask.sum())   # how many val,, tensor(59)
        data_x = data.x
        dataset_num_features = dataset.num_features


    IsDirectedGraph = test_directed(edges)
    print("This is directed graph: ", IsDirectedGraph)
    # print(torch.sum(data_train_mask), torch.sum(data_val_mask), torch.sum(data_test_mask), data_train_mask.shape,
    #       data_val_mask.shape, data_test_mask.shape)  # tensor(11600) tensor(35380) tensor(5847) torch.Size([11701, 20])
    print("data_x", data_x.shape)  # [11701, 300])

    n_cls = data_y.max().item() + 1
    data = data.to(device)



    criterion = CrossEntropy().to(device)

    # optimizer = torch.optim.Adam([dict(params=model.reg_params, weight_decay=5e-4),
    #                               dict(params=model.non_reg_params, weight_decay=0), ], lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100,
    #                                                        verbose=False)

    # normalize label, the minimum should be 0 as class index
    splits = data.train_mask.shape[1]
    print("splits", splits)
    results = np.zeros((splits, 4))
    if len(data_test_mask.shape) == 1:
        data_test_mask = data_test_mask.unsqueeze(1).repeat(1, splits)
    # print(data.edge_index.shape)    # torch.Size([2, 298])
    edge_index1, edge_weights1 = get_appr_directed_adj(args.alpha, edges.long(), data_y.size(-1), data_x.dtype)
    # print("edge_index1", edge_index1.shape)    # torch.Size([2, 737])
    # print("edge_weight1", edge_weights1)
    edge_index1 = edge_index1.to(device)
    edge_weights1 = edge_weights1.to(device)
    if args.method_name[-2:] == 'ib':
        edge_index2, edge_weights2 = get_second_directed_adj(edges.long(), data_y.size(-1), data_x.dtype)
        edge_index2 = edge_index2.to(device)
        edge_weights2 = edge_weights2.to(device)
        SparseEdges = (edge_index1, edge_index2)
        edge_weight = (edge_weights1, edge_weights2)
        del edge_index2, edge_weights2
    else:
        SparseEdges = edge_index1
        edge_weight = edge_weights1
    # print("edge_weight", edge_weight.shape, data.y.shape)
    del edge_index1, edge_weights1
    data = data.to(device)
    # results = np.zeros((splits, 4))
    for split in range(splits):
        data_train_mask, data_val_mask, data_test_mask = (data.train_mask[:, split].clone(),
                                                          data.val_mask[:, split].clone(),data.test_mask[:,split].clone())

        if args.CustomizeMask:
            ratio_val2train = 3
            minTrainClass = 60
            data_train_mask, data_val_mask, data_test_mask = generate_masks(data_y, minTrainClass, ratio_val2train)

        stats = data_y[data_train_mask]  # this is selected y. only train nodes of y
        n_data = []  # num of train in each class
        for i in range(n_cls):
            data_num = (stats == i).sum()
            n_data.append(int(data_num.item()))

        idx_info = get_idx_info(data_y, n_cls, data_train_mask)  # torch: all train nodes for each class
        if args.MakeImbalance:
            class_num_list, data_train_mask, idx_info, train_node_mask, train_edge_mask = \
                make_longtailed_data_remove(edges, data_y, n_data, n_cls, args.imb_ratio, data_train_mask.clone())
        # print("Let me see:", torch.sum(data_train_mask))
        else:
            class_num_list, data_train_mask, idx_info, train_node_mask, train_edge_mask = \
                keep_all_data(edges, data_y, n_data, n_cls, args.imb_ratio, data_train_mask)

        train_idx = data_train_mask.nonzero().squeeze()  # get the index of training data
        labels_local = data_y.view([-1])[train_idx]  # view([-1]) is "flattening" the tensor.
        train_idx_list = train_idx.cpu().tolist()
        local2global = {i: train_idx_list[i] for i in range(len(train_idx_list))}
        global2local = dict([val, key] for key, val in local2global.items())
        idx_info_list = [item.cpu().tolist() for item in idx_info]  # list of all train nodes for each class
        idx_info_local = [torch.tensor(list(map(global2local.get, cls_idx))) for cls_idx in
                          idx_info_list]  # train nodes position inside train

        if args.gdc == 'ppr':
            neighbor_dist_list = get_PPR_adj(data_x, edges[:, train_edge_mask], alpha=0.05, k=128, eps=None)
        elif args.gdc == 'hk':
            neighbor_dist_list = get_heat_adj(data_x, edges[:, train_edge_mask], t=5.0, k=None, eps=0.0001)
        elif args.gdc == 'none':
            neighbor_dist_list = get_ins_neighbor_dist(data_y.size(0), edges[:, train_edge_mask], data_train_mask,
                                                       device)

        log_str_full = ''
        if not args.method_name[-2:] == 'ib':
            model = DiModel(data.x.size(-1), num_classes, filter_num=args.num_filter,
                                    dropout=args.dropout, layer=args.layer).to(device)
        else:
            model = DiGCN_IB(data.x.size(-1), hidden=args.num_filter,
                                    num_classes=num_classes, dropout=args.dropout,
                                    layer=args.layer).to(device)
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=100,
                                                               verbose=False)
    #
    #     #################################
    #     # Train/Validation/Test
    #     #################################
        best_test_err = 1000.0
        early_stopping = 0
        for epoch in tqdm.tqdm(range(args.epoch)):
            # for epoch in range(args.epochs):
            start_time = time.time()
            ####################
            # Train
            ####################
            # for loop for batch loading

            model.train()
            opt.zero_grad()  # clear the gradients of the model's parameters.

            if args.withAug:
                if epoch > args.warmup:
                    # print("After warmup")
                    # identifying source samples
                    prev_out_local = prev_out[train_idx]
                    sampling_src_idx, sampling_dst_idx = sampling_node_source(class_num_list, prev_out_local,
                                                                              idx_info_local, train_idx, args.tau,
                                                                              args.max, args.no_mask)
                    if args.AugDirect == 1:
                        new_edge_index = neighbor_sampling(data_x.size(0), edges[:, train_edge_mask], sampling_src_idx,
                                                           neighbor_dist_list)
                    elif args.AugDirect == 2:
                        new_edge_index = neighbor_sampling_bidegree(data_x.size(0), edges[:, train_edge_mask],
                                                                    sampling_src_idx, neighbor_dist_list)
                    else:
                        pass
                    beta = torch.distributions.beta.Beta(1, 100)
                    lam = beta.sample((len(sampling_src_idx),)).unsqueeze(1)
                    new_x = saliency_mixup(data_x, sampling_src_idx, sampling_dst_idx, lam)

                    add_num = new_x.shape[0] - data_x.shape[0]  # Ben
                    new_train_mask = torch.ones(add_num, dtype=torch.bool, device=data_x.device)
                    new_train_mask = torch.cat((data_train_mask, new_train_mask), dim=0)  # add some train nodes
                    _new_y = data_y[sampling_src_idx.long()].clone()
                    # print(data_x.shape, new_x.shape, add_num)  # torch.Size([183, 1703]) torch.Size([542, 1703]) 359
                    # new_y = torch.cat((data_y[data_train_mask], _new_y), dim=0)    #
                    new_y = torch.cat((data_y, _new_y), dim=0)  #
                    # print("y:", new_y.shape)  # y: torch.Size([542])

                    # get edge_weight
                    edge_index1, edge_weights1 = get_appr_directed_adj(args.alpha, new_edge_index.long(),
                                                                       new_y.size(-1), new_x.dtype)
                    # print("edge_index1", edge_index1.shape)  # torch.Size([2, 737])
                    # print("edge_weight1", edge_weights1)
                    edge_index1 = edge_index1.to(device)
                    edge_weights1 = edge_weights1.to(device)
                    if args.method_name[-2:] == 'ib':
                        edge_index2, edge_weights2 = get_second_directed_adj(new_edge_index.long(), new_y.size(-1),
                                                                             new_x.dtype)
                        edge_index2 = edge_index2.to(device)
                        edge_weights2 = edge_weights2.to(device)
                        new_SparseEdges = (edge_index1, edge_index2)
                        new_edge_weight = (edge_weights1, edge_weights2)
                        del edge_index2, edge_weights2
                    else:
                        new_SparseEdges = edge_index1
                        new_edge_weight = edge_weights1
                    # print("edge_weight", new_edge_weight.shape, data.y.shape)
                    del edge_index1, edge_weights1

                else:
                    sampling_src_idx, sampling_dst_idx = sampling_idx_individual_dst(class_num_list, idx_info, device)
                    # print(len(sampling_src_idx), sampling_src_idx)  # 359 tensor([  58,   58,
                    beta = torch.distributions.beta.Beta(2, 2)
                    lam = beta.sample((len(sampling_src_idx),)).unsqueeze(1)
                    # print(torch.sum(train_edge_mask))
                    # print(SparseEdges.shape,train_edge_mask.shape)  # torch.Size([2, 737]) torch.Size([298])
                    new_edge_index = duplicate_neighbor(data_x.size(0), edges[:, train_edge_mask], sampling_src_idx)
                    # print("New edge:,", new_edge_index.shape)   # New edge:, torch.Size([2, 470])
                    new_x = saliency_mixup(data_x, sampling_src_idx, sampling_dst_idx, lam)

                    # add_num = new_x.shape[0] - data_x.shape[0]  # Ben
                    add_num = len(sampling_src_idx)  # Ben
                    new_train_mask = torch.ones(add_num, dtype=torch.bool, device=data_x.device)
                    new_train_mask = torch.cat((data_train_mask, new_train_mask), dim=0)  # add some train nodes
                    _new_y = data_y[sampling_src_idx].clone()
                    # print(data_x.shape, new_x.shape, add_num)  # torch.Size([183, 1703]) torch.Size([542, 1703]) 359
                    # new_y = torch.cat((data_y[data_train_mask], _new_y), dim=0)    #
                    new_y = torch.cat((data_y, _new_y), dim=0)    #
                    # print("y:", new_y.shape)    # y: torch.Size([542])

                    # get SparseEdges and edge_weight
                    edge_index1, edge_weights1 = get_appr_directed_adj(args.alpha, new_edge_index.long(),
                                                                       new_y.size(-1), new_x.dtype)
                    # print("edge_index1", edge_index1.shape)  # torch.Size([2, 737])
                    # print("edge_weight1", edge_weights1)
                    edge_index1 = edge_index1.to(device)
                    edge_weights1 = edge_weights1.to(device)
                    if args.method_name[-2:] == 'ib':
                        edge_index2, edge_weights2 = get_second_directed_adj(new_edge_index.long(), new_y.size(-1),
                                                                             new_x.dtype)
                        edge_index2 = edge_index2.to(device)
                        edge_weights2 = edge_weights2.to(device)
                        new_SparseEdges = (edge_index1, edge_index2)
                        new_edge_weight = (edge_weights1, edge_weights2)
                        del edge_index2, edge_weights2
                    else:
                        new_SparseEdges = edge_index1
                        new_edge_weight = edge_weights1
                    # print("edge_weight", new_edge_weight.shape, data.y.shape)
                    del edge_index1, edge_weights1

                # print(new_x[0][:100], new_SparseEdges.shape, new_edge_weight.shape)   # torch.Size([2, 1918]) torch.Size([1918])
                out = model(new_x, new_SparseEdges, new_edge_weight)   #
                prev_out = (out[:data_x.size(0)]).detach().clone()

                # add_num = len(sampling_src_idx)  # Ben
                # new_train_mask = torch.ones(add_num, dtype=torch.bool, device=data_x.device)
                # new_train_mask = torch.cat((data_train_mask, new_train_mask), dim=0)  # add some train nodes
                _new_y = data_y[sampling_src_idx.long()].clone()
                # print(data_x.shape, new_x.shape, add_num)  # torch.Size([183, 1703]) torch.Size([542, 1703]) 359
                # new_y = torch.cat((data_y[data_train_mask], _new_y), dim=0)
                new_y_train = torch.cat((data_y[data_train_mask], _new_y), dim=0)
                criterion(out[new_train_mask], new_y_train).backward()
            # # without aug
            else:
                out = model(data_x, SparseEdges, edge_weight)
                # print(out[data_train_mask].shape, '\n', y.shape)  # torch.Size([250, 6]) torch.Size([250])
                criterion(out[data_train_mask], data_y[data_train_mask]).backward()

            # with torch.no_grad():
            #     model.eval()
            #
            #     # val_y = data_y[data_val_mask]
            #     # val_x = data_x[data_val_mask]
            #     # val_edge_index = edges[:, train_edge_mask]
            #
            #     # # get SparseEdges and edge_weight
            #     # edge_index1, edge_weights1 = get_appr_directed_adj(args.alpha, data.edge_index.long(), data.y.size(-1),
            #     #                                                    data.x.dtype)
            #     # print("edge_index1", edge_index1.shape)  # torch.Size([2, 737])
            #     # # print("edge_weight1", edge_weights1)
            #     # edge_index1 = edge_index1.to(device)
            #     # edge_weights1 = edge_weights1.to(device)
            #     # if args.method_name[-2:] == 'ib':
            #     #     edge_index2, edge_weights2 = get_second_directed_adj(new_edge_index.long(), data.y.size(-1),
            #     #                                                          data.x.dtype, data.edge_weight)
            #     #     edge_index2 = edge_index2.to(device)
            #     #     edge_weights2 = edge_weights2.to(device)
            #     #     SparseEdges = (edge_index1, edge_index2)
            #     #     edge_weight = (edge_weights1, edge_weights2)
            #     #     del edge_index2, edge_weights2
            #     # else:
            #     #     SparseEdges = edge_index1
            #     #     edge_weight = edge_weights1
            #     # print("edge_weight", edge_weight.shape, data.y.shape)
            #     # del edge_index1, edge_weights1
            #
            #     # out = model(data_x, SparseEdges[:, train_edge_mask], edge_weight)
            #     out = model(data_x, SparseEdges, edge_weight)
            #     val_loss = F.cross_entropy(out[data_val_mask], data_y[data_val_mask])
            opt.step()
            # scheduler.step(val_loss)

            # accs, baccs, f1s = test()
            # train_acc, val_acc, tmp_test_acc = accs
            # train_f1, val_f1, tmp_test_f1 = f1s
            # val_acc_f1 = (val_acc + val_f1) / 2.
            # if val_acc_f1 > best_val_acc_f1:  # update the results
            #     best_val_acc_f1 = val_acc_f1
            #     test_acc = accs[2]
            #     test_bacc = baccs[2]
            #     test_f1 = f1s[2]

            # data_x = GraphSHA_Aug(data.x)
            # print(data.x[0][:100], data.x.shape, SparseEdges.shape, edge_weight.shape)
            # tensor([0., 0., 0.,  ..., 0., 0., 0.]) torch.Size([183, 1703]) torch.Size([2, 737]) torch.Size([737])
            out = model(data.x, SparseEdges, edge_weight)
            # print("out", out.shape, out)

            train_loss = F.nll_loss(out[data_train_mask], data_y[data_train_mask])
            pred_label = out.max(dim=1)[1]

            train_acc = acc(pred_label, data_y, data_train_mask)

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            outstrtrain = 'Train loss:, %.6f, acc:, %.3f,' % (train_loss.detach().item(), train_acc)
            # scheduler.step()

            ####################
            # Validation
            ####################
            model.eval()
            out = model(data.x, SparseEdges, edge_weight)
            pred_label = out.max(dim = 1)[1]            

            test_loss = F.nll_loss(out[data_val_mask], data_y[data_val_mask])
            # print("data_val_mask is not 0: ", data_val_mask.sum())
            test_acc = acc(pred_label, data_y, data_val_mask)

            outstrval = ' Test loss:, %.6f, acc: ,%.3f,' % (test_loss.detach().item(), test_acc)

            duration = "---, %.4f, seconds ---" % (time.time() - start_time)
            log_str = ("%d, / ,%d, epoch," % (epoch, args.epochs)) + outstrtrain + outstrval + duration
            log_str_full += log_str + '\n'
            # print(log_str)

            ####################
            # Save weights
            ####################
            save_perform = test_loss.detach().item()
            if save_perform <= best_test_err:
                early_stopping = 0
                best_test_err = save_perform
                torch.save(model.state_dict(), log_path + '/model' + str(split) + '.t7')
            else:
                early_stopping += 1
            if early_stopping > 500 or epoch == (args.epochs - 1):
                torch.save(model.state_dict(), log_path + '/model_latest' + str(split) + '.t7')
                break

        write_log(vars(args), log_path)

        ####################
        # Testing
        ####################
        model.load_state_dict(torch.load(log_path + '/model' + str(split) + '.t7'))
        model.eval()
        preds = model(data_x, SparseEdges, edge_weight)
        pred_label = preds.max(dim=1)[1]

        np.save(log_path + '/pred' + str(split), pred_label.to('cpu'))

        # acc_train = acc(pred_label, data_y, data_val_mask[:, split])
        # acc_test = acc(pred_label, data_y, data_test_mask[:, split])
        acc_train = acc(pred_label, data_y, data_val_mask)
        acc_test = acc(pred_label, data_y, data_test_mask)

        try:
            model.load_state_dict(torch.load(log_path + '/model_latest' + str(split) + '.t7'))
        except:
            print("lack the latest weights: ", epoch, split,log_path)
            model.load_state_dict(torch.load(log_path + '/model' + str(split) + '.t7'))

        model.eval()
        preds = model(data_x, SparseEdges, edge_weight)
        pred_label = preds.max(dim=1)[1]

        np.save(log_path + '/pred_latest' + str(split), pred_label.to('cpu'))

        acc_train_latest = acc(pred_label, data_y, data_val_mask)
        acc_test_latest = acc(pred_label, data_y, data_test_mask)

        ####################
        # Save testing results
        ####################
        logstr = 'val_acc: ' + str(np.round(acc_train, 3)) + ' test_acc: ' + str(
            np.round(acc_test, 3)) + ' val_acc_latest: ' + str(
            np.round(acc_train_latest, 3)) + ' test_acc_latest: ' + str(np.round(acc_test_latest, 3))
        print(logstr)
        results[split] = [acc_train, acc_test, acc_train_latest, acc_test_latest]
        log_str_full += logstr
        with open(log_path + '/log' + str(split) + '.csv', 'w') as file:
            file.write(log_str_full)
            file.write('\n')
        torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    args = parse_args()
    print(args)
    if args.debug:
        args.epochs = 1
    if args.dataset[:3] == 'syn':
        if args.dataset[4:7] == 'syn':
            if args.p_q not in [-0.08, -0.05]:
                args.dataset = 'syn/syn' + str(int(100 * args.p_q)) + 'Seed' + str(args.seed)
            elif args.p_q == -0.08:
                args.p_inter = -args.p_q
                args.dataset = 'syn/syn2Seed' + str(args.seed)
            elif args.p_q == -0.05:
                args.p_inter = -args.p_q
                args.dataset = 'syn/syn3Seed' + str(args.seed)
        elif args.dataset[4:10] == 'cyclic':
            args.dataset = 'syn/cyclic' + str(int(100 * args.p_q)) + 'Seed' + str(args.seed)
        else:
            args.dataset = 'syn/fill' + str(int(100 * args.p_q)) + 'Seed' + str(args.seed)
    dir_name = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../result_arrays', args.log_path, args.dataset + '/')
    args.log_path = os.path.join(args.log_path, args.method_name, args.dataset)
    if not args.new_setting:
        if args.dataset[:3] == 'syn':
            if args.dataset[4:7] == 'syn':
                setting_dict = pk.load(open('syn_settings.pk', 'rb'))
                dataset_name_dict = {
                    0.95: 1, 0.9: 4, 0.85: 5, 0.8: 6, 0.75: 7, 0.7: 8, 0.65: 9, 0.6: 10
                }
                if args.p_inter == 0.1:
                    dataset = 'syn/syn' + str(dataset_name_dict[args.p_q])
                elif args.p_inter == 0.08:
                    dataset = 'syn/syn2'
                elif args.p_inter == 0.05:
                    dataset = 'syn/syn3'
                else:
                    raise ValueError('Please input the correct p_q and p_inter values!')
            elif args.dataset[4:10] == 'cyclic':
                setting_dict = pk.load(open('./Cyclic_setting_dict.pk', 'rb'))
                dataset_name_dict = {
                    0.95: 0, 0.9: 1, 0.85: 2, 0.8: 3, 0.75: 4, 0.7: 5, 0.65: 6
                }
                dataset = 'syn/syn_tri_' + str(dataset_name_dict[args.p_q])
            else:
                setting_dict = pk.load(open('./Cyclic_fill_setting_dict.pk', 'rb'))
                dataset_name_dict = {
                    0.95: 0, 0.9: 1, 0.85: 2, 0.8: 3
                }
                dataset = 'syn/syn_tri_' + str(dataset_name_dict[args.p_q]) + '_fill'
            setting_dict_curr = setting_dict[dataset][args.method_name].split(',')
            args.alpha = float(setting_dict_curr[setting_dict_curr.index('alpha') + 1])
            try:
                args.num_filter = int(setting_dict_curr[setting_dict_curr.index('num_filter') + 1])
            except ValueError:
                try:
                    args.num_filter = int(setting_dict_curr[setting_dict_curr.index('num_filters') + 1])
                except ValueError:
                    pass
            args.lr = float(setting_dict_curr[setting_dict_curr.index('lr') + 1])
            try:
                args.layer = int(setting_dict_curr[setting_dict_curr.index('layer') + 1])
            except ValueError:
                pass
    if os.path.isdir(dir_name) is False:
        try:
            os.makedirs(dir_name)
        except FileExistsError:
            print('Folder exists!')
    save_name = args.method_name + 'lr' + str(int(args.lr * 1000)) + 'num_filters' + str(
        int(args.num_filter)) + 'alpha' + str(int(100 * args.alpha)) + 'layer' + str(int(args.layer))
    args.save_name = save_name
    results = main(args)
    np.save(dir_name + save_name, results)