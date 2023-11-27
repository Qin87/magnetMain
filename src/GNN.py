# external files
import numpy as np
import pickle as pk
import torch.optim as optim
from datetime import datetime
import os, time, argparse, csv
from collections import Counter
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.datasets import WebKB, WikipediaNetwork, WikiCS
import tqdm
import warnings

from src.layer.DGCN import SymModel
from src.layer.DiGCN import DiModel, DiGCN_IB

warnings.filterwarnings("ignore")

# internal files
from gens_GraphSHA import sampling_idx_individual_dst, sampling_node_source, neighbor_sampling, neighbor_sampling_BiEdge
# from layer.DiGCN import *
from nets_graphSHA import *
from layer.cheb import *
from src.ArgsBen import parse_args
from src.data_utils import make_longtailed_data_remove, get_idx_info, CrossEntropy, generate_masks, keep_all_data, \
    generate_masksRatio
from src.gens_GraphSHA import neighbor_sampling_bidegree, saliency_mixup, duplicate_neighbor, test_directed
from src.neighbor_dist import get_PPR_adj, get_heat_adj, get_ins_neighbor_dist
from src.nets_graphSHA.gcn import create_gcn
from src.utils.data_utils_graphSHA import get_dataset, load_directedData
from utils.Citation import *
from layer.geometric_baselines import *
from torch_geometric.utils import to_undirected
from utils.preprocess import geometric_dataset, load_syn
from utils.save_settings import write_log
from utils.hermitian import hermitian_decomp
from utils.edge_data import get_appr_directed_adj, get_second_directed_adj
from utils.symmetric_distochastic import desymmetric_stochastic

# select cuda device if available
cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")


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

    if args.IsDirectedData:
        dataset = load_directedData(args)
    else:
        path = args.data_path
        path = osp.join(path, args.undirect_dataset)
        dataset = get_dataset(args.undirect_dataset, path, split_type='full')
    print("Dataset is ", dataset, "\nIs DirectedData: ", args.IsDirectedData)

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
    if args.to_undirected:
        data.edge_index = to_undirected(data.edge_index)

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

    if args.IsDirectedData:
        splits = data.train_mask.shape[1]
        print("splits", splits)
        if len(data.test_mask.shape) == 1:
            data.test_mask = data.test_mask.unsqueeze(1).repeat(1, splits)
    else:
        splits = 1
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
        if splits == 1:
            data_train_mask, data_val_mask, data_test_mask = (data.train_mask.clone(),
                                                              data.val_mask.clone(),
                                                              data.test_mask.clone())
        else:
            data_train_mask, data_val_mask, data_test_mask = (data.train_mask[:, split].clone(),
                                                          data.val_mask[:, split].clone(),data.test_mask[:,split].clone())

        if args.CustomizeMask:
            data_train_mask, data_val_mask, data_test_mask = generate_masksRatio(data_y, TrainRatio=0.3, ValRatio=0.3)

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



	log_str_full = ''
        if args.method_name == 'GAT':
            model = GATModel(data.x.size(-1), num_classes, heads=args.heads, filter_num=args.num_filter,
                              dropout=args.dropout, layer=args.layer).to(device)
        elif args.method_name == 'GCN':
            model = GCNModel(data.x.size(-1), num_classes, filter_num=args.num_filter,
                             dropout=args.dropout, layer=args.layer).to(device)
        elif args.method_name == 'SAGE':
            model = SAGEModel(data.x.size(-1), num_classes, filter_num=args.num_filter,
                              dropout=args.dropout, layer=args.layer).to(device)
            # model = SAGE_Link(x.size(-1), args.num_class_link, filter_num=args.num_filter, dropout=args.dropout).to(
            #     device)
        elif args.method_name == 'GIN':
            model = GIN_Model(data.x.size(-1), num_classes, filter_num=args.num_filter,
                              dropout=args.dropout, layer=args.layer).to(device)
        elif args.method_name == 'Cheb':
            model = ChebModel(data.x.size(-1), num_classes, K=args.K,
                              filter_num=args.num_filter, dropout=args.dropout,
                              layer=args.layer).to(device)
        elif args.method_name == 'APPNP':
            model = APPNP_Model(data.x.size(-1), num_classes,
                                filter_num=args.num_filter, alpha=args.alpha,
                                dropout=args.dropout, layer=args.layer).to(device)
            # model = APPNP_Link(x.size(-1), args.num_class_link, filter_num=args.num_filter, alpha=args.alpha,
            #                    dropout=args.dropout, K=args.K).to(device)
        elif args.method_name == 'Digraph':
            if not args.method_name[-2:] == 'ib':
                model = DiModel(data.x.size(-1), num_classes, filter_num=args.num_filter,
                                dropout=args.dropout, layer=args.layer).to(device)
            else:
                model = DiGCN_IB(data.x.size(-1), hidden=args.num_filter,
                                 num_classes=num_classes, dropout=args.dropout,
                                 layer=args.layer).to(device)

        elif args.method_name == 'SymDiGCN':
            model = SymModel(data.x.size(-1), num_classes, filter_num=args.num_filter,
                             dropout=args.dropout, layer=args.layer).to(device)
        else:
            raise NotImplementedError


        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        #################################
        # Train/Validation/Test
        #################################
        best_test_err = 1000.0
        early_stopping = 0
        for epoch in range(args.epochs):
            start_time = time.time()
            ####################
            # Train
            ####################
            train_loss, train_acc = 0.0, 0.0

            # for loop for batch loading
            model.train()
            out = model(data)

            train_loss = F.nll_loss(out[data.train_mask[:, split]], data.y[data.train_mask[:, split]])
            pred_label = out.max(dim=1)[1]
            train_acc = acc(pred_label, data.y, data.train_mask[:, split])

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            outstrtrain = 'Train loss:, %.6f, acc:, %.3f,' % (train_loss.detach().item(), train_acc)
            # scheduler.step()

            ####################
            # Validation
            ####################
            model.eval()
            test_loss, test_acc = 0.0, 0.0

            out = model(data)
            pred_label = out.max(dim=1)[1]

            test_loss = F.nll_loss(out[data.val_mask[:, split]], data.y[data.val_mask[:, split]])
            test_acc = acc(pred_label, data.y, data.val_mask[:, split])

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
        preds = model(data)
        pred_label = preds.max(dim=1)[1]

        np.save(log_path + '/pred' + str(split), pred_label.to('cpu'))

        acc_train = acc(pred_label, data.y, data.val_mask[:, split])
        acc_test = acc(pred_label, data.y, data.test_mask[:, split])

        model.load_state_dict(torch.load(log_path + '/model_latest' + str(split) + '.t7'))
        model.eval()
        preds = model(data)
        pred_label = preds.max(dim=1)[1]

        np.save(log_path + '/pred_latest' + str(split), pred_label.to('cpu'))

        acc_train_latest = acc(pred_label, data.y, data.val_mask[:, split])
        acc_test_latest = acc(pred_label, data.y, data.test_mask[:, split])

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
            args.heads = int(setting_dict_curr[setting_dict_curr.index('heads') + 1])
            args.to_undirected = (setting_dict_curr[setting_dict_curr.index('to_undirected') + 1] == 'True')
            try:
                args.num_filter = int(setting_dict_curr[setting_dict_curr.index('num_filter') + 1])
            except ValueError:
                pass
            try:
                args.layer = int(setting_dict_curr[setting_dict_curr.index('layer') + 1])
            except ValueError:
                pass
            args.lr = float(setting_dict_curr[setting_dict_curr.index('lr') + 1])
    if os.path.isdir(dir_name) is False:
        try:
            os.makedirs(dir_name)
        except FileExistsError:
            print('Folder exists!')
    save_name = args.method_name + 'lr' + str(int(args.lr * 1000)) + 'num_filters' + str(
        int(args.num_filter)) + 'tud' + str(args.to_undirected) + 'heads' + str(int(args.heads)) + 'layer' + str(
        int(args.layer))
    args.save_name = save_name
    main(args)
