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

# internal files
from layer.cheb import *
from src.ArgsBen import parse_args
from utils.Citation import *
from layer.geometric_baselines import GATModel, GCNModel, GIN_Model, SAGEModel, ChebModel, APPNP_Model
from torch_geometric.utils import to_undirected
from utils.preprocess import geometric_dataset, load_syn
from utils.save_settings import write_log
from utils.hermitian import hermitian_decomp
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
    if os.path.isdir(log_path) == False:
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

    if os.path.isdir(log_path) == False:
        os.makedirs(log_path)

    data = dataset[0]
    if not data.__contains__('edge_weight'):
        data.edge_weight = None
    if args.to_undirected:
        data.edge_index = to_undirected(data.edge_index)

    data.y = data.y.long()
    num_classes = (data.y.max() - data.y.min() + 1).detach().numpy()
    data = data.to(device)
    # normalize label, the minimum should be 0 as class index
    splits = data.train_mask.shape[1]
    if len(data.test_mask.shape) == 1:
        data.test_mask = data.test_mask.unsqueeze(1).repeat(1, splits)

    results = np.zeros((splits, 4))
    for split in range(splits):
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
            elif TODO
            elif
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
    results = main(args)
    np.save(dir_name + save_name, results)
