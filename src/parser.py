import os, time, argparse, csv
from torch_geometric.datasets import WebKB, WikipediaNetwork, WikiCS
from utils.preprocess import geometric_dataset, load_syn
from utils.Citation import *
from gens_GraphSHA import *




def load_data(args):
    load_func, subset = args.dataset.split('/')[0], args.dataset.split('/')[1]
    print("dataset is ", load_func)  # Ben WebKB
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

    return dataset
def parse_args():
    parser = argparse.ArgumentParser(description="baseline--APPNP")

    parser.add_argument('--log_root', type=str, default='../logs/', help='the path saving model.t7 and the training process')
    parser.add_argument('--log_path', type=str, default='test', help='the path saving model.t7 and the training process, the name of folder will be log/(current time)')
    parser.add_argument('--data_path', type=str, default='../dataset/data/tmp/', help='data set folder, for default format see dataset/cora/cora.edges and cora.node_labels')
    parser.add_argument('--dataset', type=str, default='WebKB/Wisconsin', help='data set selection')

    parser.add_argument('--epochs', type=int, default=1500, help='training epochs')
    parser.add_argument('--num_filter', type=int, default=2, help='num of filters')

    parser.add_argument('--p_q', type=float, default=0.95, help='direction strength, from 0.5 to 1.')
    parser.add_argument('--p_inter', type=float, default=0.1, help='inter_cluster edge probabilities.')
    parser.add_argument('--method_name', type=str, default='APPNP', help='method name')
    parser.add_argument('--seed', type=int, default=0, help='random seed for training testing split/random graph generation')

    parser.add_argument('--alpha', type=float, default=0.1, help='alpha teleport prob')
    parser.add_argument('-to_undirected', '-tud', action='store_true', help='if convert graph to undirecteds')
    parser.add_argument('--debug', '-D', action='store_true', help='debug mode')
    parser.add_argument('--new_setting', '-NS', action='store_true', help='whether not to load best settings')

    parser.add_argument('--layer', type=int, default=2, help='number of layers (2 or 3), default: 2')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--l2', type=float, default=5e-4, help='l2 regularizer')
    parser.add_argument('--MakeImbalance', type=bool, default=True, help='True for turn dataset into imbalanced')
    parser.add_argument('--CustomizeMask', type=bool, default=True, help='True for generate train,val,test splits by me')

    parser.add_argument('--dropout', type=float, default=0.0, help='dropout prob')
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

    # copy from GraphSHA
    # parser.add_argument('--seed', type=int, default=100, help='seed')
    # parser.add_argument('--dataset', type=str,
    #                     choices=['Cora', 'CiteSeer', 'PubMed', 'Amazon-Photo', 'Amazon-Computers', 'Coauthor-CS',
    #                              'WiKiCS'],
    #                     default='Cora', help='dataset name')
    # parser.add_argument('--data_path', type=str, default='datasets/', help='data path')
    parser.add_argument('--imb_ratio', type=float, default=100, help='imbalance ratio')
    parser.add_argument('--net', type=str, choices=['GCN', 'GAT', 'SAGE'], default='GCN', help='GNN bachbone')
    parser.add_argument('--n_layer', type=int, default=2, help='the number of layers')
    parser.add_argument('--feat_dim', type=int, default=64, help='feature dimension')
    parser.add_argument('--warmup', type=int, default=5, help='warmup epoch')
    # parser.add_argument('--epoch', type=int, default=900, help='epoch')
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