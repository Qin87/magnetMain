# external files
import os

import openpyxl
import pandas as pd
import pickle as pk
from datetime import datetime
import time

from openpyxl.reader.excel import load_workbook
from sklearn.metrics import balanced_accuracy_score, f1_score
import warnings
import tqdm

from layer.DGCN import SymModel
from layer.DiGCN import DiModel, DiGCN_IB
from nets_graphSHA.gat import create_gat
from nets_graphSHA.sage import create_sage
from utils.preprocess import F_in_out

warnings.filterwarnings("ignore")

# internal files
from gens_GraphSHA import sampling_idx_individual_dst, sampling_node_source, neighbor_sampling, \
    neighbor_sampling_BiEdge, neighbor_sampling_BiEdge_bidegree, neighbor_sampling_bidegreeOrigin, \
    neighbor_sampling_bidegree_variant1, neighbor_sampling_bidegree_variant2
# from layer.DiGCN import *
from ArgsBen import parse_args
from utils.data_utils import make_longtailed_data_remove, get_idx_info, CrossEntropy, keep_all_data, \
    generate_masksRatio, get_step_split
from gens_GraphSHA import neighbor_sampling_bidegree, saliency_mixup, duplicate_neighbor, test_directed
from neighbor_dist import get_PPR_adj, get_heat_adj, get_ins_neighbor_dist
from nets_graphSHA.gcn import create_gcn
from utils.data_utils import get_dataset, load_directedData
from utils.Citation import *
from layer.geometric_baselines import *
from torch_geometric.utils import to_undirected
from utils.edge_data import get_appr_directed_adj, get_second_directed_adj


def main(args):
    # ********************* write to excel
    date_time = datetime.now().strftime('%m-%d-%H:%M')
    print(date_time)
    # ********************

    if args.IsDirectedData:
        dataset = load_directedData(args)
    else:
        path = args.data_path
        path = osp.join(path, args.undirect_dataset)
        dataset = get_dataset(args.undirect_dataset, path, split_type='full')
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Dataset is ", dataset, "\nChosen from DirectedData: ", args.IsDirectedData)

    data = dataset[0]
    try:
        data = data.to(device)
    except:     # dgl cuda version hard to install
        # device = torch.device("cpu")
        data = data.to(torch.device("cpu"))

    global class_num_list, idx_info, prev_out, sample_times
    global data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin  # data split: train, validation, test
    try:
        data.edge_weight = torch.FloatTensor(data.edge_weight)
    except:
        data.edge_weight = None

    # copy GraphSHA
    if args.Direct_dataset.split('/')[0].startswith('dgl'):
        edges = torch.cat((data.edges()[0].unsqueeze(0), data.edges()[1].unsqueeze(0)), dim=0)
        data_y = data.ndata['label']
        data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin = (
            data.ndata['train_mask'].clone(), data.ndata['val_mask'].clone(), data.ndata['test_mask'].clone())
        data_x = data.ndata['feat']
        dataset_num_features = data_x.shape[1]
    elif not args.IsDirectedData and args.undirect_dataset in ['Coauthor-CS', 'Amazon-Computers', 'Amazon-Photo']:
        edges = data.edge_index  # for torch_geometric librar
        data_y = data.y
        data_x = data.x
        dataset_num_features = dataset.num_features

        data_y = data_y.long()
        n_cls = (data_y.max() - data_y.min() + 1).cpu().numpy()
        n_cls = torch.tensor(n_cls).to(device)

        train_idx, valid_idx, test_idx, train_node = get_step_split(imb_ratio=args.imb_ratio,
                                                                    valid_each=int(data.x.shape[0] * 0.1 / n_cls),
                                                                    labeling_ratio=0.1,
                                                                    all_idx=[i for i in range(data.x.shape[0])],
                                                                    all_label=data.y.cpu().detach().numpy(),
                                                                    nclass=n_cls)

        data_train_maskOrigin = torch.zeros(data.x.shape[0]).bool().to(device)
        data_val_maskOrigin = torch.zeros(data.x.shape[0]).bool().to(device)
        data_test_maskOrigin = torch.zeros(data.x.shape[0]).bool().to(device)
        data_train_maskOrigin[train_idx] = True
        data_val_maskOrigin[valid_idx] = True
        data_test_maskOrigin[test_idx] = True
        train_idx = data_train_maskOrigin.nonzero().squeeze()
        train_edge_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)

        class_num_list = [len(item) for item in train_node]
        idx_info = [torch.tensor(item) for item in train_node]
    elif dataset == 'Amazon-Photo':
        pass
    else:
        edges = data.edge_index  # for torch_geometric librar
        data_y = data.y
        data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin = (data.train_mask.clone(), data.val_mask.clone(),data.test_mask.clone())
        data_x = data.x
        try:
            dataset_num_features = dataset.num_features
        except:
            dataset_num_features = data_x.shape[1]

    IsDirectedGraph = test_directed(edges)
    print("This is directed graph: ", IsDirectedGraph)
    print("data_x", data_x.shape)  # [11701, 300])

    data_y = data_y.long()
    n_cls = (data_y.max() - data_y.min() + 1).cpu().numpy()
    n_cls = torch.tensor(n_cls).to(device)
    print("Number of classes: ", n_cls)

    criterion = CrossEntropy().to(device)

    try:
        splits = data_train_maskOrigin.shape[1]
        print("splits", splits)
        if len(data_test_maskOrigin.shape) == 1:
            data_test_maskOrigin = data_test_maskOrigin.unsqueeze(1).repeat(1, splits)
    except IndexError:
        splits = 1

    if args.method_name == 'APPNP' or 'DiG':
        edge_index1, edge_weights1 = get_appr_directed_adj(args.alpha, edges.long(), data_y.size(-1), data_x.dtype)
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
        del edge_index1, edge_weights1


    if args.method_name == 'GAT':
        model = create_gat(nfeat=dataset_num_features, nhid=args.feat_dim, nclass=n_cls, dropout=0.5,
                           nlayer=args.n_layer)  # SHA
    elif args.method_name == 'GCN':
        model = create_gcn(nfeat=dataset_num_features, nhid=args.feat_dim, nclass=n_cls, dropout=0.5,
                           nlayer=args.n_layer)  # SHA
    elif args.method_name == 'SAGE':
        model = create_sage(nfeat=dataset_num_features, nhid=args.feat_dim, nclass=n_cls, dropout=0.5,
                            nlayer=args.n_layer)
    elif args.method_name == 'GIN':
        model = GIN_ModelBen(data_x.size(-1), n_cls, filter_num=args.num_filter,
                             dropout=args.dropout, layer=args.layer).to(device)
    elif args.method_name == 'Cheb':
        model = ChebModelBen(data_x.size(-1), n_cls, K=args.K,
                             filter_num=args.num_filter, dropout=args.dropout,
                             layer=args.layer).to(device)
    elif args.method_name == 'APPNP':
        model = APPNP_ModelBen(data_x.size(-1), n_cls,
                               filter_num=args.num_filter, alpha=args.alpha,
                               dropout=args.dropout, layer=args.layer).to(device)
    elif args.method_name == 'DiG':
        if not args.method_name[-2:] == 'ib':
            model = DiModel(data_x.size(-1), n_cls, filter_num=args.num_filter,
                            dropout=args.dropout, layer=args.layer).to(device)
        else:
            model = DiGCN_IB(data_x.size(-1), hidden=args.num_filter,
                             n_cls=n_cls, dropout=args.dropout,
                             layer=args.layer).to(device)

    elif args.method_name == 'SymDiGCN':
        model = SymModel(data_x.size(-1), n_cls, filter_num=args.num_filter,
                         dropout=args.dropout, layer=args.layer).to(device)
    else:
        raise NotImplementedError
    try:
        print(model)  # # StandGCN2((conv1): GCNConv(3703, 64)  (conv2): GCNConv(64, 6))
    except:
        pass
    model.to(device)

    for split in range(splits):
        # if split >4:
        #     continue
        print("Beginning for split: ", split, datetime.now().strftime('%d-%H:%M:%S'))
        if splits == 1:
            data_train_mask, data_val_mask, data_test_mask = (data_train_maskOrigin.clone(),
                                                              data_val_maskOrigin.clone(),
                                                              data_test_maskOrigin.clone())
        else:
            try:
                data_train_mask, data_val_mask, data_test_mask = (data_train_maskOrigin[:, split].clone(),
                                                              data_val_maskOrigin[:, split].clone(),
                                                              data_test_maskOrigin[:, split].clone())
            except IndexError:
                print("testIndex ,", data_test_mask.shape, data_train_mask.shape, data_val_mask.shape)
                data_train_mask, data_val_mask = (data_train_maskOrigin[:, split].clone(),data_val_maskOrigin[:, split].clone())
                data_test_mask = data_test_maskOrigin[:, 1].clone()

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

        train_edge_mask = train_edge_mask.cpu()     # Ben for GPU
        if args.gdc == 'ppr':
            neighbor_dist_list = get_PPR_adj(data_x, edges[:, train_edge_mask], alpha=0.05, k=128, eps=None)
        elif args.gdc == 'hk':
            neighbor_dist_list = get_heat_adj(data_x, edges[:, train_edge_mask], t=5.0, k=None, eps=0.0001)
        elif args.gdc == 'none':
            neighbor_dist_list = get_ins_neighbor_dist(data_y.size(0), edges[:, train_edge_mask], data_train_mask,
                                                       device)

        #     #################################
        #     # Train/Validation/Test
        #     #################################
        test_accSHA = test_bacc = test_f1 = 0.0

        # from GraphSHA
        best_val_acc_f1 = 0
        saliency, prev_out = None, None

        CountNotImproved = 0
        for epoch in tqdm.tqdm(range(args.epoch)):
            if CountNotImproved > 50:
                opt = torch.optim.Adam(model.parameters(), lr=10 * args.lr, weight_decay=args.l2)  # less accuracy*
            else:
                opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)  # less accuracy
            # opt = torch.optim.Adam(
            #     [dict(params=model.reg_params, weight_decay=5e-4), dict(params=model.non_reg_params, weight_decay=0), ],
            #     lr=args.lr)  # from SHA
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=100,
                                                                   verbose=False)
            model.train()
            model.to(device)
            opt.zero_grad()  # clear the gradients of the model's parameters.

            if args.AugDirect == 0:
                if args.method_name == 'SymDiGCN':
                    data.edge_index, edge_in, in_weight, edge_out, out_weight = F_in_out(edges,
                                                                                         data_y.size(-1),
                                                                                         data.edge_weight)
                    try:
                        out = model(data_x, edges, edge_in, in_weight, edge_out, out_weight)
                    except:
                        model.to('cpu')
                        data_x = data_x.to('cpu')
                        edges = edges.to('cpu')
                        edge_in = edge_in.to('cpu')
                        edge_out = edge_out.to('cpu')
                        in_weight = in_weight.to('cpu')
                        out_weight = out_weight.to('cpu')
                        out = model(data_x, edges, edge_in, in_weight, edge_out, out_weight)
                        model.to(device)
                        data_x = data_x.to(device)
                        edges = edges.to(device)
                        edge_in = edge_in.to(device)
                        edge_out = edge_out.to(device)
                        in_weight = in_weight.to(device)
                        out_weight = out_weight.to(device)

                elif args.method_name == 'DiG':
                    out = model(data_x, SparseEdges, edge_weight)
                else:
                    out = model(data_x, edges)
                criterion(out[data_train_mask], data_y[data_train_mask]).backward()
            else: # with Aug
                if epoch > args.warmup:
                    prev_out_local = prev_out[train_idx]
                    sampling_src_idx, sampling_dst_idx = sampling_node_source(class_num_list, prev_out_local,
                                                                              idx_info_local, train_idx, args.tau,
                                                                              args.max, args.no_mask)
                    if args.AugDirect == 1:
                        new_edge_index = neighbor_sampling(data_x.size(0), edges[:, train_edge_mask], sampling_src_idx,
                                                           neighbor_dist_list)
                    elif args.AugDirect == 2:
                        new_edge_index = neighbor_sampling_BiEdge(data_x.size(0), edges[:, train_edge_mask],
                                                                  sampling_src_idx, neighbor_dist_list)
                    elif args.AugDirect == 4:
                        new_edge_index = neighbor_sampling_BiEdge_bidegree(data_x.size(0), edges[:, train_edge_mask],
                                                                           sampling_src_idx, neighbor_dist_list)
                    elif args.AugDirect == 20:
                        new_edge_index = neighbor_sampling_bidegree(data_x.size(0), edges[:, train_edge_mask],
                                                                    sampling_src_idx,
                                                                    neighbor_dist_list)  # has two types
                    elif args.AugDirect == 21:
                        new_edge_index = neighbor_sampling_bidegreeOrigin(data_x.size(0), edges[:, train_edge_mask],
                                                                          sampling_src_idx, neighbor_dist_list)
                    elif args.AugDirect == 22:
                        new_edge_index = neighbor_sampling_bidegree_variant1(data_x.size(0), edges[:, train_edge_mask],
                                                                          sampling_src_idx, neighbor_dist_list)
                    elif args.AugDirect == 23:
                        new_edge_index = neighbor_sampling_bidegree_variant2(data_x.size(0), edges[:, train_edge_mask],
                                                                          sampling_src_idx, neighbor_dist_list)

                    else:
                        pass
                    beta = torch.distributions.beta.Beta(1, 100)
                    lam = beta.sample((len(sampling_src_idx),)).unsqueeze(1)
                    new_x = saliency_mixup(data_x, sampling_src_idx, sampling_dst_idx, lam)

                    add_num = new_x.shape[0] - data_x.shape[0]  # Ben
                    new_train_mask = torch.ones(add_num, dtype=torch.bool, device=data_x.device)
                    new_train_mask = new_train_mask.to(data_train_mask.device)  # Ben for GPU
                    # new_train_mask = torch.cat((data_train_mask, new_train_mask), dim=0)  # add some train nodes
                    new_train_mask = torch.cat((data_train_mask, new_train_mask), dim=0).cpu()  # for GPU
                    _new_y = data_y[sampling_src_idx.long()].clone()
                    new_y = torch.cat((data_y, _new_y), dim=0)  #

                    # get edge_weight
                    edge_index1, edge_weights1 = get_appr_directed_adj(args.alpha, new_edge_index.long(),
                                                                       new_y.size(-1), new_x.dtype)
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
                    sampling_src_idx = sampling_src_idx.cpu()  # Ben for GPU
                    sampling_dst_idx = sampling_dst_idx.cpu()
                    beta = torch.distributions.beta.Beta(2, 2)
                    lam = beta.sample((len(sampling_src_idx),)).unsqueeze(1)
                    new_edge_index = duplicate_neighbor(data_x.size(0), edges[:, train_edge_mask], sampling_src_idx)
                    new_x = saliency_mixup(data_x, sampling_src_idx, sampling_dst_idx, lam)

                    add_num = len(sampling_src_idx)  # Ben
                    new_train_mask = torch.ones(add_num, dtype=torch.bool, device=data_x.device)
                    new_train_mask = new_train_mask.to(data_train_mask.device)      # Ben for GPU
                    new_train_mask = torch.cat((data_train_mask, new_train_mask), dim=0)  # add some train nodes
                    _new_y = data_y[sampling_src_idx].clone()
                    new_y = torch.cat((data_y, _new_y), dim=0)  #
                    edge_index1, edge_weights1 = get_appr_directed_adj(args.alpha, new_edge_index.long(),
                                                                       new_y.size(-1), new_x.dtype)
                    edge_index1 = edge_index1.to(device)
                    edge_weights1 = edge_weights1.to(device)
                    if args.method_name[-2:] == 'ib':
                        edge_index2, edge_weights2 = get_second_directed_adj(new_edge_index.long(), new_y.size(-1),new_x.dtype)
                        edge_index2 = edge_index2.to(device)
                        edge_weights2 = edge_weights2.to(device)
                        new_SparseEdges = (edge_index1, edge_index2)
                        new_edge_weight = (edge_weights1, edge_weights2)
                        del edge_index2, edge_weights2
                    else:
                        new_SparseEdges = edge_index1
                        new_edge_weight = edge_weights1
                    del edge_index1, edge_weights1
                if args.method_name == 'SymDiGCN':
                    data.edge_index, edge_in, in_weight, edge_out, out_weight = F_in_out(edges,
                                                                                         data_y.size(-1),
                                                                                         data.edge_weight)
                    out = model(new_x, data.edge_index, edge_in, in_weight, edge_out, out_weight)
                else:
                    try:
                        out = model(new_x, new_SparseEdges, new_edge_weight)  #
                    except RuntimeError:
                        model.to('cpu')
                        new_x = new_x.to('cpu')
                        new_SparseEdges = new_SparseEdges.to('cpu')
                        new_edge_weight = new_edge_weight.to('cpu')
                        out = model(new_x, new_SparseEdges, new_edge_weight)  #
                        model.to(device)
                        new_x = new_x.to(device)
                        new_SparseEdges = new_SparseEdges.to(device)
                        new_edge_weight = new_edge_weight.to(device)
                    except TypeError:
                        try:
                            out = model(new_x, new_edge_index)
                        except TypeError:
                            out= model(new_x)
                try:
                    prev_out = (out[:data_x.size(0)]).clone().to(device)
                except:
                    data_x = data_x.cpu()
                    prev_out = (out[:data_x.size(0)]).clone().to(device)
                _new_y = data_y[sampling_src_idx.long()].clone()    # AttributeError: 'tuple' object has no attribute 'detach'
                new_y = torch.cat((data_y[data_train_mask], _new_y), dim=0)
                out = out.to(device)
                new_y = new_y.to(out.device)
                new_train_mask = new_train_mask.to(out.device)  # Ben for GPU
                print(out.device)
                criterion(out[new_train_mask], new_y).backward()


            torch.cuda.empty_cache()

            with torch.no_grad():
                model.eval()
                if args.method_name == 'SymDiGCN':
                    out = model(data_x, edges[:, train_edge_mask], edge_in, in_weight, edge_out, out_weight)
                elif args.method_name == 'DiG':
                    out = model(data_x, SparseEdges, edge_weight)
                else:
                    out = model(data_x, edges[:, train_edge_mask])

                val_loss = F.cross_entropy(out[data_val_mask], data_y[data_val_mask])
            opt.step()
            scheduler.step(val_loss)
            # from graphSHA
            model.eval()
            if args.method_name == 'SymDiGCN':
                logits = model(data_x, edges[:, train_edge_mask], edge_in, in_weight, edge_out, out_weight)
            elif args.method_name == 'DiG':
                logits = model(data_x, SparseEdges, edge_weight)
            else:
                logits = model(data_x, edges[:, train_edge_mask])

            accs, baccs, f1s = [], [], []
            for mask in [data_train_mask, data_val_mask, data_test_mask]:
                # print(mask.shape, logits.shape)  # torch.Size([3327, 1]) torch.Size([3327, 6])
                pred = logits[mask].max(1)[1]
                y_pred = pred.cpu().numpy()
                y_true = data_y[mask].cpu().numpy()
                acc_epoch = pred.eq(data_y[mask]).sum().item() / mask.sum().item()
                bacc = balanced_accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average='macro')
                accs.append(acc_epoch)
                baccs.append(bacc)
                f1s.append(f1)
            train_accSHA, val_accSHA, tmp_test_acc = accs
            train_f1, val_f1, tmp_test_f1 = f1s
            val_acc_f1 = (val_accSHA + val_f1) / 2.
            if val_acc_f1 > best_val_acc_f1:
                best_val_acc_f1 = val_acc_f1
                test_accSHA = accs[2]
                test_bacc = baccs[2]
                test_f1 = f1s[2]
            else:
                CountNotImproved += 1

        if args.IsDirectedData:
            dataset_to_print = args.Direct_dataset
        else:
            dataset_to_print = args.undirect_dataset

        print(args.method_name, dataset_to_print, "imb_ratio", args.imb_ratio, "Aug", str(args.AugDirect), "epoch", args.epoch)
        print('split: {:3d}, val_loss: {:6.2f}, test_Acc: {:6.2f}, test_bacc: {:6.2f}, test_f1: {:6.2f}'.format(split,val_loss, test_accSHA * 100, test_bacc * 100,test_f1 * 100))



if __name__ == "__main__":
    start_sum_time = time.time()
    args = parse_args()
    print(args)

    cuda_device = args.GPUdevice
    if torch.cuda.is_available():
        print("CUDA Device Index:", cuda_device)
        device = torch.device("cuda:%d" % cuda_device)
    else:
        print("CUDA is not available, using CPU.")
        device = torch.device("cpu")

    main(args)
    end_sum_time = time.time()
    total_time = end_sum_time- start_sum_time
    print("Total time(all splits): ", total_time)
