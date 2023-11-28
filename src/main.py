# external files
import torch.optim as optim
from datetime import datetime

# internal files
from src.utils.data_utils import *
from neighbor_dist import *
from parser import *
from main import *
from utils.Citation import *
from layer.geometric_baselines import *
# from torch_geometric.utils import to_undirected
from utils.edge_data import to_undirectedBen

from utils.save_settings import write_log


def validation(split):
    model.eval()

    out = model(data)
    pred_label = out.max(dim=1)[1]

    test_loss = F.nll_loss(out[data.val_mask[:, split]], data.y[data.val_mask[:, split]])
    test_acc = acc(pred_label, data.y, data.val_mask[:, split])



    return test_loss, test_acc



def test(split, log_path):
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

    return acc_train, acc_test, acc_train_latest, acc_test_latest

def NumDirectedEdge():
    count = 0
    all_edges = set()
    self_loop = 0
    for i in range(data.edge_index.shape[1]):
        if data.edge_index[0][i].item() == data.edge_index[1][i].item():
            self_loop += 1
            continue

        edge = frozenset((data.edge_index[0][i].item(), data.edge_index[1][i].item()))
        inv_edge = frozenset((data.edge_index[1][i].item(), data.edge_index[0][i].item()))
        if edge in all_edges:
            count += 1
        all_edges.add(inv_edge)
    print("Bi-directed edges num: ", count, ", total edge num: ",
          data.edge_index.shape[1], "\nself loop num: ", self_loop,
          "\tundirected graph: ")

def acc(pred, label, mask):
    correct = int(pred[mask].eq(label[mask]).sum().item())
    acc = correct / int(mask.sum())
    return acc

def main(args):
    global data, model, device
    global class_num_list, idx_info, prev_out, sample_times
    global data_train_mask, data_val_mask, data_test_mask  # data split: train, validation, test
    global opt, scheduler, criterion
    # select cuda device if available
    cuda_device = 0
    device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")

    criterion = CrossEntropy().to(device)
    # opt = torch.optim.Adam([dict(params=model.reg_params, weight_decay=5e-4),
    #                               dict(params=model.non_reg_params, weight_decay=0), ], lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=100,
    #                                                        verbose=False)

    # print(args)
    if args.randomseed > 0:
        torch.manual_seed(args.randomseed)

    date_time = datetime.now().strftime('%m-%d-%H:%M:%S')
    log_path = os.path.join(args.log_root, args.log_path, args.save_name, date_time)
    if os.path.isdir(log_path) is False:
        try:
            os.makedirs(log_path)
        except FileExistsError:
            print('Folder exists!')
    print("Save Logs:", log_path)
    dataset = load_data(args)
    data = dataset[0]

    if os.path.isdir(log_path) is False:
        os.makedirs(log_path)
    if not data.__contains__('edge_weight'):
        data.edge_weight = None


    data.y = data.y.long()
    n_cls = (data.y.max() - data.y.min() + 1).detach().numpy()
    data = data.to(device)

    if data.edge_weight is not None:
        data.edge_weight = torch.FloatTensor(data.edge_weight).to(device)
    if args.to_undirected:
        print("Before Converting, num of edges is ", data.edge_index.shape, data.edge_index.shape[1])
        data.edge_index = to_undirectedBen(data.edge_index)
        print("After Converting, num of edges is ", data.edge_index.shape, data.edge_index.shape)

    splits = data.train_mask.shape[1]
    if len(data.test_mask.shape) == 1:
        data.test_mask = data.test_mask.unsqueeze(1).repeat(1, splits)

    results = np.zeros((splits, 4))
    for split in range(splits):

        data_train_mask, data_val_mask, data_test_mask = data.train_mask[:,split].clone(), data.val_mask[:,split].clone(), data.test_mask[:,split].clone()
        idx_info = get_idx_info(data.y, n_cls, data_train_mask)  # all train nodes for each class
        stats = data.y[data_train_mask]  # this is selected y. only train nodes of y
        n_data = []  # num of train in each class
        for i in range(n_cls):
            data_num = (stats == i).sum()
            n_data.append(int(data_num.item()))

        class_num_list, data_train_mask, idx_info, train_node_mask, train_edge_mask = \
            make_longtailed_data_remove(data.edge_index, data.y, n_data, n_cls, args.imb_ratio, data_train_mask.clone())

        log_str_full = ''
        graphmodel = APPNP_Model(data.x.size(-1), n_cls,
                                 filter_num=args.num_filter, alpha=args.alpha,
                                 dropout=args.dropout, layer=args.layer).to(device)
        model = graphmodel  # nn.DataParallel(graphmodel)
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
            train_loss, train_acc = train(args, split, epoch)
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            outstrtrain = 'Train loss: %.6f, acc: %.3f; ' % (train_loss.detach().item(), train_acc)
            ####################
            # Validation
            ####################
            test_loss, test_acc = validation(split)
            outstrval = ' Test loss: %.6f, acc: %.3f; ' % (test_loss.detach().item(), test_acc)
            duration = "---%.4f seconds---" % (time.time() - start_time)
            log_str = ("%d/%d epoch: " % (epoch, args.epochs)) + outstrtrain + outstrval + duration
            log_str_full += log_str + '\n'

            ####################
            # Save weights
            ####################
            save_perform = test_loss.detach().item()
            if save_perform <= best_test_err:  # get a better model
                early_stopping = 0
                best_test_err = save_perform
                torch.save(model.state_dict(), log_path + '/model' + str(split) + '.t7')
            else:
                early_stopping += 1
            if early_stopping > 500 or epoch == (args.epochs - 1):  # after 500 epochs, get no improvement
                torch.save(model.state_dict(), log_path + '/model_latest' + str(split) + '.t7')
                print(log_str)
                break

        write_log(vars(args), log_path)
        ####################
        # Testing
        ####################
        model.load_state_dict(torch.load(log_path + '/model' + str(split) + '.t7'))
        acc_train, acc_test, acc_train_latest, acc_test_latest = test(split, log_path)

        ####################
        # Save testing results
        ####################
        logstr = "Training" + str(split) + '\tval_acc: ' + str(np.round(acc_train, 3)) + ' test_acc: ' + str(
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

def train(args, split, epoch):
    # for loop for batch loading
    model.train()

    train_idx = data_train_mask.nonzero().squeeze()    # get the index of training data
    train_edge_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)  # 238162
    if min(class_num_list)*2 < max(class_num_list):
        new_x, new_edge_index, train_loss= aug_nodeandedge(args, epoch, train_idx, train_edge_mask)
        print(new_x.shape, new_edge_index.shape)  # torch.Size([426, 1703]) torch.Size([2, 920])
        print(data.x[:, split].shape, data.edge_index[:, split].shape)   # torch.Size([251]) torch.Size([2])
        print(data.x.shape, data.edge_index.shape)   # torch.Size([251, 1703]) torch.Size([2, 515])
        data.x, data.edge_index = new_x, new_edge_index
    out = model(data.x, data.edge_index)

    print(data.train_mask[:, split].shape, data.y.shape, out.shape)   # torch.Size([251]) torch.Size([426, 5])
    # train_loss = F.nll_loss(out[data.train_mask[:, split]], data.y[data.train_mask[:, split]])
    # train_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    pred_label = out.max(dim=1)[1]

    train_acc = acc(pred_label, data.y, data.train_mask[:, split])

    return train_loss, train_acc

def get_neigh_dist_list(args, train_edge_mask):
    if args.gdc == 'ppr':
        neighbor_dist_list = get_PPR_adj(data.x, data.edge_index[:, train_edge_mask], alpha=0.05, k=128, eps=None)
    elif args.gdc == 'hk':
        neighbor_dist_list = get_heat_adj(data.x, data.edge_index[:, train_edge_mask], t=5.0, k=None, eps=0.0001)
    elif args.gdc == 'none':
        neighbor_dist_list = get_ins_neighbor_dist(data.y.size(0), data.edge_index[:, train_edge_mask], data_train_mask,
                                                   device)

    return neighbor_dist_list

def aug_nodeandedge(args, epoch, train_idx,train_edge_mask):

    train_idx_list = train_idx.cpu().tolist()
    local2global = {i: train_idx_list[i] for i in range(len(train_idx_list))}
    global2local = dict([val, key] for key, val in local2global.items())
    idx_info_list = [item.cpu().tolist() for item in idx_info]
    idx_info_local = [torch.tensor(list(map(global2local.get, cls_idx))) for cls_idx in idx_info_list]

    neighbor_dist_list = get_neigh_dist_list(args, train_edge_mask)

    if epoch > args.warmup:
        # identifying source samples
        prev_out_local = prev_out[train_idx]
        sampling_src_idx, sampling_dst_idx = sampling_node_source(class_num_list, prev_out_local,
                                                                  idx_info_local, train_idx, args.tau, args.max,
                                                                  args.no_mask)

        new_edge_index = neighbor_sampling(data.x.size(0), data.edge_index[:, train_edge_mask],
                                           sampling_src_idx, neighbor_dist_list)
        beta = torch.distributions.beta.Beta(1, 100)
        lam = beta.sample((len(sampling_src_idx),)).unsqueeze(1)
        new_x = saliency_mixup(data.x, sampling_src_idx, sampling_dst_idx, lam)

    else:
        sampling_src_idx, sampling_dst_idx = sampling_idx_individual_dst(class_num_list, idx_info, device)
        beta = torch.distributions.beta.Beta(2, 2)
        lam = beta.sample((len(sampling_src_idx),)).unsqueeze(1)
        new_edge_index = duplicate_neighbor(data.x.size(0), data.edge_index[:, train_edge_mask],
                                            sampling_src_idx)
        new_x = saliency_mixup(data.x, sampling_src_idx, sampling_dst_idx, lam)

    out = model(data.x, data.edge_index)
    prev_out = (out[:data.x.size(0)]).detach().clone()
    add_num = out.shape[0] - data_train_mask.shape[0]
    new_train_mask = torch.ones(add_num, dtype=torch.bool, device=data.x.device)
    new_train_mask = torch.cat((data_train_mask, new_train_mask), dim=0)  # add some train nodes
    _new_y = data.y[sampling_src_idx].clone()
    new_y = torch.cat((data.y[data_train_mask], _new_y), dim=0)
    train_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

    return new_x, new_edge_index, train_loss


def syn_dataset(args):
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



