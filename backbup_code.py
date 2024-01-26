# def Uni_VarData():
#     # if args.IsDirectedData:
#     #     dataset = load_directedData(args)
#     # else:
#     #     path = args.data_path
#     #     path = osp.join(path, args.undirect_dataset)
#     #     dataset = get_dataset(args.undirect_dataset, path, split_type='full')
#     os.chdir(os.path.dirname(os.path.abspath(__file__)))
#     # print("Dataset is ", dataset, "\nChosen from DirectedData: ", args.IsDirectedData)
#
#     # if os.path.isdir(log_path) is False:
#     #     os.makedirs(log_path)
#
#     data = dataset[0]
#     data = data.to(device)
#
#     global class_num_list, idx_info, prev_out, sample_times
#     global data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin  # data split: train, validation, test
#     try:
#         data.edge_weight = torch.FloatTensor(data.edge_weight)
#     except:
#         data.edge_weight = None
#
#     # if args.to_undirected:
#     #     data.edge_index = to_undirected(data.edge_index)
#
#     # copy GraphSHA
#     if args.undirect_dataset.split('/')[0].startswith('dgl'):
#         edges = torch.cat((data.edges()[0].unsqueeze(0), data.edges()[1].unsqueeze(0)), dim=0)
#         data_y = data.ndata['label']
#         data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin = (
#             data.ndata['train_mask'].clone(), data.ndata['val_mask'].clone(), data.ndata['test_mask'].clone())
#         data_x = data.ndata['feat']
#         dataset_num_features = data_x.shape[1]
#     # elif not args.IsDirectedData and args.undirect_dataset in ['Coauthor-CS', 'Amazon-Computers', 'Amazon-Photo']:
#     elif not args.IsDirectedData and args.undirect_dataset in ['Coauthor-CS', 'Amazon-Computers', 'Amazon-Photo']:
#         edges = data.edge_index  # for torch_geometric librar
#         data_y = data.y
#         data_x = data.x
#         dataset_num_features = dataset.num_features
#
#         data_y = data_y.long()
#         n_cls = (data_y.max() - data_y.min() + 1).cpu().numpy()
#         n_cls = torch.tensor(n_cls).to(device)
#
#         train_idx, valid_idx, test_idx, train_node = get_step_split(imb_ratio=args.imb_ratio,
#                                                                     valid_each=int(data.x.shape[0] * 0.1 / n_cls),
#                                                                     labeling_ratio=0.1,
#                                                                     all_idx=[i for i in range(data.x.shape[0])],
#                                                                     all_label=data.y.cpu().detach().numpy(),
#                                                                     nclass=n_cls)
#
#         data_train_maskOrigin = torch.zeros(data.x.shape[0]).bool().to(device)
#         data_val_maskOrigin = torch.zeros(data.x.shape[0]).bool().to(device)
#         data_test_maskOrigin = torch.zeros(data.x.shape[0]).bool().to(device)
#         data_train_maskOrigin[train_idx] = True
#         data_val_maskOrigin[valid_idx] = True
#         data_test_maskOrigin[test_idx] = True
#         train_idx = data_train_maskOrigin.nonzero().squeeze()
#         train_edge_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)
#
#         class_num_list = [len(item) for item in train_node]
#         idx_info = [torch.tensor(item) for item in train_node]
#     else:
#         edges = data.edge_index  # for torch_geometric librar
#         data_y = data.y
#         data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin = (data.train_mask.clone(), data.val_mask.clone(),data.test_mask.clone())
#         data_x = data.x
#         try:
#             dataset_num_features = dataset.num_features
#         except:
#             dataset_num_features = data_x.shape[1]
#
#     # IsDirectedGraph = test_directed(edges)
#     # print("This is directed graph: ", IsDirectedGraph)
#     # print("data_x", data_x.shape)  # [11701, 300])
#
#     data = data.to(device)
#
#     data_y = data_y.long()
#     # n_cls = (data_y.max() - data_y.min() + 1).cpu().numpy()
#     # n_cls = torch.tensor(n_cls).to(device)
#     # print("Number of classes: ", n_cls)
#
#     return data, data_x, data_y, edges, dataset_num_features,data_train_maskOrigin, data_val_maskOrigin, data_test_maskOrigin
