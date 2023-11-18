
# internal files
from gens_GraphSHA import *
from parser import *
from main import *
from layer.cheb import *
from utils.Citation import *


if __name__ == "__main__":
    args = parse_args()
    print(args)
    if args.debug:
        args.epochs = 1

    # syn/cyclic dataset only
    if args.dataset[:3] == 'syn':
        args.dataset = syn_dataset(args)

    # directory
    dir_name = os.path.join(os.path.dirname(os.path.realpath(
            __file__)), '../result_arrays',args.log_path,args.dataset+'/')
    args.log_path = os.path.join(args.log_path,args.method_name, args.dataset)

    if os.path.isdir(dir_name) is False:
        try:
            os.makedirs(dir_name)
        except FileExistsError:
            print('Folder exists!')
    save_name = args.method_name + 'lr' + str(int(args.lr*1000)) + 'num_filters' + str(int(args.num_filter)) + 'tud' + str(args.to_undirected) + 'alpha' + str(int(100*args.alpha)) + 'layer' + str(int(args.layer))
    args.save_name = save_name
    results = main(args)
    np.save(dir_name+save_name, results)