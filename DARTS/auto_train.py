import subprocess, configparser, argparse, logging, sys, os, time
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='config/')
parser.add_argument('--config', type=str, default='train')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
logpath = os.path.join(args.log, time.strftime("%Y%m%d-%H%M%S")+'-gpu'+args.gpu)
os.makedirs(logpath)
fh = logging.FileHandler(os.path.join(logpath, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

config_parser = configparser.ConfigParser()
gpu = args.gpu

def run(config):
    print('saving at:', logpath)
    save = config.get('other', 'save')
    big_alpha = config.getboolean('other', 'big_alpha')
    dataset = config.get('other', 'dataset')
    epoch = config.get('other', 'epoch')

    adv = config.get('inner', 'adv')
    adv_lambda = config.get('inner', 'adv_lambda')
    acc_lambda = config.get('inner', 'acc_lambda')
    ood_lambda = config.get('inner', 'ood_lambda')

    nop_outer = config.getboolean('outer', 'nop_outer')
    adv_outer = config.getboolean('outer', 'adv_outer')
    ood_outer = config.getboolean('outer', 'ood_outer')
    flp_outer = config.getboolean('outer', 'flp_outer')    
    mgda = config.getboolean('outer', 'mgda')
    grad_norm = config.get('outer', 'grad_norm')
    constrain = config.get('outer', 'constrain')
    constrain_min = config.get('outer', 'constrain_min')
    temperature = config.get('outer', 'temperature')
    fx = config.get('outer', 'fx')
    nop_later = config.get('outer', 'nop_later')
    adv_later = config.get('outer', 'adv_later')

    for keys in config:
        print(keys)
        for k in config[keys]:
            print('\t', k, config[keys][k])
    

    print('saving at:', logpath)
    init_channels = '36'

    logging.info("now running train")
    train_args = ['python', 'train.py', '--cutout', '--auxiliary']
    train_args += ['--gpu', gpu]
    train_args += ['--save', save]
    train_args += ['--arch', save]
    train_args += ['--init_channels', init_channels]
    train_args += ['--dataset', dataset]
    proc = subprocess.check_output(train_args)
    logging.info('clean_acc ' + proc.decode('utf-8').split()[-1])

    print('saving at:', logpath)
    model_path = '{}/channel{}_{}/best_model.pt'.format(save, init_channels, dataset)
    logging.info("now running test_adv")
    # attack = 'FGSM'
    # attack_args = ['python', 'test_adv.py', '--cutout', '--auxiliary']
    # attack_args += ['--gpu', gpu, '--model_path', model_path, '--dataset', dataset]
    # attack_args += ['--attack', attack, '--arch', save, '--init_channels', init_channels]
    # proc = subprocess.check_output(attack_args)
    # logging.info('FGSM_acc ' + proc.decode('utf-8').split()[-1])

    attack = 'PGD'
    attack_args = ['python', 'test_adv.py', '--cutout', '--auxiliary']
    attack_args += ['--gpu', gpu, '--model_path', model_path, '--dataset', dataset]
    attack_args += ['--attack', attack, '--arch', save, '--init_channels', init_channels]
    proc = subprocess.check_output(attack_args)
    logging.info('PGD_acc ' + proc.decode('utf-8').split()[-1])

    ood_args = ['python', 'test_ood.py', '--arch', save]
    proc = subprocess.check_output(ood_args)
    proc_info = proc.decode('utf-8').split()
    logging.info('OOD frp ' + proc_info[-1])

    flp_args = ['python', 'test_flp.py', '--arch', save]
    proc = subprocess.check_output(flp_args)
    proc_info = proc.decode('utf-8').split()
    logging.info('Params in MB ' + proc_info[-2])
    logging.info('FLOPs in MB ' + proc_info[-1])

if args.config != 'none':
    if args.config == 'train':
        dirpath = 'config/train/'
        filenames = os.listdir(dirpath)
        for filename in filenames:
            ini = os.path.join(dirpath, filename)
            print('now using ini:', ini)
            config_parser.read(ini)
            run(config_parser)
    else:   
        config_parser.read(args.config)
        run(config_parser)

else:
    copyfile('auto.py', os.path.join(logpath, 'auto.py'))
    adv = 'fast'
    inner_values = [(0, 1, 1)] # adv_lambda, acc_lambda, ood_lambda
    constrain = 'abs' # min, abs
    constrain_mins = [4, 2] # [2, 3] # [1, 2, 3]
    temperature = 'none' # GumbelA, none, A
    fxs = ['none'] # ['Sqr', 'Cub', 'Exp', 'Tan'] # none, Sqr, Cub, Exp, Tan
    nop_outer = 1
    adv_outer = 1
    ood_outer = 0
    flp_outer = 1
    mgda = 0 # 1
    grad_norms = ['l2'] #['l2', 'loss'] #'loss' # none, l2, loss+
    nop_later = 0 # 30
    adv_later = 0 # 30
    epoch = 50

    big_alpha = 0
    search = 1
    train = 0
    test_adv = 0
    datasets = ['cifar10'] #, 'cifar100']

    for dataset in datasets:
        for adv_lambda, acc_lambda, ood_lambda in inner_values:
            for constrain_min in constrain_mins:
                for fx in fxs:
                    for grad_norm in grad_norms:
                        # if big_alpha:
                        #     config_name = 'bigAlphaInit'
                        # else:
                        #     config_name = 'randomInit'

                        config_name = 'LL'
                        if adv != 'none':
                            if adv_lambda:
                                config_name += '_adv' + str(adv_lambda)
                        if acc_lambda:
                            config_name += '_acc' + str(acc_lambda)
                        if ood_lambda:
                            config_name += '_ood' + str(ood_lambda)

                        config_name += '_UL'
                        
                        if adv_outer:
                            config_name += '_adv'
                        if nop_outer:
                            config_name += '_nop'
                        if ood_outer:
                            config_name += '_ood'
                        if flp_outer:
                            config_name += '_flp'
                        if mgda:
                            config_name += '_mgda'
                        if constrain != 'none':
                            config_name += '_' + constrain + str(int(constrain_min*10))
                        # if temperature != 'none':
                        #     config_name += '_temp' + temperature
                        # if fx != 'none':
                        #     config_name += '_fx' + fx

                        if nop_later != 0:
                            config_name += '_nopLater' + str(nop_later)
                        if adv_later != 0:
                            config_name += '_advLater' + str(adv_later)
                        if epoch != 50:
                            config_name += '_epoch' + str(epoch)
                        if grad_norm != 'none':
                            config_name += '_gn' + grad_norm

                        if dataset != 'cifar10':
                            config_name += '_' + dataset
                        
                        # save_path = os.path.join(logpath, config_name)

                        config_dict = {'other':{'save':config_name, 'search':search, 'train':train, 'test_adv':test_adv, 'big_alpha':big_alpha, 'dataset':dataset, 'epoch':epoch},
                                        'inner':{'adv':adv, 'adv_lambda':adv_lambda, 'acc_lambda':acc_lambda, 'ood_lambda':ood_lambda}, 
                                        'outer':{'nop_outer':nop_outer, 'adv_outer':adv_outer, 'ood_outer':ood_outer, 'flp_outer':flp_outer, 'mgda':mgda, 'constrain':constrain, 'constrain_min':constrain_min, 
                                                'temperature':temperature, 'fx':fx, 'nop_later':nop_later, 'adv_later':adv_later, 'grad_norm':grad_norm}}
                        
                        config_parser.read_dict(config_dict)
                        config_parser.write(open(os.path.join(logpath, config_name + '.ini'), 'w'))

                        logging.info("now running: " + config_name + ' at gpu: ' + args.gpu)
                        run(config_parser)