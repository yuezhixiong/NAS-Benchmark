import subprocess, configparser, argparse, logging, sys, os, time

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='config/')
parser.add_argument('--config', type=str, default='config/')
parser.add_argument('--gpu', type=str, default='1')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.log, time.strftime("%Y%m%d-%H%M%S")+'.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

config_parser = configparser.ConfigParser()
gpu = args.gpu

def run(config):

    save = config.get('other', 'save')
    search = config.getboolean('other', 'search')
    train = config.getboolean('other', 'train')
    test_adv = config.getboolean('other', 'test_adv')

    adv = config.get('inner', 'adv')
    adv_lambda = config.get('inner', 'adv_lambda')
    acc_lambda = config.get('inner', 'acc_lambda')

    nop_outer = config.getboolean('outer', 'nop_outer')
    adv_outer = config.getboolean('outer', 'adv_outer')
    mgda = config.getboolean('outer', 'mgda')
    constrain = config.get('outer', 'constrain')
    temperature = config.get('outer', 'temperature')

    for keys in config:
        print(keys)
        for k in config[keys]:
            print('\t', k, config[keys][k])
    
    if search:
        logging.info("now running search")
        search_args = ['python', 'train_search.py', '--unrolled', '--cutout', '--gpu', gpu, '--save', save]
        #inner
        search_args += ['--adv', adv, '--adv_lambda', adv_lambda, '--acc_lambda', acc_lambda]
        #outer
        if nop_outer:
            search_args += ['--nop_outer']
        if adv_outer:
            search_args += ['--adv_outer']
        if mgda:
            search_args += ['--MGDA']
        search_args += ['--constrain', constrain]
        search_args += ['--temperature', temperature]

        result = subprocess.run(search_args)
        if result.returncode != 0:
            print('An error occurred')
            exit()
        else:
            subprocess.run(['python', 'copy_genotype.py', '--save', save])

    init_channels = '36'
    dataset = 'cifar10'
    if train:
        logging.info("now running train")
        train_args = ['python', 'train.py', '--cutout', '--auxiliary']
        train_args += ['--gpu', gpu]
        train_args += ['--save', save]
        train_args += ['--arch', save]
        train_args += ['--init_channels', init_channels]
        train_args += ['--dataset', dataset]
        subprocess.run(train_args)

    model_path = '{}/channel{}_{}/best_model.pt'.format(save, init_channels, dataset)
    if test_adv:
        logging.info("now running test_adv")
        attack = 'FGSM'
        attack_args = ['python', 'test_adv.py', '--cutout', '--auxiliary']
        attack_args += ['--gpu', gpu, '--model_path', model_path]
        attack_args += ['--attack', attack, '--arch', save, '--init_channels', init_channels]
        proc = subprocess.check_output(attack_args)
        logging.info('FGSM_acc ' + proc.decode('utf-8'))

        # attack = 'PGD'
        # attack_args = ['python', 'test_adv.py', '--cutout', '--auxiliary']
        # attack_args += ['--gpu', gpu, '--model_path', model_path, '--attack', attack]
        # attack_args += ['--arch', save, '--init_channels', init_channels]
        # adv_acc = subprocess.check_output(attack_args)
        # logging.info('PGD_acc ' + str(adv_acc))

# adv_acc_values = [(0, 1), (1, 0)]
adv_acc_values = [(0, 1)]
constrain = 'min'
temperature = 'GumbelB'
nop_outer = 1
mgda = 1

search = 1
train = 0
test_adv = 0

for adv_lambda, acc_lambda in adv_acc_values:
    config_name = 'inner'
    if adv_lambda:
        config_name += '_adv' + str(adv_lambda)
    if acc_lambda:
        config_name += '_acc' + str(acc_lambda)

    config_name += '_outer'
    
    if nop_outer:
        config_name += '_nop'
    if mgda:
        config_name += '_mgda'
    if constrain != 'none':
        config_name += '_' + constrain
    if temperature != 'none':
        config_name += '_temp' + temperature
    
    config_dict = {'other':{'save':config_name, 'search':search, 'train':train, 'test_adv':test_adv},
                    'inner':{'adv':'fast', 'adv_lambda':adv_lambda, 'acc_lambda':acc_lambda}, 
                    'outer':{'nop_outer':nop_outer, 'adv_outer':0, 'mgda':1, 'constrain':constrain, 'temperature':temperature}}
    
    config_parser.read_dict(config_dict)
    config_parser.write(open(os.path.join('config/', config_name) + '.ini', 'w'))

    logging.info("now running: " + config_name + ' at gpu: ' + args.gpu)
    run(config_parser)