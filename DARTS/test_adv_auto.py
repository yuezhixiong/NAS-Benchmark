import subprocess, configparser, argparse, logging, sys, os, time
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='config/')
parser.add_argument('--config', type=str, default='none')
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

def run(arch, model_path, PGD=False, init_channels='36'):

    # model_path = '{}/channel{}_{}/best_model.pt'.format(save, init_channels, dataset)
    logging.info("now running "+model_path)
    attack = 'FGSM'
    attack_args = ['python', 'test_adv.py', '--cutout', '--auxiliary']
    attack_args += ['--gpu', gpu, '--model_path', model_path]
    attack_args += ['--attack', attack, '--arch', arch, '--init_channels', init_channels]
    proc = subprocess.check_output(attack_args)
    logging.info('FGSM_acc ' + proc.decode('utf-8').split()[-1])

    if PGD:
      attack = 'PGD'
      attack_args = ['python', 'test_adv.py', '--cutout', '--auxiliary']
      attack_args += ['--gpu', gpu, '--model_path', model_path]
      attack_args += ['--attack', attack, '--arch', arch, '--init_channels', init_channels]
      proc = subprocess.check_output(attack_args)
      logging.info('PGD_acc ' + proc.decode('utf-8').split()[-1])

model_paths = ["/home/yuezx/NAS-Benchmark/DARTS_LBJ/DARTS_V2/auxiliary0.4_cutout16_batchsize96_channel28_cifar10/best_model.pt"
                ]
arch = 'DARTS_V2'

for model_path in model_paths:
    run(arch, model_path, init_channels='28')
