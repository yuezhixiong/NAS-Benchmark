import subprocess, configparser, argparse, logging, sys, os, time
from shutil import copyfile
import pandas as pd

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

def run(arch, model_path, ACC=False, FGSM=False, PGD=False, init_channels='36', epsilon='2', step_num='10'):

  

  logging.info(model_path +'_eps'+ epsilon)
  results = [model_path]
  if ACC:
    attack_args = ['python', 'test.py', '--cutout', '--auxiliary']
    attack_args += ['--gpu', gpu, '--model_path', model_path]
    attack_args += ['--arch', arch, '--init_channels', init_channels]
    proc = subprocess.check_output(attack_args)
    clean_acc = proc.decode('utf-8').split()[-1]
    logging.info('clean_acc ' + clean_acc)
    results.append(clean_acc)

  if FGSM:
    attack = 'FGSM'
    attack_args = ['python', 'test_adv.py', '--cutout', '--auxiliary']
    attack_args += ['--gpu', gpu, '--model_path', model_path, '--arch', arch, '--init_channels', init_channels]
    attack_args += ['--attack', attack, '--epsilon', epsilon]
    proc = subprocess.check_output(attack_args)
    fgsm_acc = proc.decode('utf-8').split()[-1]
    logging.info('FGSM_acc ' + fgsm_acc)
    results.append(fgsm_acc)


  if PGD:
    attack = 'PGD'
    attack_args = ['python', 'test_adv.py', '--cutout', '--auxiliary']
    attack_args += ['--gpu', gpu, '--model_path', model_path, '--epsilon', epsilon, '--step_num', step_num]
    attack_args += ['--attack', attack, '--arch', arch, '--init_channels', init_channels]
    proc = subprocess.check_output(attack_args)
    pgd_acc = proc.decode('utf-8').split()[-1]
    logging.info('PGD_acc ' + pgd_acc)
    results.append(pgd_acc)

  return results



# model_paths = ['randomInit_inner_acc1_outer_adv_nop_mgda_abs10/channel36_cifar10/model{:03d}.pt'.format(i) for i in range(590, 600, 10)]
model_paths = ['models/abs10_best.pt'] #['models/DARTS_V2_c30_best.pt']
arch = 'randomInit_inner_acc1_outer_adv_nop_mgda_abs10' #'DARTS_V2'
epsilons = [str(x) for x in range(1,2)]

all_results = []
for epsilon in epsilons:
  logging.info('epsilon = ' + epsilon)
  for model_path in model_paths:
    results = run(arch, model_path, ACC=False, FGSM=True, PGD=False, epsilon=epsilon, step_num='7', init_channels='36')
    all_results.append(results)

df = pd.DataFrame(all_results)
print(df)
df.to_csv(os.path.join(logpath,'all_results.csv'), index=False)

