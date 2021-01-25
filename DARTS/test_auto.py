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

  
  batch_size = str(64)
  logging.info(model_path +'_eps'+ epsilon)
  results = []
  if ACC:
    attack_args = ['python', 'test.py', '--cutout', '--auxiliary']
    attack_args += ['--gpu', gpu, '--model_path', model_path, '--batch_size', batch_size]
    attack_args += ['--arch', arch, '--init_channels', init_channels]
    proc = subprocess.check_output(attack_args)
    clean_acc = proc.decode('utf-8').split()[-1]
    logging.info('clean_acc ' + clean_acc)
    results.append(clean_acc)

  if FGSM:
    attack = 'FGSM'
    attack_args = ['python', 'test_adv.py', '--cutout', '--auxiliary']
    attack_args += ['--gpu', gpu, '--model_path', model_path, '--arch', arch, '--init_channels', init_channels]
    attack_args += ['--attack', attack, '--epsilon', epsilon, '--batch_size', batch_size]
    proc = subprocess.check_output(attack_args)
    fgsm_acc = proc.decode('utf-8').split()[-1]
    logging.info('FGSM_acc ' + fgsm_acc)
    results.append(fgsm_acc)


  if PGD:
    attack = 'PGD'
    attack_args = ['python', 'test_adv.py', '--cutout', '--auxiliary', '--batch_size', batch_size]
    attack_args += ['--gpu', gpu, '--model_path', model_path, '--epsilon', epsilon, '--step_num', step_num]
    attack_args += ['--attack', attack, '--arch', arch, '--init_channels', init_channels]
    proc = subprocess.check_output(attack_args)
    pgd_acc = proc.decode('utf-8').split()[-1]
    logging.info('PGD_acc ' + pgd_acc)
    results.append(pgd_acc)

  return results



# model_paths = ['randomInit_inner_acc1_outer_adv_nop_mgda_abs10/channel36_cifar10/model{:03d}.pt'.format(i) for i in range(500, 600, 10)]
# model_paths = ['randomInit_inner_acc1_outer_adv_nop_mgda_abs30/channel36_cifar10/model530.pt']
model_paths = ['models/'+x for x in ['DARTS_V2_c26_best.pt', 'DARTS_V2_c30_best.pt', 'DARTS_V2_c34_best.pt',
               'abs10_best.pt', 'abs20_best.pt', 'abs30_best.pt']] #['models/DARTS_V2_c30_best.pt']
archs = ['DARTS_V2', 'DARTS_V2', 'DARTS_V2', 'randomInit_inner_acc1_outer_adv_nop_mgda_abs10', 
         'randomInit_inner_acc1_outer_adv_nop_mgda_abs20', 'randomInit_inner_acc1_outer_adv_nop_mgda_abs30'] #'DARTS_V2'
channels = [str(x) for x in [26, 30, 34, 36, 36, 36]]

epsilons = [str(x) for x in [2]] # range(1,2)]

all_results = []
for model_path, arch, c in zip(model_paths, archs, channels):
  model_results = [model_path.split('/')[-1].split('.')[0]]
  for epsilon in epsilons:
    results = run(arch, model_path, ACC=True, FGSM=True, PGD=True, epsilon=epsilon, step_num='10', init_channels=c)
    model_results += results
  all_results.append(model_results)

df = pd.DataFrame(all_results)
print(df)
df.to_csv(os.path.join(logpath,'all_results.csv'), index=False)

