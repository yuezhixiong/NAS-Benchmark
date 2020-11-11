import subprocess


save = 'test'
gpu = '6'
adv = 'fast'
inner_lambda = '1'
args = ['python', 'train_search.py', '--unrolled', '--cutout']
args += ['--gpu', gpu]
args += ['--save', save]
args += ['--adv', adv]
args += ['--inner_lambda', inner_lambda]
args += ['--nop_outer', '--MGDA', '--adv_outer']
args += ['--constrain', 'both']
result = subprocess.run(args)
if result.returncode != 0:
    print('An error occurred')
    exit()

args = ['python', 'copy_genotype.py', '--save', save]
subprocess.run(args)

init_channels = '36'
dataset = 'cifar10'
args = ['python', 'train.py', '--cutout', '--auxiliary']
args += ['--gpu', gpu]
args += ['--save', save]
args += ['--arch', save]
args += ['--init_channels', init_channels]
args += ['--dataset', dataset]
subprocess.run(args)

attack = 'PGD'
batch_size = '64'
model_path = '{}/channel{}_{}'.format(save, init_channels, dataset)
args = ['python', 'test_adv.py', '--cutout', '--auxiliary']
args += ['--gpu', gpu]
args += ['--batch_size', batch_size]
args += ['--attack', attack]
args += ['--arch', save]
args += ['--init_channels', init_channels]
subprocess.run(args)
