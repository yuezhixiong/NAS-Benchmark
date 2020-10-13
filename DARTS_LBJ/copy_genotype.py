import argparse, os

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--save', type=str, default='fgsm_nop_max_etp_cutout_batchsize', help='experiment name')
args = parser.parse_args()

f = open(os.path.join(args.save, 'log.txt'))
f_list = f.readlines()
f.close()
for i in range(len(f_list)-1, 0, -1):
    if f_list[i][24:32] == 'genotype':
        genotype = f_list[i][35:-1]
        break
f = open('./genotypes.py', 'a')
f.write('\n'+args.save+' = '+genotype+'\n')
f.close()

print(args.save)