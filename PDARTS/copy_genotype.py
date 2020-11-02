import argparse, os

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--save', type=str, default='fgsm_nop_max_etp_cutout_batchsize', help='experiment name')
args = parser.parse_args()

f = open(os.path.join(args.save, 'best_genotype.txt'))
f_list = f.readlines()
f.close()
f = open('./genotypes.py', 'a')
f.write('\n'+args.save+' = '+f_list[0]+'\n')
f.close()
