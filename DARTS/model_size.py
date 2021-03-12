import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torch.nn.functional as F

from torch.autograd import Variable
from model import NetworkCIFAR as trainNetwork
from model_search import Network as searchNetwork

from genotypes import PRIMITIVES


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'svhn'])
parser.add_argument('--batch_size', type=int, default=96, help='batch size') # 96
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels') # 36
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower') # False
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout') # False
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
# parser.add_argument('--save', type=str, default='adv_nop_train', help='experiment name')
# parser.add_argument('--log_save', type=str, default='adv_nop', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='adv_nop', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()


torch.cuda.set_device(args.gpu)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

def nop(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name and '_ops' in name)/1e6

criterion = nn.CrossEntropyLoss()
class_num = 10

genotype = eval("genotypes.%s" % args.arch)

train_model = trainNetwork(args.init_channels, class_num, args.layers, args.auxiliary, genotype)
train_model = train_model.cuda()

print("_ops param size in MB: ", nop(train_model))
print("train param size in MB: ", utils.count_parameters_in_MB(train_model))
filename = open(os.path.join(args.arch, 'train_model_size.txt'), 'w')
for name, v in train_model.named_parameters():
    print(name, np.prod(v.size()), file=filename)


