import os
import sys
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
import torch.nn.functional as F
from model import NetworkCIFAR as Network

from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='models/DARTS_V2_best.pt', help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS_V2', help='which architecture to use')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

CIFAR_CLASSES = 10


def main():
  if (args.arch != 'DARTS_V2') and args.model_path == ('models/DARTS_V2_best.pt'):
    args.model_path = args.arch + '/channel36_cifar10/best_model.pt'

  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
  model = model.cuda()
  utils.load(model, args.model_path)

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  _, test_transform = utils._data_transforms_cifar10(args)
  test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)
  test_queue = torch.utils.data.DataLoader(
      test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=1)

  _, ood_transform = utils._data_transforms_svhn(args)
  ood_data = dset.SVHN(root=args.data, split='test', download=True, transform=ood_transform)
  ood_queue = torch.utils.data.DataLoader(
      ood_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=1)

  # print('data length', len(test_data), len(ood_data))

  model.drop_path_prob = args.drop_path_prob
  results = infer(test_queue, ood_queue, model, criterion)
  results = ['{:.2%}'.format(x)[:-1] for x in results]
  print('auroc, aupr_in, aupr_out, fpr, DE', results)
  print(results[-2])
  # logging.info('test_acc %f', test_acc)


def infer(test_queue, ood_queue, model, criterion):
  # objs = utils.AvgrageMeter()
  # top1 = utils.AvgrageMeter()
  # top5 = utils.AvgrageMeter()
  model.eval()
  
  in_prob = []
  for step, (input, target) in enumerate(test_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda()

    logits, _ = model(input)
    in_prob += list(F.softmax(logits, dim=-1).max(dim=-1)[0].data.cpu().numpy())
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    # objs.update(loss.data[0], n)
    # top1.update(prec1.data[0], n)
    # top5.update(prec5.data[0], n)

    # if step % args.report_freq == 0:
    #   logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    test_length = step
  
  out_prob = []
  for step, (ood_input, _) in enumerate(ood_queue):
    ood_input = Variable(ood_input).cuda()
    ood_logits, _ = model(ood_input)
    out_prob += list(F.softmax(ood_logits, dim=-1).max(dim=-1)[0].data.cpu().numpy())

    if step >= test_length:
      print('test_length', step)
      break
    
  in_prob, out_prob = np.array(in_prob), np.array(out_prob)
  label = np.concatenate([np.ones_like(in_prob), np.zeros_like(out_prob)])
  prob = np.concatenate([in_prob, out_prob])
  # auroc
  auroc = roc_auc_score(label, prob)
  # aupr in
  precision, recall, _ = precision_recall_curve(label, prob)
  aupr_in = auc(recall, precision)
  # aupr out
  precision, recall, _ = precision_recall_curve(1-label, prob)
  aupr_out = auc(recall, precision)
  # fpr when tpr95
  fpr = utils.fpr_tpr95(in_prob, out_prob)
  # detection error
  DE = utils.DE(in_prob, out_prob)
  return auroc, aupr_in, aupr_out, fpr, DE


if __name__ == '__main__':
  main() 
