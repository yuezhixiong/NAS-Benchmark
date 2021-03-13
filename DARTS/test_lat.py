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
from model import NetworkCIFAR as Network
from thop import profile


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
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'svhn'])
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

CIFAR_CLASSES = 10


def main():
  if (args.arch != 'DARTS_V2') and args.model_path == ('models/DARTS_V2_best.pt'):
    args.model_path = args.arch + '/channel36_{}/best_model.pt'.format(args.dataset)

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

  if args.dataset == 'cifar10':
    class_num = 10
  elif args.dataset == 'cifar100':
    class_num = 100

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, class_num, args.layers, args.auxiliary, genotype)
  model = model.cuda()
  utils.load(model, args.model_path)

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  # _, test_transform = utils._data_transforms_cifar10(args)
  # test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)

  # test_queue = torch.utils.data.DataLoader(
  #     test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  model.drop_path_prob = args.drop_path_prob
  model.eval()
  # test_acc, test_obj = infer(test_queue, model, criterion)

  # test_input = torch.randn(1, 3, 32, 32).cuda()
  # macs, params = profile(model, inputs=(test_input, ))

  # mb = 1e6
  # gb = 1e9
  # macs_mb = macs/mb
  # params_mb = params/mb
  # macs_gb = macs/gb
  # print('MACs: {:.2f}'.format(macs_mb), 'Params: {:.3f}'.format(params_mb))
  # print('{:.3f}'.format(params_mb))
  # print('{:.2f}'.format(macs_mb))
  # logging.info('test_acc %f', test_acc)

  test_input = torch.randn(1, 3, 32, 32, dtype=torch.float).cuda()
  starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
  repetitions = 300
  timings=np.zeros((repetitions,1))
  # GPU-WARM-UP
  for _ in range(10):
    _ = model(test_input)
  # MEASURE PERFORMANCE
  with torch.no_grad():
    for rep in range(repetitions):
      starter.record()
      _ = model(test_input)
      ender.record()
      # WAIT FOR GPU SYNC
      torch.cuda.synchronize()
      curr_time = starter.elapsed_time(ender)
      timings[rep] = curr_time
  mean_syn = np.sum(timings) / repetitions
  std_syn = np.std(timings)
  print('latency: {:.2f} ms'.format(mean_syn))
  print('{:.2f}'.format(mean_syn))

# def infer(test_queue, model, criterion):
#   objs = utils.AvgrageMeter()
#   top1 = utils.AvgrageMeter()
#   top5 = utils.AvgrageMeter()
#   model.eval()

#   for step, (input, target) in enumerate(test_queue):
#     input = Variable(input).cuda()
#     target = Variable(target).cuda()

#     logits, _ = model(input)
#     loss = criterion(logits, target)

#     prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
#     n = input.size(0)
#     objs.update(loss.data.item(), n)
#     top1.update(prec1.data.item(), n)
#     top5.update(prec5.data.item(), n)

#     if step % args.report_freq == 0:
#       logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

#   return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

