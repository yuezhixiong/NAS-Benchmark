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
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkCIFAR as Network
import torch.nn.functional as F


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
parser.add_argument('--save', type=str, default='train_ood', help='experiment name')
# parser.add_argument('--log_save', type=str, default='adv_nop', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS_V2', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

if not os.path.isdir(args.save):
  os.makedirs(args.save)
args.save = '{}/channel{}_{}'.format(args.save, args.init_channels, args.dataset)

print(args.save)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
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
      _, valid_transform = utils._data_transforms_cifar10(args)
      train_transform = valid_transform
      train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
      valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
  elif args.dataset == 'cifar100':
      class_num = 100
      train_transform, valid_transform = utils._data_transforms_cifar100(args)
      train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
      valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
  elif args.dataset == 'svhn':
      class_num = 10 
      train_transform, valid_transform = utils._data_transforms_svhn(args)
      train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)
      valid_data = dset.SVHN(root=args.data, split='test', download=True, transform=valid_transform)

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, class_num, args.layers, args.auxiliary, genotype)
  
  model = model.cuda()

  # print(model)
  # for name, v in model.named_parameters():
  #   print(name, np.prod(v.size()))

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
  # exit()

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

  # ood_transform, _ = utils._data_transforms_svhn(args)
  _, ood_transform = utils._data_transforms_svhn(args)
  # SVHN_MEAN = [0.4377, 0.4438, 0.4728]
  # SVHN_STD = [0.1980, 0.2010, 0.1970]
  # ood_transform = transforms.Compose([
  #   transforms.ToTensor(),
  #   transforms.Normalize(SVHN_MEAN, SVHN_STD),
  #   ])
  ood_data = dset.SVHN(root=args.data, split='train', download=True, transform=ood_transform)
  ood_indices = list(range(len(ood_data)))
  ood_queue = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size,
              sampler=torch.utils.data.sampler.SubsetRandomSampler(ood_indices),
              pin_memory=True, num_workers=4)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

  best_acc = 0
  for epoch in range(args.epochs):
    scheduler.step()
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, criterion, optimizer, ood_queue)
    logging.info('train_acc %f', train_acc)

    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    if epoch%10 == 0:
        utils.save(model, os.path.join(args.save, 'model{:03d}.pt'.format(epoch)))
    if valid_acc > best_acc:
      best_acc = valid_acc
      utils.save(model, os.path.join(args.save, 'best_model.pt'))
    logging.info('best_acc %f', best_acc)
  
  print('{:.2f}'.format(best_acc))
      

def train(train_queue, model, criterion, optimizer, ood_queue):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda()

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)

    # ood loss start
    try:
      ood_input, _ = next(ood_queue_iter)
    except:
      ood_queue_iter = iter(ood_queue)
      ood_input, _ = next(ood_queue_iter)
    # ood_input, ood_target = next(iter(ood_queue))
    # ood_input = Variable(ood_input, requires_grad=False).cuda()

    ood_logits, _ = model(ood_input.cuda())
    ood_loss = F.kl_div(input=F.log_softmax(ood_logits), target=torch.ones_like(ood_logits)/ood_logits.size()[-1])
    loss += ood_loss
    # ood loss end

    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits.detach(), target.detach(), topk=(1, 5))
    n = input.size(0)
    if torch.__version__[0]=='0':
      objs.update(loss.data[0], n)
      top1.update(prec1.data[0], n)
      top5.update(prec5.data[0], n)
    else:
      objs.update(loss.data.item(), n)
      top1.update(prec1.data.item(), n)
      top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda()

    logits, _ = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    if torch.__version__[0]=='0':
      objs.update(loss.data[0], n)
      top1.update(prec1.data[0], n)
      top5.update(prec5.data[0], n)
    else:
      objs.update(loss.data.item(), n)
      top1.update(prec1.data.item(), n)
      top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

