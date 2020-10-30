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
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from architect import Architect


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--datapath', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='CIFAR10',choices=["CIFAR10", "CIFAR100", "Sport8", "MIT67", "flowers102"])
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=2, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='adv_nop', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

# parser.add_argument('--entropy', default=False, action='store_true', help='use entropy in arch softmax')
parser.add_argument('--constrain', type=str, default='none', choices=['max', 'min', 'none'], help='use constraint in model size')
parser.add_argument('--constrain_size', type=int, default=1e6, help='constrain the model size')
parser.add_argument('--MGDA', default=False, action='store_true', help='use MGDA')
parser.add_argument('--grad_norm', default=False, action='store_true', help='use gradient normalization in MGDA')
parser.add_argument('--original', default=False, action='store_true', help='original version')
parser.add_argument('--epsilon', default=2, type=int)
parser.add_argument('--fgsm', default=False, action='store_true', help='use fgsm adversarial training')
args = parser.parse_args()

#args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
#utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
if not os.path.exists(args.save):
    os.makedirs(args.save)
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if args.dataset == "CIFAR100":
  CLASSES = 100
elif args.dataset == "CIFAR10":
  CLASSES = 10
elif args.dataset == 'MIT67':
  dset_cls = dset.ImageFolder
  CLASSES = 67
elif args.dataset == 'Sport8':
  dset_cls = dset.ImageFolder
  CLASSES = 8
elif args.dataset == "flowers102":
  dset_cls = dset.ImageFolder
  CLASSES = 102
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

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, CLASSES, args.layers, criterion, largemode=True if args.dataset in utils.LARGE_DATASETS else False)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  train_transform, valid_transform = utils.data_transforms(args.dataset, args.cutout, args.cutout_length)
  if args.dataset == "CIFAR100":
    train_data = dset.CIFAR100(root=args.datapath, train=True, download=True, transform=train_transform)
  elif args.dataset == "CIFAR10":
    train_data = dset.CIFAR10(root=args.datapath, train=True, download=True, transform=train_transform)
  elif args.dataset == 'MIT67':
    dset_cls = dset.ImageFolder
    data_path = '%s/MIT67/train' % args.datapath  # 'data/MIT67/train'
    val_path = '%s/MIT67/test' % args.datapath  # 'data/MIT67/val'
    train_data = dset_cls(root=data_path, transform=train_transform)
    valid_data = dset_cls(root=val_path, transform=valid_transform)
  elif args.dataset == 'Sport8':
    dset_cls = dset.ImageFolder
    data_path = '%s/Sport8/train' % args.datapath  # 'data/Sport8/train'
    val_path = '%s/Sport8/test' % args.datapath  # 'data/Sport8/val'
    train_data = dset_cls(root=data_path, transform=train_transform)
    valid_data = dset_cls(root=val_path, transform=valid_transform)
  elif args.dataset == "flowers102":
    dset_cls = dset.ImageFolder
    data_path = '%s/flowers102/train' % args.datapath
    val_path = '%s/flowers102/test' % args.datapath
    train_data = dset_cls(root=data_path, transform=train_transform)
    valid_data = dset_cls(root=val_path, transform=valid_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))
  import random;random.shuffle(indices)

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)

  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    #print(F.softmax(model.alphas_normal, dim=-1))
    #print(F.softmax(model.alphas_reduce, dim=-1))

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,epoch)
    logging.info('train_acc %f', train_acc)

    # validation
    if args.epochs-epoch<=1:
      with open(args.save + "/best_genotype.txt", "w") as f:
        f.write(str(genotype))
      valid_acc, valid_obj = infer(valid_queue, model, criterion)
      logging.info('valid_acc %f', valid_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)
    input1 = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda()

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    #try:
    #  input_search, target_search = next(valid_queue_iter)
    #except:
    #  valid_queue_iter = iter(valid_queue)
    #  input_search, target_search = next(valid_queue_iter)
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda()

    if epoch>=15:
    # if True:
    #   print('warning if True rather than epoch>15')
      # architect.step(input1, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled, C=args.init_channels)
      architect.step(input1, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input1)
    loss = criterion(logits, target)

    if args.fgsm:
      input = Variable(input, requires_grad=True).cuda()

      mean = (0.4914, 0.4822, 0.4465)
      std = (0.2471, 0.2435, 0.2616)
      mean = torch.FloatTensor(mean).view(3,1,1)
      std = torch.FloatTensor(std).view(3,1,1)
      upper_limit = ((1 - mean)/ std).cuda()
      lower_limit = ((0 - mean)/ std).cuda()

      epsilon = ((args.epsilon / 255.) / std).cuda()
      alpha = epsilon * 1.25
      delta = ((torch.rand(input.size())-0.5)*2).cuda() * epsilon

      loss.backward(retain_graph=True)
      grad = torch.autograd.grad(loss, input, retain_graph=False, create_graph=False)[0].detach().data
      
      delta = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
      delta = clamp(delta, lower_limit - input.data, upper_limit - input.data)
      adv_input = Variable(input.data + delta, requires_grad=False).cuda()
      logits_adv = model(adv_input)  

      loss = criterion(logits_adv, target)      

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg

def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
#     input = input.cuda()
#     target = target.cuda(non_blocking=True)
    input = Variable(input).cuda()
    target = Variable(target).cuda(async=True)
#     with torch.no_grad():
    logits = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 
