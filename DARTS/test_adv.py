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


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'svhn'])
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='./saved_models/weights.pt', help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='adv_nop', help='which architecture to use')
parser.add_argument('--attack', type=str, default='PGD', help='which attack to use')

parser.add_argument('--epsilon', default=1, type=int)
parser.add_argument('--step_num', default=10, type=int, help='Step number of PGD attack')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')


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
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)
    _, valid_transform = utils._data_transforms_cifar10(args)
    test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
  elif args.dataset == 'cifar100':
    class_num = 100
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    _, valid_transform = utils._data_transforms_cifar100(args)
    test_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
  elif args.dataset == 'svhn':
    class_num = 10 
    mean = (0.4377, 0.4438, 0.4728)
    std = (0.1980, 0.2010, 0.1970)
    _, valid_transform = utils._data_transforms_svhn(args)
    test_data = dset.SVHN(root=args.data, split='test', download=True, transform=valid_transform)

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, class_num, args.layers, args.auxiliary, genotype)
  model = model.cuda()
  utils.load(model, args.model_path)

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  # _, test_transform = utils._data_transforms_cifar10(args)
  # test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)

  test_queue = torch.utils.data.DataLoader(
      test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=1)

  model.drop_path_prob = 0
  
  mean = torch.FloatTensor(mean).view(3,1,1)
  std = torch.FloatTensor(std).view(3,1,1)
  upper_limit = ((1 - mean)/ std).cuda()
  lower_limit = ((0 - mean)/ std).cuda()
  epsilon = ((args.epsilon / 255.) / std).cuda()
  step_size = 2.5 * epsilon / args.step_num

  print('using {} attack'.format(args.attack))
  if args.attack == 'FGSM':
      test_adv_acc = test_FGSM(model, test_queue, upper_limit, lower_limit, epsilon)
  elif args.attack == 'PGD':  
      test_adv_acc = test_PGD(model, test_queue, upper_limit, lower_limit, epsilon, step_size)
  logging.info('test_adv_acc ' + test_adv_acc)

  print(test_adv_acc)

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def test_FGSM(net, testloader, upper_limit, lower_limit, epsilon):

    net.eval()
    criterion = nn.CrossEntropyLoss().cuda()
    correctNum = 0
    totalNum = 0
    for images,labels in testloader:
        # print(images)
        labels = Variable(labels, requires_grad=False).cuda()
        images = Variable(images, requires_grad=True).cuda()
        logits, _ = net(images)
        loss = criterion(logits, labels)
        loss.backward(retain_graph=True)
        grad = torch.autograd.grad(loss, images, 
                                   retain_graph=False, create_graph=False)[0]
        grad = grad.detach().data
        delta = clamp(epsilon * torch.sign(grad), -epsilon, epsilon)
        delta = clamp(delta, lower_limit.cuda() - images.data, upper_limit.cuda() - images.data)
        adv_input = Variable(images.data + delta, requires_grad=False).cuda()

        adv_logits, _ = net(adv_input)
        
        adv_pred = adv_logits.data.cpu().numpy().argmax(1)
        true_label = labels.data.cpu().numpy()
        # print(adv_pred, true_label)
        correctNum += (adv_pred == true_label).sum().item()
        totalNum += len(labels)
        acc = correctNum / totalNum *100

    return '{:.2f}'.format(acc)

def test_PGD(net, testloader, upper_limit, lower_limit, epsilon, step_size):

    net.eval()
    criterion = nn.CrossEntropyLoss().cuda()
    correctNum = 0
    totalNum = 0
    for images,labels in testloader:
        images = images.cuda()
        labels = Variable(labels, requires_grad=False).cuda()
        adv_input = Variable(images, requires_grad=True).cuda()
        # print(images.min(),images.max())

        for i in range(args.step_num):
          logits, _ = net(adv_input)
          loss = criterion(logits, labels)
          loss.backward(retain_graph=True)
          grad = torch.autograd.grad(loss, adv_input, 
                                    retain_graph=False, create_graph=False)[0]
          grad = grad.detach().data
          adv_images = adv_input.detach().data + step_size * torch.sign(grad)
          delta = clamp(adv_images - images, -epsilon, epsilon)
          adv_images = clamp(images + delta, lower_limit, upper_limit)
          adv_input = Variable(adv_images, requires_grad=True).cuda()
          # print(i, delta*std*255)

        adv_logits, _ = net(adv_input)
        
        adv_pred = adv_logits.data.cpu().numpy().argmax(1)
        true_label = labels.data.cpu().numpy()
        # print(adv_pred, true_label)
        correctNum += (adv_pred == true_label).sum().item()
        totalNum += len(labels)
        acc = correctNum / totalNum *100

    return '{:.2f}'.format(acc)

def infer(test_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(test_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda()

    logits, _ = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

