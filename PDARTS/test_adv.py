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
from model import NetworkImageNet as NetworkLarge

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--dataset', default="CIFAR10", help='cifar10/mit67/sport8/cifar100')
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='adv_nop_train/weights.pt', help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--arch', type=str, default='adv_nop', help='which architecture to use')

parser.add_argument('--attack', type=str, default='FGSM', help='which attack to use')
parser.add_argument('--epsilon', default=8, type=int)
parser.add_argument('--alpha', default=0.8, type=float, help='Step size')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

if args.dataset=="CIFAR100":
    CLASSES = 100
elif args.dataset=="CIFAR10":
    CLASSES = 10
elif args.dataset == 'mit67':
    CLASSES = 67
elif args.dataset == 'sport8':
    CLASSES = 8
elif args.dataset == 'flowers102':
    CLASSES = 102

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  torch.cuda.set_device(args.gpu)
  cudnn.enabled=True
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
  model = model.cuda()
  
  utils.load(model, args.model_path)

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  _, test_transform = utils.data_transforms(args.dataset,args.cutout,args.cutout_length)
  if args.dataset=="CIFAR100":
    test_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=test_transform)
  elif args.dataset=="CIFAR10":
    test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)
  elif args.dataset=="sport8":
    dset_cls = dset.ImageFolder
    val_path = '%s/Sport8/test' %args.data
    test_data = dset_cls(root=val_path, transform=test_transform)
  elif args.dataset=="mit67":
    dset_cls = dset.ImageFolder
    val_path = '%s/MIT67/test' %args.data
    test_data = dset_cls(root=val_path, transform=test_transform)
  elif args.dataset == "flowers102":
    dset_cls = dset.ImageFolder
    val_path = '%s/flowers102/test' % args.tmp_data_dir
    test_data = dset_cls(root=val_path, transform=test_transform)
  test_queue = torch.utils.data.DataLoader(
      test_data, batch_size=args.batch_size, shuffle=False, pin_memory=False, num_workers=1)

  model.drop_path_prob = 0.0
  
  print('using {} attack'.format(args.attack))
  if args.attack == 'FGSM':
      test_adv_acc = test_FGSM(model, test_queue)
  elif args.attack == 'PGD':  
      test_adv_acc = test_PGD(model, test_queue)
  logging.info('test_adv_acc %f', test_adv_acc)

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def test_FGSM(net, testloader):
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)
    std = torch.FloatTensor(cifar10_std).view(3,1,1).cuda()
    mu = torch.FloatTensor(cifar10_mean).view(3,1,1).cuda()
    std = torch.FloatTensor(cifar10_std).view(3,1,1).cuda()
    upper_limit = ((1 - mu)/ std)
    lower_limit = ((0 - mu)/ std)

    epsilon = (args.epsilon / 255.) / std
    epsilon = epsilon.cuda()
    alpha = (args.alpha / 255.) / std
    alpha = alpha.cuda()

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
        delta = clamp(alpha * torch.sign(grad), -epsilon, epsilon)
        delta = clamp(delta, lower_limit.cuda() - images.data, upper_limit.cuda() - images.data)
        adv_input = Variable(images.data + delta, requires_grad=False).cuda()

        adv_logits, _ = net(adv_input)
        
        adv_pred = adv_logits.data.cpu().numpy().argmax(1)
        true_label = labels.data.cpu().numpy()
        # print(adv_pred, true_label)
        correctNum += (adv_pred == true_label).sum().item()
        totalNum += len(labels)
        acc = correctNum / totalNum *100
        print(acc, end='\r')
    
    return acc

def test_PGD(net, testloader, step_num=10):
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)
    std = torch.FloatTensor(cifar10_std).view(3,1,1).cuda()
    mu = torch.FloatTensor(cifar10_mean).view(3,1,1).cuda()
    std = torch.FloatTensor(cifar10_std).view(3,1,1).cuda()
    upper_limit = ((1 - mu)/ std).cuda()
    lower_limit = ((0 - mu)/ std).cuda()

    epsilon = (args.epsilon / 255.) / std
    epsilon = epsilon.cuda()
    alpha = (args.alpha / 255.) / std
    alpha = alpha.cuda()

    net.eval()
    criterion = nn.CrossEntropyLoss().cuda()
    correctNum = 0
    totalNum = 0
    for images,labels in testloader:
        images = images.cuda()
        labels = Variable(labels, requires_grad=False).cuda()
        adv_input = Variable(images, requires_grad=True).cuda()
        # print(images.min(),images.max())

        for i in range(step_num):
          logits, _ = net(adv_input)
          loss = criterion(logits, labels)
          loss.backward(retain_graph=True)
          grad = torch.autograd.grad(loss, adv_input, 
                                    retain_graph=False, create_graph=False)[0]
          grad = grad.detach().data
          adv_images = adv_input.detach().data + alpha * torch.sign(grad)
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
        print(acc, end='\r')
    
    return acc

def infer(test_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(test_queue):
    input = input.cuda()
    target = target.cuda()
    with torch.no_grad():
        logits, _ = model(input)
        loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

