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
from model import NetworkImageNet as NetworkLarge

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--dataset', default="cifar10", help='cifar10/mit67/sport8/cifar100/flowers102')
parser.add_argument('--workers', type=int, default=1, help='number of workers')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='./adv_nop', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='adv_nop', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--tmp_data_dir', type=str, default='../data', help='temp data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--gpu', type=int, default=3, help='GPU device id')
parser.add_argument('--model_path', type=str, default='adv_nop_train', help='path to save the model')

args, unparsed = parser.parse_known_args()
args.model_path = '{}/{}_channel{}'.format(args.save, args.dataset, args.init_channels)
#utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
try:
    os.mkdir(args.model_path) #debug
except:
    pass
fh = logging.FileHandler(os.path.join(args.model_path, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if args.dataset=="cifar100":
    CLASSES = 100
    data_folder = 'cifar-100-python'
elif args.dataset=="cifar10":
    CLASSES = 10
    data_folder = 'cifar-10-batches-py'
elif args.dataset == 'mit67':
    dset_cls = dset.ImageFolder
    CLASSES = 67
    data_path = '%s/MIT67/train' % args.tmp_data_dir
    val_path = '%s/MIT67/test' % args.tmp_data_dir
elif args.dataset == 'sport8':
    dset_cls = dset.ImageFolder
    CLASSES = 8
    data_path = '%s/Sport8/train' % args.tmp_data_dir
    val_path = '%s/Sport8/test' % args.tmp_data_dir
elif args.dataset == "flowers102":
    dset_cls = dset.ImageFolder
    CLASSES = 102
    data_path = '%s/flowers102/train' % args.tmp_data_dir
    val_path = '%s/flowers102/test' % args.tmp_data_dir
elif args.dataset == "svhn":
    CLASSES = 10

def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)
    logging.info("unparsed args = %s", unparsed)
    # num_gpus = torch.cuda.device_count()
    
    # f = open(os.path.join(args.save, 'best_genotype.txt'))
    # f_list = f.readlines()
    # f.close()
    # f = open('./genotypes.py', 'a')
    # f.write(args.arch+' = '+f_list[0]+'\n')
    # f.close()
    
    genotype = eval("genotypes.%s" % args.arch)
    print('---------Genotype---------')
    logging.info(genotype)
    print('--------------------------')
    if args.dataset in utils.LARGE_DATASETS:
        model = NetworkLarge(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
    else:
        model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
    # if num_gpus > 1:
    #     model = torch.nn.DataParallel(model)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
        )

    if args.dataset == "svhn":
        train_transform, valid_transform = utils._data_transforms_svhn(args)
        train_data = dset.SVHN(root=args.tmp_data_dir, split='train', download=True, transform=train_transform)
        valid_data = dset.SVHN(root=args.tmp_data_dir, split='test', download=True, transform=valid_transform)
    else:
        train_transform, valid_transform = utils.data_transforms(args.dataset,args.cutout,args.cutout_length)
        if args.dataset == "cifar100":
            train_data = dset.CIFAR100(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)
            valid_data = dset.CIFAR100(root=args.tmp_data_dir, train=False, download=True, transform=valid_transform)
        elif args.dataset == "cifar10":
            train_data = dset.CIFAR10(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)
            valid_data = dset.CIFAR10(root=args.tmp_data_dir, train=False, download=True, transform=valid_transform)
        elif args.dataset == 'mit67':
            dset_cls = dset.ImageFolder
            data_path = '%s/MIT67/train' % args.tmp_data_dir  
            val_path = '%s/MIT67/test' % args.tmp_data_dir 
            train_data = dset_cls(root=data_path, transform=train_transform)
            valid_data = dset_cls(root=val_path, transform=valid_transform)
        elif args.dataset == 'sport8':
            dset_cls = dset.ImageFolder
            data_path = '%s/Sport8/train' % args.tmp_data_dir 
            val_path = '%s/Sport8/test' % args.tmp_data_dir  
            train_data = dset_cls(root=data_path, transform=train_transform)
            valid_data = dset_cls(root=val_path, transform=valid_transform)
        elif args.dataset == "flowers102":
            dset_cls = dset.ImageFolder
            data_path = '%s/flowers102/train' % args.tmp_data_dir
            val_path = '%s/flowers102/test' % args.tmp_data_dir
            train_data = dset_cls(root=data_path, transform=train_transform)
            valid_data = dset_cls(root=val_path, transform=valid_transform)
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    best_acc = 0.0
    for epoch in range(args.epochs):
        # scheduler.step()
        logging.info('Epoch: %d lr %e', epoch, scheduler.get_lr()[0])
        # if num_gpus > 1:
        #     model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        # else:
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        start_time = time.time()
        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        logging.info('Train_acc: %f', train_acc)

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        if valid_acc > best_acc:
            best_acc = valid_acc
        logging.info('Valid_acc: %f', valid_acc)
        end_time = time.time()
        duration = end_time - start_time
        print('Epoch time: %ds.' % duration )
        utils.save(model, os.path.join(args.model_path, 'weights.pt'))
        scheduler.step()

def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight*loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, _ = utils.accuracy(logits, target, topk=(1,5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('Train Step: %03d Objs: %e Acc: %f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits, _ = model(input)
            loss = criterion(logits, target)

        prec1, _ = utils.accuracy(logits, target, topk=(1,5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('Valid Step: %03d Objs: %e Acc: %f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    start_time = time.time()
    main() 
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Eval time: %ds.', duration)
   
