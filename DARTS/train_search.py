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
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'svhn', 'imagenet'])
parser.add_argument('--batch_size', type=int, default=64, help='batch size') # 64
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=3, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout') # false
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='adv_nop_etp', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--adv', type=str, default='none', choices=['none', 'FGSM', 'PGD', 'fast'], help='use FGSM/PGD advsarial training')
parser.add_argument('--epsilon', default=2, type=int)
parser.add_argument('--acc_lambda', default=1, type=int)
parser.add_argument('--adv_lambda', default=1, type=int)
parser.add_argument('--step_num', type=int, default=5, help='step size m for PGD free adversarial training')

parser.add_argument('--nop_outer', default=False, action='store_true', help='optimize number of parameter')
# parser.add_argument('--entropy', default=False, action='store_true', help='use entropy in arch softmax')
parser.add_argument('--constrain', type=str, default='none', choices=['max', 'min', 'both', 'none'], help='use constraint in model size')
# parser.add_argument('--constrain_size', type=int, default=1e6, help='constrain the model size')
parser.add_argument('--MGDA', default=False, action='store_true', help='use MGDA')
parser.add_argument('--grad_norm', default=False, action='store_true', help='use gradient normalization in MGDA')
parser.add_argument('--adv_outer', default=False, action='store_true', help='use adv in outer loop')
parser.add_argument('--constrain_min', type=float, default=0.25, help='constrain the model size')
parser.add_argument('--constrain_max', type=float, default=1.0, help='constrain the model size')
# parser.add_argument('--temperature', default=False, action='store_true', help='use tau in alpha softmax of param_loss')
parser.add_argument('--temperature', type=str, default='none', choices=['none', 'A', 'B', 'C', 'D', 'GumbelA', 'GumbelB'], help='use tau in alpha softmax of param_loss')
parser.add_argument('--big_alpha', default=False, action='store_true', help='use big_alpha initialization in search')
args = parser.parse_args()

# args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
sols = []
loss_datas = []
alphas_normals = []
alphas_reduces = []

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
        train_transform, _ = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    elif args.dataset == 'cifar100':
        class_num = 100
        train_transform, _ = utils._data_transforms_cifar100(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    elif args.dataset == 'svhn':
        class_num = 10
        train_transform, _ = utils._data_transforms_svhn(args)
        train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)
    elif args.dataset == 'imagenet':
        class_num = 1000
        train_transform, _ = utils._data_transforms_imagenet(args)
        imagenet_train_dir = '/data/dataset/imagenet/ILSVRC2012_img_train_caffemapping/'
        train_data = dset.ImageFolder(imagenet_train_dir, transform=train_transform)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    tau = 1
    if args.temperature == 'none':
        tau = 1
    else:
        tau = 0.1
    model = Network(args.init_channels, class_num, args.layers, criterion, tau=tau, big_alpha=args.big_alpha)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
            model.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay)


    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True, num_workers=1)

    valid_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True, num_workers=1)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    if args.adv == 'PGD':
        args.epochs = int(args.epochs / args.step_num)
        print('free PGD adversarial training for {} epoch'.format(args.epochs))

    architect = Architect(model, args)


    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        # print(model.alphas_normal.data.cpu().numpy())
        alphas_normal = F.softmax(model.alphas_normal/tau, dim=-1).data.cpu().numpy()
        logging.info('alphas_normal[0]: '+str(alphas_normal[0]))
        alphas_normals.append(alphas_normal)
        alphas_reduce = F.softmax(model.alphas_reduce/tau, dim=-1).data.cpu().numpy()
        alphas_reduces.append(alphas_reduce)
        np.save(os.path.join(args.save, 'alphas_normal.npy'), alphas_normals)
        np.save(os.path.join(args.save, 'alphas_reduce.npy'), alphas_reduces)

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, args, epoch)
        logging.info('train_acc %f', train_acc)
        np.save(os.path.join(args.save, 'sols.npy'), sols)
        np.save(os.path.join(args.save, 'loss_data.npy'), loss_datas)

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))

def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, args, epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    if args.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)
    elif args.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif args.dataset == 'svhn':
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
    elif args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    mean = torch.FloatTensor(mean).view(3,1,1)
    std = torch.FloatTensor(std).view(3,1,1)
    upper_limit = ((1 - mean)/ std).cuda()
    lower_limit = ((0 - mean)/ std).cuda()
    epsilon = ((args.epsilon / 255.) / std).cuda()

    if args.temperature == 'none':
        tau = 1
    elif args.temperature == 'A':
        tau = 0.1
    elif args.temperature == 'B':
        tau = 1 / 10**(epoch//10)
    elif args.temperature=='GumbelA':
        tau = 0.1
    elif args.temperature=='GumbelB':
        tau = 1 / 10**(epoch//10)

    logging.info('temperature tau %f', tau)

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input1 = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda(async=True)

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(target_search, requires_grad=False).cuda(async=True)
        
        logs = architect.step(input1, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled, epsilon=epsilon, upper_limit=upper_limit, lower_limit=lower_limit, tau=tau, epoch=epoch)
        sols.append(logs.sol)
        loss_datas.append(logs.loss_data)
        # logging.info('grad_data = ' + str(logs.grad))
        if args.adv == 'FGSM':
            input = Variable(input, requires_grad=True).cuda()

            alpha = epsilon * 1.25
            delta = ((torch.rand(input.size())-0.5)*2).cuda() * epsilon

            logits = model(input)
            loss = criterion(logits, target)
            loss.backward(retain_graph=True)
            grad = torch.autograd.grad(loss, input, retain_graph=False, create_graph=False)[0].detach().data
            
            delta = utils.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            delta = utils.clamp(delta, lower_limit - input.data, upper_limit - input.data)
            adv_input = Variable(input.data + delta, requires_grad=False).cuda()

            loss = args.acc_lambda * criterion(model(input), target) + args.adv_lambda * criterion(model(adv_input), target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
            optimizer.step()

        elif args.adv == 'fast':
            input = Variable(input, requires_grad=True).cuda()

            alpha = epsilon * 1.25
            delta = ((torch.rand(input.size())-0.5)*2).cuda() * epsilon
            delta = utils.clamp(delta, lower_limit - input.data, upper_limit - input.data)
            delta = Variable(delta, requires_grad=True).cuda()

            logits = model(input + delta)
            loss = criterion(logits, target)
            loss.backward(retain_graph=True)
            grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0].detach().data
            
            delta = utils.clamp(delta.data + alpha * torch.sign(grad), -epsilon, epsilon)
            delta = utils.clamp(delta, lower_limit - input.data, upper_limit - input.data)
            adv_input = Variable(input.data + delta, requires_grad=False).cuda()

            loss = 0
            if args.acc_lambda:
                loss += args.acc_lambda * criterion(model(input), target)
            if args.adv_lambda:
                loss += args.adv_lambda * criterion(model(adv_input), target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
            optimizer.step()            
        
        elif args.adv == 'PGD':
            # adv_input = Variable(input, requires_grad=True).cuda()
            input = input.cuda()
            
            delta = ((torch.rand(input.size())-0.5)*2).cuda() * epsilon
            # print(adv_input.size(),delta.size())
            adv_input = Variable(input + delta, requires_grad=True)
            
            for i in range(args.step_num):
                logits = model(adv_input)
                loss = criterion(logits, target)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                # loss.backward()
                grad_adv = torch.autograd.grad(loss, adv_input, retain_graph=False, create_graph=False)[0].detach().data
                # grad_adv = adv_input.grad.detach().data
                # print(grad_adv)
                delta = delta + epsilon * torch.sign(grad_adv)
                delta = utils.clamp(delta, -epsilon, epsilon)
                adv_next = utils.clamp(input + delta, lower_limit, upper_limit)
                adv_input = Variable(adv_next, requires_grad=True).cuda()

                nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
                optimizer.step()
            
        else:
            optimizer.zero_grad()
            logits = model(input1)
            loss = criterion(logits, target)
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
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda(async=True)

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
