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
import copy
from model_search import Network
from genotypes import PRIMITIVES
from genotypes import Genotype
import random
sys.path.append('../')
from min_norm_solvers import MinNormSolver, gradient_normalizers
from torch.autograd import Variable


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--dataset', default="CIFAR10", help='cifar10/mit67/sport8/cifar100/flowers102')
parser.add_argument('--workers', type=int, default=2, help='number of workers to load ifdataset')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=3, help='GPU device id')
parser.add_argument('--epochs', type=int, default=25, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=5, help='total number of layers')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='adv_nop', help='experiment path')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--tmp_data_dir', type=str, default='../data', help='temp data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--dropout_rate', action='append', default=[], help='dropout rate of skip connect')
parser.add_argument('--add_width', action='append', default=['0'], help='add channels')
parser.add_argument('--add_layers', action='append', default=['0'], help='add layers')

parser.add_argument('--constrain', type=str, default='none', choices=['max', 'min', 'none'], help='use constraint in model size')
parser.add_argument('--constrain_size', type=int, default=1e6, help='constrain the model size')
parser.add_argument('--MGDA', default=False, action='store_true', help='use MGDA')
parser.add_argument('--grad_norm', default=False, action='store_true', help='use gradient normalization in MGDA')
parser.add_argument('--original', default=False, action='store_true', help='original version')
parser.add_argument('--fgsm', default=False, action='store_true', help='use fgsm adversarial training')
parser.add_argument('--epsilon', default=2, type=int)
args = parser.parse_args()

utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if args.dataset=="CIFAR100":
    CLASSES = 100
    data_folder = 'cifar-100-python'
elif args.dataset=="CIFAR10":
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
    data_path = '%s/flowers102/train' % args.tmp_data_dir
    val_path = '%s/flowers102/test' % args.tmp_data_dir
    train_data = dset_cls(root=data_path, transform=train_transform)
    valid_data = dset_cls(root=val_path, transform=valid_transform)
    
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
    logging.info('GPU device = %d' % args.gpu)
    logging.info("args = %s", args)
    #  prepare dataset
    train_transform, valid_transform = utils.data_transforms(args.dataset,args.cutout,args.cutout_length)
    if args.dataset == "CIFAR100":
        train_data = dset.CIFAR100(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)
    elif args.dataset == "CIFAR10":
        train_data = dset.CIFAR10(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)
    elif args.dataset == 'mit67':
        dset_cls = dset.ImageFolder
        data_path = '%s/MIT67/train' % args.tmp_data_dir  # 'data/MIT67/train'
        val_path = '%s/MIT67/test' % args.tmp_data_dir  # 'data/MIT67/val'
        train_data = dset_cls(root=data_path, transform=train_transform)
        valid_data = dset_cls(root=val_path, transform=valid_transform)
    elif args.dataset == 'sport8':
        dset_cls = dset.ImageFolder
        data_path = '%s/Sport8/train' % args.tmp_data_dir  # 'data/Sport8/train'
        val_path = '%s/Sport8/test' % args.tmp_data_dir  # 'data/Sport8/val'
        train_data = dset_cls(root=data_path, transform=train_transform)
        valid_data = dset_cls(root=val_path, transform=valid_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))
    random.shuffle(indices)
    
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=args.workers)
    
    # build Network
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    switches = []
    for i in range(14):
        switches.append([True for j in range(len(PRIMITIVES))])
    switches_normal = copy.deepcopy(switches)
    switches_reduce = copy.deepcopy(switches)
    # To be moved to args
    num_to_keep = [5, 3, 1]
    num_to_drop = [3, 2, 2]
    if len(args.add_width) == 3:
        add_width = args.add_width
    else:
        add_width = [0, 0, 0]
    if len(args.add_layers) == 3:
        add_layers = args.add_layers
    else:
        add_layers = [0, 3, 6]
    if len(args.dropout_rate) ==3:
        drop_rate = args.dropout_rate
    else:
        drop_rate = [0.0, 0.0, 0.0]
    eps_no_archs = [10, 10, 10]
    for sp in range(len(num_to_keep)):
        model = Network(args.init_channels + int(add_width[sp]), CLASSES, args.layers + int(add_layers[sp]), criterion, switches_normal=switches_normal, switches_reduce=switches_reduce, p=float(drop_rate[sp]), largemode=args.dataset in utils.LARGE_DATASETS)
        
        model = model.cuda()
        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
        network_params = []
        for k, v in model.named_parameters():
            if not (k.endswith('alphas_normal') or k.endswith('alphas_reduce')):
                network_params.append(v)       
        optimizer = torch.optim.SGD(
                network_params,
                args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay)
        optimizer_a = torch.optim.Adam(model.arch_parameters(),
                    lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(args.epochs), eta_min=args.learning_rate_min)
        sm_dim = -1
        epochs = args.epochs
        eps_no_arch = eps_no_archs[sp]
        scale_factor = 0.2
        for epoch in range(epochs):
            scheduler.step()
            lr = scheduler.get_lr()[0]
            logging.info('Epoch: %d lr: %e', epoch, lr)
            epoch_start = time.time()
            # training
            if epoch < eps_no_arch:
                model.p = float(drop_rate[sp]) * (epochs - epoch - 1) / epochs
                model.update_p()
                # train_acc, train_obj = train(train_queue, valid_queue, model, network_params, criterion, optimizer, optimizer_a, lr, train_arch=False)
                train_acc, train_obj = train(train_queue, valid_queue, model, network_params, criterion, optimizer, optimizer_a, lr, args, train_arch=False)
            else:
                model.p = float(drop_rate[sp]) * np.exp(-(epoch - eps_no_arch) * scale_factor) 
                model.update_p()                
                train_acc, train_obj = train(train_queue, valid_queue, model, network_params, criterion, optimizer, optimizer_a, lr, train_arch=True)
            logging.info('Train_acc %f', train_acc)
            epoch_duration = time.time() - epoch_start
            logging.info('Epoch time: %ds', epoch_duration)
            # validation
            if epochs - epoch < 5:
                valid_acc, valid_obj = infer(valid_queue, model, criterion)
                logging.info('Valid_acc %f', valid_acc)
        utils.save(model, os.path.join(args.save, 'weights.pt'))
        print('------Dropping %d paths------' % num_to_drop[sp])
        # Save switches info for s-c refinement. 
        if sp == len(num_to_keep) - 1:
            switches_normal_2 = copy.deepcopy(switches_normal)
            switches_reduce_2 = copy.deepcopy(switches_reduce)
        # drop operations with low architecture weights
        arch_param = model.arch_parameters()
        normal_prob = F.softmax(arch_param[0], dim=sm_dim).data.cpu().numpy()        
        for i in range(14):
            idxs = []
            for j in range(len(PRIMITIVES)):
                if switches_normal[i][j]:
                    idxs.append(j)
            if sp == len(num_to_keep) - 1:
                drop = get_min_k_no_zero(normal_prob[i, :], idxs, num_to_drop[sp])
            else:
                drop = get_min_k(normal_prob[i, :], num_to_drop[sp])
            for idx in drop:
                switches_normal[i][idxs[idx]] = False
        reduce_prob = F.softmax(arch_param[1], dim=-1).data.cpu().numpy()
        for i in range(14):
            idxs = []
            for j in range(len(PRIMITIVES)):
                if switches_reduce[i][j]:
                    idxs.append(j)
            if sp == len(num_to_keep) - 1:
                drop = get_min_k_no_zero(reduce_prob[i, :], idxs, num_to_drop[sp])
            else:
                drop = get_min_k(reduce_prob[i, :], num_to_drop[sp])
            for idx in drop:
                switches_reduce[i][idxs[idx]] = False
        logging.info('switches_normal = %s', switches_normal)
        logging_switches(switches_normal)
        logging.info('switches_reduce = %s', switches_reduce)
        logging_switches(switches_reduce)
        
        if sp == len(num_to_keep) - 1:
            arch_param = model.arch_parameters()
            normal_prob = F.softmax(arch_param[0], dim=sm_dim).data.cpu().numpy()
            reduce_prob = F.softmax(arch_param[1], dim=sm_dim).data.cpu().numpy()
            normal_final = [0 for idx in range(14)]
            reduce_final = [0 for idx in range(14)]
            # remove all Zero operations
            for i in range(14):
                if switches_normal_2[i][0] == True:
                    normal_prob[i][0] = 0
                normal_final[i] = max(normal_prob[i])
                if switches_reduce_2[i][0] == True:
                    reduce_prob[i][0] = 0
                reduce_final[i] = max(reduce_prob[i])                
            # Generate Architecture
            keep_normal = [0, 1]
            keep_reduce = [0, 1]
            n = 3
            start = 2
            for i in range(3):
                end = start + n
                tbsn = normal_final[start:end]
                tbsr = reduce_final[start:end]
                edge_n = sorted(range(n), key=lambda x: tbsn[x])
                keep_normal.append(edge_n[-1] + start)
                keep_normal.append(edge_n[-2] + start)
                edge_r = sorted(range(n), key=lambda x: tbsr[x])
                keep_reduce.append(edge_r[-1] + start)
                keep_reduce.append(edge_r[-2] + start)
                start = end
                n = n + 1
            for i in range(14):
                if not i in keep_normal:
                    for j in range(len(PRIMITIVES)):
                        switches_normal[i][j] = False
                if not i in keep_reduce:
                    for j in range(len(PRIMITIVES)):
                        switches_reduce[i][j] = False
            # translate switches into genotype
            genotype = parse_network(switches_normal, switches_reduce)
            logging.info(genotype)
            ## restrict skipconnect (normal cell only)
            logging.info('Restricting skipconnect...')
            for sks in range(0, len(PRIMITIVES)+1):
                max_sk = len(PRIMITIVES) - sks
                num_sk = check_sk_number(switches_normal)
                if num_sk < max_sk:
                    continue
                while num_sk > max_sk:
                    normal_prob = delete_min_sk_prob(switches_normal, switches_normal_2, normal_prob)
                    switches_normal = keep_1_on(switches_normal_2, normal_prob)
                    switches_normal = keep_2_branches(switches_normal, normal_prob)
                    num_sk = check_sk_number(switches_normal)
                logging.info('Number of skip-connect: %d', max_sk)
                genotype = parse_network(switches_normal, switches_reduce)
                logging.info(genotype)
    with open(args.save + "/best_genotype.txt", "w") as f:
        f.write(str(genotype))
              

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def train(train_queue, valid_queue, model, network_params, criterion, optimizer, optimizer_a, lr, args, train_arch=True):
# def train(train_queue, valid_queue, model, network_params, criterion, optimizer, optimizer_a, lr, train_arch=True):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    
    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)
        input = input.cuda()
        target = target.cuda()
#         target = target.cuda(async=True)
        # if train_arch:
        if True:
            print('warning if True rather than if train_arch')
            # In the original implementation of DARTS, it is input_search, target_search = next(iter(valid_queue), which slows down
            # the training when using PyTorch 0.4 and above. 
            try:
                input_search, target_search = next(valid_queue_iter)
            except:
                valid_queue_iter = iter(valid_queue)
                input_search, target_search = next(valid_queue_iter)
            input_search = Variable(input_search).cuda()
            target_search = target_search.cuda()
            loss_data = {}
            optimizer_a.zero_grad()
            logits = model(input_search)
            loss_a = criterion(logits, target_search)
            # if entropy:
            #     entropy_loss = -1.0 * (F.softmax(model.arch_parameters()[0], dim=1)*F.log_softmax(model.arch_parameters()[0], dim=1)).sum() - \
            #                 (F.softmax(model.arch_parameters()[1], dim=1)*F.log_softmax(model.arch_parameters()[1], dim=1)).sum()
            #     loss_a = loss_a + lambda_entropy * entropy_loss
            loss_data['darts'] = loss_a.item()
            loss_a.backward()

            if not args.original:
                # ---- MGDA ----
                grads = {}
                grads['darts'] = []
                # nn.utils.clip_grad_norm_(model.arch_parameters(), args.grad_clip)
                for param in model.arch_parameters():
                    if param.grad is not None:
                        grads['darts'].append(Variable(param.grad.data.clone(), requires_grad=False))

                optimizer_a.zero_grad()
                param_loss = model.param_number(args.constraint, args.constrain_size)
                loss_data['param'] = param_loss.item()
                param_loss.backward()
                grads['param'] = []
                for param in model.arch_parameters():
                    if param.grad is not None:
                        grads['param'].append(Variable(param.grad.data.clone(), requires_grad=False))

                # gn = gradient_normalizers(grads, loss_data, normalization_type='loss+')
                if args.grad_norm:
                    gn = gradient_normalizers(grads, loss_data, normalization_type='l2') # loss+, loss, l2
                else:
                    gn = gradient_normalizers(grads, loss_data, normalization_type='none')
                for t in ['darts', 'param']:
                    for gr_i in range(len(grads[t])):
                        grads[t][gr_i] = grads[t][gr_i] / gn[t]

                optimizer_a.zero_grad()
                if args.MGDA:
                    sol, _ = MinNormSolver.find_min_norm_element([grads[t] for t in grads])
                else:
                    sol = [1, 1]
    #             print('-'*8, sol)
                logits = model(input_search)
                loss_a = criterion(logits, target_search)
                # if entropy:
                #     entropy_loss = -1.0 * (F.softmax(model.arch_parameters()[0], dim=1)*F.log_softmax(model.arch_parameters()[0], dim=1)).sum() - \
                #                 (F.softmax(model.arch_parameters()[1], dim=1)*F.log_softmax(model.arch_parameters()[1], dim=1)).sum()
                #     loss_a = loss_a + lambda_entropy * entropy_loss
                param_loss = model.param_number(args.constraint, args.constrain_size)
                loss = sol[0] * loss_a + sol[1] * param_loss
                loss.backward()

            optimizer_a.step()

            # ---- MGDA ----

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        if args.fgsm:
            input.requires_grad = True

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
        nn.utils.clip_grad_norm_(network_params, args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
#         target = target.cuda(async=True)
        target = target.cuda()
        with torch.no_grad():
            logits = model(input)
            loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def parse_network(switches_normal, switches_reduce):

    def _parse_switches(switches):
        n = 2
        start = 0
        gene = []
        step = 4
        for i in range(step):
            end = start + n
            for j in range(start, end):
                for k in range(len(switches[j])):
                    if switches[j][k]:
                        gene.append((PRIMITIVES[k], j - start))
            start = end
            n = n + 1
        return gene
    gene_normal = _parse_switches(switches_normal)
    gene_reduce = _parse_switches(switches_reduce)
    
    concat = range(2, 6)
    
    genotype = Genotype(
        normal=gene_normal, normal_concat=concat, 
        reduce=gene_reduce, reduce_concat=concat
    )
    
    return genotype

def get_min_k(input_in, k):
    input = copy.deepcopy(input_in)
    index = []
    for i in range(k):
        idx = np.argmin(input)
        index.append(idx)
        input[idx] = 1
    
    return index
def get_min_k_no_zero(w_in, idxs, k):
    w = copy.deepcopy(w_in)
    index = []
    if 0 in idxs:
        zf = True 
    else:
        zf = False
    if zf:
        w = w[1:]
        index.append(0)
        k = k - 1
    for i in range(k):
        idx = np.argmin(w)
        w[idx] = 1
        if zf:
            idx = idx + 1
        index.append(idx)
    return index
        
def logging_switches(switches):
    for i in range(len(switches)):
        ops = []
        for j in range(len(switches[i])):
            if switches[i][j]:
                ops.append(PRIMITIVES[j])
        logging.info(ops)
        
def check_sk_number(switches):
    count = 0
    for i in range(len(switches)):
        if switches[i][3]:
            count = count + 1
    
    return count

def delete_min_sk_prob(switches_in, switches_bk, probs_in):
    def _get_sk_idx(switches_in, switches_bk, k):
        if not switches_in[k][3]:
            idx = -1
        else:
            idx = 0
            for i in range(3):
                if switches_bk[k][i]:
                    idx = idx + 1
        return idx
    probs_out = copy.deepcopy(probs_in)
    sk_prob = [1.0 for i in range(len(switches_bk))]
    for i in range(len(switches_in)):
        idx = _get_sk_idx(switches_in, switches_bk, i)
        if not idx == -1:
            sk_prob[i] = probs_out[i][idx]
    d_idx = np.argmin(sk_prob)
    idx = _get_sk_idx(switches_in, switches_bk, d_idx)
    probs_out[d_idx][idx] = 0.0
    
    return probs_out

def keep_1_on(switches_in, probs):
    switches = copy.deepcopy(switches_in)
    for i in range(len(switches)):
        idxs = []
        for j in range(len(PRIMITIVES)):
            if switches[i][j]:
                idxs.append(j)
        drop = get_min_k_no_zero(probs[i, :], idxs, 2)
        for idx in drop:
            switches[i][idxs[idx]] = False            
    return switches

def keep_2_branches(switches_in, probs):
    switches = copy.deepcopy(switches_in)
    final_prob = [0.0 for i in range(len(switches))]
    for i in range(len(switches)):
        final_prob[i] = max(probs[i])
    keep = [0, 1]
    n = 3
    start = 2
    for i in range(3):
        end = start + n
        tb = final_prob[start:end]
        edge = sorted(range(n), key=lambda x: tb[x])
        keep.append(edge[-1] + start)
        keep.append(edge[-2] + start)
        start = end
        n = n + 1
    for i in range(len(switches)):
        if not i in keep:
            for j in range(len(PRIMITIVES)):
                switches[i][j] = False  
    return switches  

if __name__ == '__main__':
    start_time = time.time()
    main() 
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Total searching time: %ds', duration)
