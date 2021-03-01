import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
import numpy as np


class MixedOp(nn.Module):

    def __init__(self, C, stride, switch, p):
        super(MixedOp, self).__init__()
        self.m_ops = nn.ModuleList()
        self.p = p
        for i in range(len(switch)):
            if switch[i]:
                primitive = PRIMITIVES[i]
                op = OPS[primitive](C, stride, False)
                if 'pool' in primitive:
                    op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
                if isinstance(op, Identity) and p > 0:
                    op = nn.Sequential(op, nn.Dropout(self.p))
                self.m_ops.append(op)
                
    def update_p(self):
        for op in self.m_ops:
            if isinstance(op, nn.Sequential):
                if isinstance(op[0], Identity):
                    op[1].p = self.p
                    
    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self.m_ops))


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, switches, p):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.p = p
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self.cell_ops = nn.ModuleList()
        switch_count = 0
        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, switch=switches[switch_count], p=self.p)
                self.cell_ops.append(op)
                switch_count = switch_count + 1
    
    def update_p(self):
        for op in self.cell_ops:
            op.p = self.p
            op.update_p()

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self.cell_ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3, switches_normal=[], switches_reduce=[], p=0.0, largemode=False):
        super(Network, self).__init__()
        self._C = C
        self._C_list = []
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.p = p
        self.switches_normal = switches_normal
        self.switches_reduce = switches_reduce
        self.largemode=largemode
        switch_ons = []
        for i in range(len(switches_normal)):
            ons = 0
            for j in range(len(switches_normal[i])):
                if switches_normal[i][j]:
                    ons = ons + 1
            switch_ons.append(ons)
            ons = 0
        self.switch_on = switch_ons[0]
        
        if self.largemode:
            self.stem0 = nn.Sequential(
                nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C),
            )

            self.stem1 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C),
            )
            C_prev_prev, C_prev, C_curr = C, C, C
        else:
            C_curr = stem_multiplier*C
            self.stem = nn.Sequential(
                nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
                nn.BatchNorm2d(C_curr)
            )
            C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
            
        self.cells = nn.ModuleList()
        reduction_prev = self.largemode
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
                cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, switches_reduce, self.p)
            else:
                reduction = False
                cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, switches_normal, self.p)
            reduction_prev = reduction
            self._C_list.append(C_curr)
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier*C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def forward(self, input):
        if self.largemode:
            s0 = self.stem0(input)
            s1 = self.stem1(s0)
        else:
            s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                if self.alphas_reduce.size(1) == 1:
                    weights = F.softmax(self.alphas_reduce, dim=0)
                else:
                    weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                if self.alphas_normal.size(1) == 1:
                    weights = F.softmax(self.alphas_normal, dim=0)
                else:
                    weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))
        return logits

    def update_p(self):
        for cell in self.cells:
            cell.p = self.p
            cell.update_p()
    
    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target) 

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2+i))
        num_ops = self.switch_on
        self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]
    
    def arch_parameters(self):
        return self._arch_parameters


    def param_number(self, constrain, constrain_min, constrain_max=None):
        # tau = self.tau
        # constrain = self.args.constrain
        # constrain_max = torch.Tensor([self.args.constrain_max]).cuda()
        constrain_min = torch.Tensor([constrain_min]).cuda()
        def compute_u(C, is_reduction):
            a = np.array([0, 0, 0, 0, 2*(C**2+11*C), 2*(C**2+27*C), C**2+11*C, C**2+27*C]).reshape(8, 1)
            u = np.repeat(a, 14, axis=1)
            switches_array = self.switches_normal
            if is_reduction:
                u[3, :] = u[3, :] + np.array([C**2+2*C, C**2+2*C, C**2+2*C, C**2+2*C, 0, C**2+2*C, C**2+2*C, 0, 0, C**2+2*C, C**2+2*C, 0, 0, 0])
                switches_array = self.switches_reduce
            switches_index = torch.nonzero(torch.from_numpy(np.array(switches_array)).cuda().float())
            u = torch.from_numpy(u).cuda().float().t()
            switches_u = u[switches_index[:,0], switches_index[:,1]].reshape(u.size()[0], -1)
            return switches_u

        loss = 0
        for i in range(self._layers):
            if self.cells[i].reduction:
                if self.alphas_reduce.size(1) == 1:
                    alpha = F.softmax(self.alphas_reduce, dim=0)
                else:
                    alpha = F.softmax(self.alphas_reduce, dim=-1)
                u = compute_u(self._C_list[i], is_reduction=True)
            else:
                if self.alphas_normal.size(1) == 1:
                    alpha = F.softmax(self.alphas_normal, dim=0)
                else:
                     alpha = F.softmax(self.alphas_normal, dim=-1)
                u = compute_u(self._C_list[i], is_reduction=False)
            loss += torch.mul(alpha, u).sum()
        
        loss = loss / 1e5
        if constrain=='max':
            return torch.max(torch.ones(1).cuda(), loss-constrain_max)[0]
        elif constrain=='min':
            # return torch.max(Variable(torch.ones(1)).cuda(), constrain_min-loss)[0]
            return torch.max(constrain_min, loss)[0]
        elif constrain=='both':
            # return torch.min(constrain_max, torch.max(constrain_min, loss)[0])[0]
            return loss + torch.max(torch.max(torch.ones(1).cuda(), loss-constrain_max)[0], constrain_min-loss)[0]
        elif constrain=='abs':
            return torch.abs(constrain_min - loss)[0]
        else:
            return loss

    def cal_flops(self):
        def compute_u(c, is_reduction):
            a = np.array([0, 0, 1024, 0, 20808*c + 2048*c*c, 64800*c + 2048*c*c, 10404*c + 1024*c*c, 36100*c + 1024*c*c]).reshape(8, 1)
            u = np.repeat(a, 14, axis=1)
            switches_array = self.switches_normal
            if is_reduction:
                u[3, :] = u[3, :] + np.array([256*c*c, 256*c*c, 256*c*c, 256*c*c, 0, 256*c*c, 256*c*c, 0, 0, 256*c*c, 256*c*c, 0, 0, 0])
                switches_array = self.switches_reduce
            switches_index = torch.nonzero(torch.from_numpy(np.array(switches_array)).cuda().float())
            u = torch.from_numpy(u).cuda().float().t()
            switches_u = u[switches_index[:,0], switches_index[:,1]].reshape(u.size()[0], -1)
            return switches_u

        loss = 0
        for i in range(self._layers):
            if self.cells[i].reduction:
                if self.alphas_reduce.size(1) == 1:
                    alpha = F.softmax(self.alphas_reduce, dim=0)
                else:
                    alpha = F.softmax(self.alphas_reduce, dim=-1)
                u = compute_u(self._C_list[i], is_reduction=True)
            else:
                if self.alphas_normal.size(1) == 1:
                    alpha = F.softmax(self.alphas_normal, dim=0)
                else:
                     alpha = F.softmax(self.alphas_normal, dim=-1)
                u = compute_u(self._C_list[i], is_reduction=False)
            loss += torch.mul(alpha, u).sum()

        loss = loss / 1e8
        constrain_min = Variable(torch.Tensor([self.args.constrain_min])).cuda()
        loss = torch.abs(constrain_min - loss)[0]
        return loss
