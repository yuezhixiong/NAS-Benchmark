import torch, sys
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
sys.path.append('../')
from min_norm_solvers import MinNormSolver, gradient_normalizers
import torch.nn.functional as F
from utils import clamp, _concat
from collections import namedtuple
from gumbel_softmax import gumbel_softmax

class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
    self.args = args

  def _compute_unrolled_model(self, input, target, eta, network_optimizer):
    # loss on train data
    loss = self.model._loss(input, target)
    # w
    theta = _concat(self.model.parameters()).data
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    # w_grad + weight_decay * w
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
    # eta: learning rate
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    return unrolled_model

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled, **kwargs): #C, constrain, constrain_size, entropy, lambda_entropy):
    self.optimizer.zero_grad()
    if self.args.adv_outer:
      self.upper_limit = kwargs.get('upper_limit')
      self.lower_limit = kwargs.get('lower_limit')
      self.epsilon = kwargs.get('epsilon')
    # if unrolled:
    #   self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    # else:
    #     self._backward_step(input_valid, target_valid)
    self.tau = kwargs.get('tau')
    logs = self._backward_step(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    self.optimizer.step()
    return logs

  # def _backward_step(self, input_valid, target_valid):
  #   loss = self.model._loss(input_valid, target_valid)
  #   loss.backward()

  def param_number(self, unrolled_model):
    tau = self.tau
    # tau = 0.001
    constrain = self.args.constrain
    constrain_max = Variable(torch.Tensor([self.args.constrain_max])).cuda()
    constrain_min = Variable(torch.Tensor([self.args.constrain_min])).cuda()
    def compute_u(C, is_reduction):
      a = np.array([0, 0, 0, 0, 2*(C**2+9*C), 2*(C**2+25*C), C**2+9*C, C**2+25*C]).reshape(8, 1)
#       u = torch.from_numpy(np.repeat(a, 14, axis=1))
      u = np.repeat(a, 14, axis=1)
      if is_reduction:
        u[3, :] = u[3, :] + np.array([C**2, C**2, C**2, C**2, 0, C**2, C**2, 0, 0, C**2, C**2, 0, 0, 0])
      return Variable(torch.from_numpy(u)).float().cuda()
    loss = 0
    # u = torch.from_numpy(np.array([0, 0, 0, 0, 2*(C**2+9*C), 2*(C**2+25*C), C**2+9*C, C**2+25*C]))
    # C_list = [C, C, 2*C, 2*C, 2*C, 4*C, 4*C, 4*C]
    C_list = unrolled_model._C_list
    for i in range(unrolled_model._layers):
      if unrolled_model.cells[i].reduction:
        if self.args.temperature[:-1] == 'Gumbel':
          alpha = F.softmax(unrolled_model.arch_parameters()[1], dim=-1)
          alpha = gumbel_softmax(alpha, tau=tau, hard=True)
          # print(alpha)
        else:
          alpha = F.softmax(unrolled_model.arch_parameters()[1]/tau, dim=-1)

        u = compute_u(C_list[i], is_reduction=True)
      else:
        if self.args.temperature[:-1] == 'Gumbel':
          alpha = F.softmax(unrolled_model.arch_parameters()[0], dim=-1)
          alpha = gumbel_softmax(alpha, tau=tau, hard=True)
        else:
          alpha = F.softmax(unrolled_model.arch_parameters()[0]/tau, dim=-1)

        u = compute_u(C_list[i], is_reduction=False)
      loss += (2 * torch.mul(alpha, u.t()).sum(dim=1) / Variable(torch.from_numpy(np.repeat(range(2, 6), [2, 3, 4, 5]))).float().cuda()).sum()
    print(alpha[0].data.cpu().numpy())
    loss = loss / 1e6
    if constrain=='max':
      return torch.max(Variable(torch.ones(1)).cuda(), loss-constrain_max)[0]
    elif constrain=='min':
      # return torch.max(Variable(torch.ones(1)).cuda(), constrain_min-loss)[0]
      return torch.max(constrain_min, loss)[0]
    elif constrain=='both':
      # return torch.min(constrain_max, torch.max(constrain_min, loss)[0])[0]

      return loss + torch.max( torch.max(Variable(torch.ones(1)).cuda(), loss-constrain_max)[0], constrain_min-loss)[0]
    else:
      return loss

  def _backward_step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    grads = {}
    loss_data = {}
    self.optimizer.zero_grad()
    if self.args.unrolled:
      unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    else:
      unrolled_model = self.model
    if self.args.adv_outer:
      input_valid = Variable(input_valid.data, requires_grad=True).cuda()
    unrolled_loss = unrolled_model._loss(input_valid, target_valid)
    # if entropy:
    #   entropy_loss = -1.0 * (F.softmax(unrolled_model.arch_parameters()[0], dim=1)*F.log_softmax(unrolled_model.arch_parameters()[0], dim=1)).sum() - \
    #                 (F.softmax(unrolled_model.arch_parameters()[1], dim=1)*F.log_softmax(unrolled_model.arch_parameters()[1], dim=1)).sum()
    #   unrolled_loss = unrolled_loss + lambda_entropy * entropy_loss
    loss_data['acc'] = unrolled_loss.data[0]
    if self.args.adv_outer:
      unrolled_loss.backward(retain_graph=True)
    else:
      unrolled_loss.backward(retain_graph=False)

    grads['acc'] = []
    for param in unrolled_model.arch_parameters():
      if param.grad is not None:
          grads['acc'].append(Variable(param.grad.data.clone(), requires_grad=False))

    # ---- adv loss ----
    if self.args.adv_outer:
      alpha = self.epsilon * 1.25
      delta = ((torch.rand(input_valid.size())-0.5)*2).cuda() * self.epsilon
      adv_grad = torch.autograd.grad(unrolled_loss, input_valid, retain_graph=False, create_graph=False)[0]
      adv_grad = adv_grad.detach().data
      delta = clamp(delta + alpha * torch.sign(adv_grad), -self.epsilon, self.epsilon)
      delta = clamp(delta, self.lower_limit - input_valid.data, self.upper_limit - input_valid.data)
      adv_input = Variable(input_valid.data + delta, requires_grad=False).cuda()
      self.optimizer.zero_grad()
      unrolled_loss_adv = unrolled_model._loss(adv_input, target_valid)
      unrolled_loss_adv.backward()
      loss_data['adv'] = unrolled_loss_adv.data[0]

      grads['adv'] = []
      for param in unrolled_model.arch_parameters():
        if param.grad is not None:
            grads['adv'].append(Variable(param.grad.data.clone(), requires_grad=False))
    # ---- adv loss ----

    # ---- param loss ----
    if self.args.nop_outer:
      self.optimizer.zero_grad()
      param_loss = self.param_number(unrolled_model)
      loss_data['nop'] = param_loss.data[0]
      param_loss.backward()
      grads['nop'] = []
      for param in unrolled_model.arch_parameters():
        if param.grad is not None:
            grads['nop'].append(Variable(param.grad.data.clone(), requires_grad=False))
    # dalpha_param = [v.grad for v in unrolled_model.arch_parameters()]
    # ---- param loss ----
    
    if self.args.grad_norm:
      gn = gradient_normalizers(grads, loss_data, normalization_type='l2') # loss+, loss, l2
    else:
      gn = gradient_normalizers(grads, loss_data, normalization_type='none')

    for t in grads:
      for gr_i in range(len(grads[t])):
        grads[t][gr_i] = grads[t][gr_i] / gn[t]
    
    # ---- MGDA -----
    if self.args.MGDA:
      sol, _ = MinNormSolver.find_min_norm_element([grads[t] for t in grads])
    else:
      sol = [1] * len(grads)
    # print(sol)

    loss = 0
    for kk, t in enumerate(grads):
      if t == 'acc':
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)
        loss += float(sol[kk]) * unrolled_loss
      elif t == 'adv':
        unrolled_loss_adv = unrolled_model._loss(adv_input, target_valid)
        loss += float(sol[kk]) * unrolled_loss_adv
      elif t == 'nop':
        param_loss = self.param_number(unrolled_model)
        loss += float(sol[kk]) * param_loss
    self.optimizer.zero_grad()
    loss.backward()
    # ---- MGDA -----

    if self.args.unrolled:
      dalpha = [v.grad for v in unrolled_model.arch_parameters()]
      vector = [v.grad.data for v in unrolled_model.parameters()]
      implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

      for g, ig in zip(dalpha, implicit_grads):
        g.data.sub_(eta, ig.data)

      for v, g in zip(self.model.arch_parameters(), dalpha):
        if v.grad is None:
          v.grad = Variable(g.data)
        else:
          v.grad.data.copy_(g.data)

    # aa = [[gr.pow(2).sum().data[0] for gr in grads[t]] for t in grads]
    logs = namedtuple("logs", ['sol', 'loss_data'])(sol, loss_data)
    # logs.sol = sol
    # logs.param_loss = param_loss
    print(logs)
    return logs

  def _construct_model_from_theta(self, theta):
    # a new model with the same alpha
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)
    loss = self.model._loss(input, target)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss = self.model._loss(input, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]
