""" Architect controls architecture of cell by computing gradients of alphas """
import copy
import torch
import torch.nn.functional as F
from min_norm_solvers_lbj import MinNormSolver, gradient_normalizers
import numpy as np

class Architect():
    """ Compute gradients of alphas """
    def __init__(self, net, w_momentum, w_weight_decay):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net
        self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay

    def virtual_step(self, trn_X, trn_y, xi, w_optim):
        """
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        loss = self.net.loss(trn_X, trn_y) # L_trn(w)

        # compute gradient
        gradients = torch.autograd.grad(loss, self.net.weights())

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw, g in zip(self.net.weights(), self.v_net.weights(), gradients):
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - xi * (m + g + self.w_weight_decay*w))

            # synchronize alphas
            for a, va in zip(self.net.alphas(), self.v_net.alphas()):
                va.copy_(a)

    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # do virtual step (calc w`)
        self.virtual_step(trn_X, trn_y, xi, w_optim)

        # calc unrolled loss
        unrolled_loss = self.v_net.loss(val_X, val_y) # L_val(w`)

        # ---- MGDA ----
#         print('1'*5)
        grads = {}
        v_alphas = tuple(self.v_net.alphas())
        grads_darts = torch.autograd.grad(unrolled_loss, v_alphas)
        grads['darts'] = [torch.cat(grads_darts, dim=0)]
#         for p in v_alphas:
#             p.grad.data.zero_()
#         print('2'*5)
        param_loss = self.param_number()
#         print('4'*5, param_loss)
        v_alphas = tuple(self.v_net.alphas())
        grads_param = torch.autograd.grad(param_loss, v_alphas) 
        grads['param'] = [torch.cat(grads_param, dim=0)]
#         for p in v_alphas:
#             p.grad.data.zero_()
#         print('3'*5, grads['darts'][0].shape)
        
#         print('-'*5)
#         for i in range(len(grads['darts'])):
#             print(grads['darts'][i].shape)
#         print('-'*5)
#         for i in range(len(grads['param'])):
#             print(grads['param'][i].shape)
#         print('3'*5, grads['param'][2].shape)
        sol, _ = MinNormSolver.find_min_norm_element([grads[t] for t in grads])
        unrolled_loss = self.v_net.loss(val_X, val_y)
        param_loss = self.param_number()
        loss = sol[0] * unrolled_loss + sol[1] * param_loss
#         print('-'*5, sol)
        # ---- MGDA ----

        # compute gradient
        v_alphas = tuple(self.v_net.alphas())
        v_weights = tuple(self.v_net.weights())
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights) 
        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]

        hessian = self.compute_hessian(dw, trn_X, trn_y)

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
                alpha.grad = da - xi*h

    def compute_hessian(self, dw, trn_X, trn_y):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_pos = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(w+) } 

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p -= 2. * eps * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_neg = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(w-) } 

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d

        hessian = [(p-n) / (2.*eps) for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian


    def param_number(self):
        def compute_u(C, is_reduction):
            a = np.array([0, 0, 0, 0, 2*(C**2+9*C), 2*(C**2+25*C), C**2+9*C, C**2+25*C]).reshape(8, 1)
            u = np.repeat(a, 14, axis=1)
            if is_reduction:
                u[3, :] = u[3, :] + np.array([C**2, C**2, C**2, C**2, 0, C**2, C**2, 0, 0, C**2, C**2, 0, 0, 0])
            u = torch.from_numpy(u)
            return u
        loss = 0
        C = self.v_net.net.C
#         print('8'*5, C)
#         print('7'*5, self.v_net.net.n_layers)
#         a = []
#         for alpha in self.v_net.alpha_normal:
#             a.append(alpha)
#         print('6'*5, torch.cat(a, dim=0))
#         print('5'*5, self.v_net.alpha_reduce)
        # u = torch.from_numpy(np.array([0, 0, 0, 0, 2*(C**2+9*C), 2*(C**2+25*C), C**2+9*C, C**2+25*C]))
        C_list = [C, C, 2*C, 2*C, 2*C, 4*C, 4*C, 4*C]
        for i in range(self.v_net.net.n_layers):
            if self.v_net.net.cells[i].reduction:
#                 print('-'*5)
                a = []
                for alpha in self.v_net.alpha_reduce:
                    a.append(alpha)
                alpha = F.softmax(torch.cat(a, dim=0), dim=-1)
                u = compute_u(C_list[i], is_reduction=True)
            else:
#                 print('+'*5)
                a = []
                for alpha in self.v_net.alpha_normal:
                    a.append(alpha)
                alpha = F.softmax(torch.cat(a, dim=0), dim=-1)
                u = compute_u(C_list[i], is_reduction=False)
#             print('8'*5, u.shape, alpha.shape)
            loss_ = (2 * torch.mm(alpha, u.cuda().float()).sum(dim=1) / torch.from_numpy(np.repeat(range(2, 6), [2, 3, 4, 5])).cuda().float()).sum()
#             print('9'*5, i, loss_)
            loss += loss_
        return loss
