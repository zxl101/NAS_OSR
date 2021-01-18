import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from pdb import set_trace as bp
from operations import *


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, args, distill=False):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self._args = args
        self._distill = distill
        self._kl = nn.KLDivLoss().cuda()
        self.optimizers = [
            torch.optim.Adam(arch_param, lr=args.arch_learning_rate, betas=(0.5, 0.999))#, weight_decay=args.arch_weight_decay)
            for arch_param in self.model.module._arch_parameters ]
        self.latency_weight = args.latency_weight
        # assert len(self.latency_weight) == len(self.optimizers)
        self.latency = 0

        print("architect initialized!")

    def _compute_unrolled_model(self, input, target, target_de, eta, network_optimizer):
        ce_loss, re_loss, kl_loss, _ = self.model(input, target, target_de)
        loss = ce_loss + re_loss + kl_loss
        theta = _concat(self.model.module.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.module.parameters()).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.model.module.parameters())).data + self.network_weight_decay*theta
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
        return unrolled_model

    def step(self, input_train, target_train, target_train_de, input_valid, target_valid, target_valid_de, eta=None, network_optimizer=None, unrolled=False):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        if unrolled:
                loss = self._backward_step_unrolled(input_train, target_train, target_train_de, input_valid, target_valid, target_valid_de, eta, network_optimizer)
        else:
                ce_loss, re_loss, kl_loss = self._backward_step(input_valid, target_valid, target_valid_de)
                loss = ce_loss + re_loss + kl_loss
        loss = torch.mean(loss)
        loss.backward()
        # if loss_latency != 0: loss_latency.backward()
        for optimizer in self.optimizers:
            optimizer.step()
        return loss

    def _backward_step(self, input_valid, target_valid, target_de):
        ce_loss, re_loss, kl_loss, _ = self.model(input_valid, target_valid, target_de)
        loss_latency = 0
        self.latency_supernet = 0
        self.model.module.prun_mode = None
        # for idx in range(len(self.optimizers)):
        #     self.model.module.arch_idx = idx
        #     if self.latency_weight[idx] > 0:
        #         latency = 0
        #         if len(self.model.module._width_mult_list) == 1:
        #             r0 = 1./500; r1 = 499./500
        #             latency = latency + r0 * self.model.module.forward_latency((3, 1024, 2048), alpha=True, beta=False, ratio=False)
        #             latency = latency + r1 * self.model.module.forward_latency((3, 1024, 2048), alpha=False, beta=True, ratio=False)
        #         else:
        #             r0 = 1./500; r1 = 497./500; r2 = 2./500
        #             latency = latency + r0 * self.model.module.forward_latency((3, 1024, 2048), alpha=True, beta=False, ratio=False)
        #             latency = latency + r1 * self.model.module.forward_latency((3, 1024, 2048), alpha=False, beta=True, ratio=False)
        #             latency = latency + r2 * self.model.module.forward_latency((3, 1024, 2048), alpha=False, beta=False, ratio=True)
        #         self.latency_supernet = latency
        #         loss_latency = loss_latency + latency * self.latency_weight[idx]

        return ce_loss, re_loss, kl_loss

    def _backward_step_unrolled(self, input_train, target_train, target_train_de, input_valid, target_valid, target_valid_de, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train, target_train, target_train_de, eta, network_optimizer)
        # unrolled_loss = unrolled_model._loss(input_valid, target_valid, target_valid_de)
        unrolled_model = nn.DataParallel(unrolled_model)
        ce_loss, kl_loss, re_loss, _ = unrolled_model(input_valid, target_valid, target_valid_de)
        unrolled_loss = ce_loss + kl_loss + re_loss

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.module.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.module.parameters()]
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train, target_train_de)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.module.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)
        return unrolled_loss

    def _construct_model_from_theta(self, theta):
        model_new = self.model.module.new()
        model_dict = self.model.module.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, target_de, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        loss = self.model(input, target, target_de)
        grads_p = torch.autograd.grad(loss[:-1], self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2*R, v)
        loss = self.model(input, target, target_de)
        grads_n = torch.autograd.grad(loss[:-1], self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

