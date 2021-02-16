import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
# from utils.darts_utils import drop_path, compute_speed, compute_speed_tensorrt
from pdb import set_trace as bp
from seg_oprs import Head
import numpy as np
import os
import shutil
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


reconstruction_function = nn.MSELoss()
reconstruction_function.size_average = False
nllloss = nn.NLLLoss()

# https://github.com/YongfeiYan/Gumbel_Softmax_VAE/blob/master/gumbel_softmax_vae.py
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    
    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


# from cgdl
def sample_gaussian(m, v):
    sample = torch.randn(m.shape).cuda()
    # sample = torch.randn(m.shape)
    z = m + (v**0.5)*sample
    return z

def kl_normal(qm, qv, pm, pv, yh):
	element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm - yh).pow(2) / pv - 1)
	kl = element_wise.sum(-1)
	#print("log var1", qv)
	return kl

class TCONV(nn.Module):
    def __init__(self, in_size, unflat_dim, t_in_ch, t_out_ch, t_kernel, t_padding, t_stride, out_dim, t_latent_dim, outpadding = 0):
        super(TCONV, self).__init__()
        self.in_size = in_size
        self.unflat_dim = unflat_dim
        self.t_in_ch = t_in_ch
        self.t_out_ch = t_out_ch
        self.t_kernel = t_kernel
        self.t_stride = t_stride
        self.t_padding = t_padding
        self.out_dim = out_dim
        self.t_latent_dim = t_latent_dim
        self.weight = 1
        self.bias = 0
        self.outpadding = outpadding

        self.fc = nn.Linear(in_size, t_in_ch * unflat_dim * unflat_dim)
        self.net = nn.Sequential(

            nn.ConvTranspose2d(t_in_ch, t_out_ch, kernel_size=t_kernel, padding=t_padding, stride=t_stride, output_padding=self.outpadding),  # (w-k+2p)/s+1
            nn.BatchNorm2d(t_out_ch),
            nn.LeakyReLU(),
        )
        self.mean_layer = nn.Sequential(
            nn.Linear(t_out_ch*out_dim*out_dim, t_latent_dim)
        )
        self.var_layer = nn.Sequential(
            nn.Linear(t_out_ch*out_dim*out_dim, t_latent_dim)
        )

    def decode(self, x):
        x = self.fc(x)
        x = x.view(-1, self.t_in_ch, self.unflat_dim, self.unflat_dim)
        # print(x.shape)
        h = self.net(x)
        # print(h.shape)
        h_flat = h.view(-1, self.t_out_ch * self.out_dim * self.out_dim)
        mu, var = self.mean_layer(h_flat), self.var_layer(h_flat)
        var = F.softplus(var) + 1e-8
        # mu, var = ut.gaussian_parameters(h, dim=1)
        return h, mu, var

    def decode2(self, x):
        h = self.net(x)
        return h


class FCONV(nn.Module):
    def __init__(self, in_size, unflat_dim, t_in_ch, t_out_ch, t_kernel, t_padding, t_stride):
        super(FCONV, self).__init__()
        self.in_size = in_size
        self.unflat_dim = unflat_dim
        self.t_in_ch = t_in_ch
        self.t_out_ch = t_out_ch
        self.t_kernel = t_kernel
        self.t_stride = t_stride
        self.t_padding = t_padding

        self.fc_final = nn.Linear(in_size, t_in_ch * unflat_dim * unflat_dim)
        self.final = nn.Sequential(
            # nn.PReLU(),
            nn.ConvTranspose2d(t_in_ch, t_out_ch, kernel_size=t_kernel, padding=t_padding, stride=t_stride),  # (w-k+2p)/s+1
            #nn.Sigmoid()
            nn.Tanh()
        )

    def final_decode(self,x):
        x = self.fc_final(x)
        x = x.view(-1, self.t_in_ch, self.unflat_dim, self.unflat_dim)
        x_re = self.final(x)
        return x_re
    def final_decode2(self,x):
        # x = x.view(-1, self.t_in_ch, self.unflat_dim, self.unflat_dim)
        x_re = self.final(x)
        return x_re


class MixedOp(nn.Module):

    def __init__(self, C_in, C_out, stride=1, width_mult_list=[1.]):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self._width_mult_list = width_mult_list
        for primitive in PRIMITIVES:
            op = OPS[primitive](C_in, C_out, stride, True, width_mult_list=width_mult_list)
            self._ops.append(op)

    def set_prun_ratio(self, ratio):
        for op in self._ops:
            op.set_ratio(ratio)

    def forward(self, x, weights, ratios):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0
        if isinstance(ratios[0], torch.Tensor):
            ratio0 = self._width_mult_list[ratios[0].argmax()]
            r_score0 = ratios[0][ratios[0].argmax()]
        else:
            ratio0 = ratios[0]
            r_score0 = 1.
        if isinstance(ratios[1], torch.Tensor):
            ratio1 = self._width_mult_list[ratios[1].argmax()]
            r_score1 = ratios[1][ratios[1].argmax()]
        else:
            ratio1 = ratios[1]
            r_score1 = 1.
        self.set_prun_ratio((ratio0, ratio1))
        for w, op in zip(weights, self._ops):
            # print(x.device)
            # print(w.device)
            # print(r_score0.device)
            result = result + op(x) * w * r_score0 * r_score1
        return result

    def forward_latency(self, size, weights, ratios):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0
        if isinstance(ratios[0], torch.Tensor):
            ratio0 = self._width_mult_list[ratios[0].argmax()]
            r_score0 = ratios[0][ratios[0].argmax()]
        else:
            ratio0 = ratios[0]
            r_score0 = 1.
        if isinstance(ratios[1], torch.Tensor):
            ratio1 = self._width_mult_list[ratios[1].argmax()]
            r_score1 = ratios[1][ratios[1].argmax()]
        else:
            ratio1 = ratios[1]
            r_score1 = 1.
        self.set_prun_ratio((ratio0, ratio1))
        for w, op in zip(weights, self._ops):
            latency, size_out = op.forward_latency(size)
            result = result + latency * w * r_score0 * r_score1
        return result, size_out


class Cell(nn.Module):
    def __init__(self, C_in, C_out=None, down=True, width_mult_list=[1.], latent_dim = None, flat_dim = None):
        super(Cell, self).__init__()
        self._C_in = C_in
        if C_out is None: C_out = C_in
        self._C_out = C_out
        self._down = down
        self._width_mult_list = width_mult_list
        self.latent_dim = latent_dim
        self.flat_dim = flat_dim

        self._op = MixedOp(C_in, C_out, width_mult_list=width_mult_list)
        # self.flatten = nn.Flatten()
        self.mean_layer = nn.Sequential(
            nn.Linear(int(self.flat_dim*self.flat_dim*self._C_out/2), self.latent_dim)
        )
        self.var_layer = nn.Sequential(
            nn.Linear(int(self.flat_dim*self.flat_dim*self._C_out/2), self.latent_dim)
        )

        if self._down:
            self.downsample = MixedOp(C_in, C_in*2, stride=2, width_mult_list=width_mult_list)
    
    def forward(self, input, alphas, ratios):
        # ratios: (in, out, down)
        out = self._op(input, alphas, (ratios[0], ratios[1]))
        # print("Input")
        # print(input.shape)
        # print("Output")
        # print(out.shape)
        # # print("Flat_dim")
        # # print(self.flat_dim)
        # print(self._C_out)
        # print(ratios[0])
        # print(ratios[1])
        # print(ratios)
        # print(self._C_out)
        assert (self._down and (ratios[2] is not None)) or ((not self._down) and (ratios[2] is None))
        down = self.downsample(input, alphas, (ratios[0], ratios[2])) if self._down else None
        # out_flat = out.view(-1, int(self._C_out*self.flat_dim*self.flat_dim/2))
        # mu, var = self.mean_layer(out_flat), self.var_layer(out_flat)
        # var = F.softplus(var) + 1e-8
        return out, down

    def forward_latency(self, size, alphas, ratios):
        # ratios: (in, out, down)
        out = self._op.forward_latency(size, alphas, (ratios[0], ratios[1]))
        assert (self._down and (ratios[2] is not None)) or ((not self._down) and (ratios[2] is None))
        down = self.downsample.forward_latency(size, alphas, (ratios[0], ratios[2])) if self._down else None
        return out, down


class Network_Multi_Path(nn.Module):
    def __init__(self, num_classes=10, in_channel=3, layers=16, criterion=nn.CrossEntropyLoss(ignore_index=-1), Fch=12,
                 width_mult_list=[1.,], prun_modes=['arch_ratio',], stem_head_width=[(1., 1.),], latent_dim32 = 32*2,
                 latent_dim64=64*2, latent_dim128=128*2):
        super(Network_Multi_Path, self).__init__()
        self._num_classes = num_classes
        assert layers >= 3
        self._layers = layers
        self._criterion = criterion
        self._Fch = Fch
        self._width_mult_list = width_mult_list
        self._prun_modes = prun_modes
        self.prun_mode = None # prun_mode is higher priority than _prun_modes
        self._stem_head_width = stem_head_width
        self._flops = 0
        self._params = 0
        self.latent_dim32 = latent_dim32
        self.latent_dim64 = latent_dim64
        self.latent_dim128 = latent_dim128
        self.in_channel = in_channel


        self.classifier32 = nn.Linear(self.latent_dim32, self._num_classes)
        self.one_hot32 = nn.Linear(self._num_classes, self.latent_dim32)

        self.mean_layer32 = nn.Sequential(
            nn.Linear(int(384 * 2 * 2), self.latent_dim32)
        )
        self.var_layer32 = nn.Sequential(
            nn.Linear(int(384 * 2 * 2), self.latent_dim32)
        )

        self.classifier16 = nn.Linear(self.latent_dim64, self._num_classes)
        self.one_hot16 = nn.Linear(self._num_classes, self.latent_dim64)

        self.mean_layer16 = nn.Sequential(
            nn.Linear(int(192 * 4 * 4), self.latent_dim64)
        )
        self.var_layer16 = nn.Sequential(
            nn.Linear(int(192 * 4 * 4), self.latent_dim64)
        )

        self.classifier8 = nn.Linear(self.latent_dim128, self._num_classes)
        self.one_hot8 = nn.Linear(self._num_classes, self.latent_dim128)

        self.mean_layer8 = nn.Sequential(
            nn.Linear(int(96 * 8 * 8), self.latent_dim128)
        )
        self.var_layer8 = nn.Sequential(
            nn.Linear(int(96 * 8 * 8), self.latent_dim128)
        )

        self.stem = nn.ModuleList([
            nn.Sequential(
                ConvNorm(self.in_channel, self.num_filters(2, stem_ratio)*2, kernel_size=3, stride=2, padding=1, bias=False, groups=1, slimmable=False),
                BasicResidual2x(self.num_filters(2, stem_ratio)*2, self.num_filters(4, stem_ratio)*2, kernel_size=3, stride=2, groups=1, slimmable=False),
                BasicResidual2x(self.num_filters(4, stem_ratio)*2, self.num_filters(8, stem_ratio), kernel_size=3, stride=2, groups=1, slimmable=False)
            ) for stem_ratio, _ in self._stem_head_width ])

        self.cells = nn.ModuleList()
        for l in range(layers):
            cells = nn.ModuleList()
            if l == 0:
                # first node has only one input (prev cell's output)
                cells.append(Cell(self.num_filters(8), width_mult_list=width_mult_list, latent_dim=128, flat_dim=4))
            elif l == 1:
                cells.append(Cell(self.num_filters(8), width_mult_list=width_mult_list, latent_dim=128, flat_dim=4))
                cells.append(Cell(self.num_filters(16), width_mult_list=width_mult_list, latent_dim=64, flat_dim=2))
            elif l < layers - 1:
                cells.append(Cell(self.num_filters(8), width_mult_list=width_mult_list, latent_dim=128, flat_dim=4))
                cells.append(Cell(self.num_filters(16), width_mult_list=width_mult_list, latent_dim=64, flat_dim=2))
                cells.append(Cell(self.num_filters(32), down=False, width_mult_list=width_mult_list, latent_dim=32, flat_dim=1))
            else:
                cells.append(Cell(self.num_filters(8), down=False, width_mult_list=width_mult_list, latent_dim=128, flat_dim=4))
                cells.append(Cell(self.num_filters(16), down=False, width_mult_list=width_mult_list, latent_dim=64, flat_dim=2))
                cells.append(Cell(self.num_filters(32), down=False, width_mult_list=width_mult_list, latent_dim=32, flat_dim=1))
            self.cells.append(cells)


        # self.refine32 = nn.ModuleList([
        #     nn.ModuleList([
        #         ConvNorm(self.num_filters(32, head_ratio), self.num_filters(16, head_ratio), kernel_size=1, bias=False, groups=1, slimmable=False),
        #         ConvNorm(self.num_filters(32, head_ratio), self.num_filters(16, head_ratio), kernel_size=3, padding=1, bias=False, groups=1, slimmable=False),
        #         ConvNorm(self.num_filters(16, head_ratio), self.num_filters(8, head_ratio), kernel_size=1, bias=False, groups=1, slimmable=False),
        #         ConvNorm(self.num_filters(16, head_ratio), self.num_filters(8, head_ratio), kernel_size=3, padding=1, bias=False, groups=1, slimmable=False)]) for _, head_ratio in self._stem_head_width ])
        # self.refine16 = nn.ModuleList([
        #     nn.ModuleList([
        #         ConvNorm(self.num_filters(16, head_ratio), self.num_filters(8, head_ratio), kernel_size=1, bias=False, groups=1, slimmable=False),
        #         ConvNorm(self.num_filters(16, head_ratio), self.num_filters(8, head_ratio), kernel_size=3, padding=1, bias=False, groups=1, slimmable=False)]) for _, head_ratio in self._stem_head_width ])
        # self.refine8 = nn.ModuleList([
        #         ConvNorm(self.num_filters(8, head_ratio), self.num_filters(4, head_ratio), kernel_size=3,
        #                         padding=1, bias=False, groups=1, slimmable=False) for _, head_ratio in self._stem_head_width])
        # # self.refine8 = nn.ModuleList([
        # #     ConvNorm(self.num_filters(8, head_ratio) * 3, self.num_filters(4, head_ratio), kernel_size=3,
        # #              padding=1, bias=False, groups=1, slimmable=False) for _, head_ratio in self._stem_head_width])
        # self.refine4 = nn.ModuleList([
        #         ConvNorm(self.num_filters(4, head_ratio), self.num_filters(2, head_ratio), kernel_size=3,
        #                         padding=1, bias=False, groups=1, slimmable=False) for _, head_ratio in self._stem_head_width])
        # self.refine2 = nn.ModuleList([
        #         ConvNorm(self.num_filters(2, head_ratio), self.num_filters(1, head_ratio), kernel_size=3,
        #                         padding=1, bias=False, groups=1, slimmable=False) for _, head_ratio in self._stem_head_width])
        # self.refine1 = nn.ModuleList([
        #     ConvNorm(self.num_filters(1, head_ratio), self.num_filters(1, head_ratio), kernel_size=3,
        #              padding=1, bias=False, groups=1, slimmable=False) for _, head_ratio in self._stem_head_width])
        # self.reconstruct = nn.ModuleList([
        #         ConvNorm(self.num_filters(1, head_ratio), 3, kernel_size=3,
        #                         padding=1, bias=False, groups=1, slimmable=False) for _, head_ratio in self._stem_head_width])


        # contains arch_param names: {"alphas": alphas, "betas": betas, "ratios": ratios}
        self._arch_names = []
        self._arch_parameters = []
        for i in range(len(self._prun_modes)):
            arch_name, arch_param = self._build_arch_parameters(i)
            self._arch_names.append(arch_name)
            self._arch_parameters.append(arch_param)
            self._reset_arch_parameters(i)
        # switch set of arch if we have more than 1 arch
        self.arch_idx = 0

        self.fc1 = nn.Linear(30, 30)
        self.fc2 = nn.Linear(30, 15)
        self.fc3 = nn.Linear(15, 10)

        self.TCONV5_2 = TCONV(self.latent_dim32, 2, 512, 512, 2,
                              0, 2, 4, self.latent_dim64)
        # self.TCONV5_1 = TCONV(32, 4, 512, 512, 1,
        #                       0, 1, 4, 64)

        self.TCONV4_2 = TCONV(self.latent_dim64, 4, 512, 512, 2,
                              0, 2, 8, self.latent_dim128)
        # self.TCONV4_1 = TCONV(64, 8, 512, 256, 1,
        #                       0, 1, 8, 128)

        # self.TCONV3_2 = TCONV(128, 8, 256, 256, 2,
        #                       0, 2, 16, 256)
        self.TCONV3_2 = FCONV(self.latent_dim128, 8, 256, 256, 2, 0, 2)
        # self.TCONV3_1 = TCONV(128, 16, 256, 128, 1,
        #                       0, 1, 16, 256)
        # self.TCONV2_2 = TCONV(256, 16, 128, 128, 2,
        #                       0, 2, 32, 512)
        self.TCONV2_2 = FCONV(256, 16, 256, 128, 2,
                              0, 2)
        # self.TCONV1_2 = TCONV(512, 32, 64, 64, 2,
        #                       0, 2, 64, 512)
        self.TCONV1_2 = FCONV(512, 32, 128, 64, 2,
                              0, 2)
        self.TCONV1_1 = FCONV(512, 64, 64, 3, 1,
                              0, 1)
    
    def num_filters(self, scale, width=1.0):
        return int(np.round(scale * self._Fch * width))

    def new(self):
        model_new = Network(self._num_classes, self._layers, self._criterion, self._Fch).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
                x.data.copy_(y.data)
        return model_new

    def sample_prun_ratio(self, mode="arch_ratio"):
        '''
        mode: "min"|"max"|"random"|"arch_ratio"(default)
        '''
        assert mode in ["min", "max", "random", "arch_ratio"]
        if mode == "arch_ratio":
            ratios = self._arch_names[self.arch_idx]["ratios"]
            ratios0 = getattr(self, ratios[0])
            ratios0_sampled = []
            for layer in range(self._layers - 1):
                ratios0_sampled.append(gumbel_softmax(F.log_softmax(ratios0[layer], dim=-1), hard=True))
            ratios1 = getattr(self, ratios[1])
            ratios1_sampled = []
            for layer in range(self._layers - 1):
                ratios1_sampled.append(gumbel_softmax(F.log_softmax(ratios1[layer], dim=-1), hard=True))
            ratios2 = getattr(self, ratios[2])
            ratios2_sampled = []
            for layer in range(self._layers - 2):
                ratios2_sampled.append(gumbel_softmax(F.log_softmax(ratios2[layer], dim=-1), hard=True))
            return [ratios0_sampled, ratios1_sampled, ratios2_sampled]
        elif mode == "min":
            ratios0_sampled = []
            for layer in range(self._layers - 1):
                ratios0_sampled.append(self._width_mult_list[0])
            ratios1_sampled = []
            for layer in range(self._layers - 1):
                ratios1_sampled.append(self._width_mult_list[0])
            ratios2_sampled = []
            for layer in range(self._layers - 2):
                ratios2_sampled.append(self._width_mult_list[0])
            return [ratios0_sampled, ratios1_sampled, ratios2_sampled]
        elif mode == "max":
            ratios0_sampled = []
            for layer in range(self._layers - 1):
                ratios0_sampled.append(self._width_mult_list[-1])
            ratios1_sampled = []
            for layer in range(self._layers - 1):
                ratios1_sampled.append(self._width_mult_list[-1])
            ratios2_sampled = []
            for layer in range(self._layers - 2):
                ratios2_sampled.append(self._width_mult_list[-1])
            return [ratios0_sampled, ratios1_sampled, ratios2_sampled]
        elif mode == "random":
            ratios0_sampled = []
            for layer in range(self._layers - 1):
                ratios0_sampled.append(np.random.choice(self._width_mult_list))
            ratios1_sampled = []
            for layer in range(self._layers - 1):
                ratios1_sampled.append(np.random.choice(self._width_mult_list))
            ratios2_sampled = []
            for layer in range(self._layers - 2):
                ratios2_sampled.append(np.random.choice(self._width_mult_list))
            return [ratios0_sampled, ratios1_sampled, ratios2_sampled]

    def _forward(self, input, label, label_en):
        # out_prev: cell-state
        # index 0: keep; index 1: down
        stem = self.stem[self.arch_idx]
        # refine16 = self.refine16[self.arch_idx]
        # refine32 = self.refine32[self.arch_idx]
        # refine8 = self.refine8[self.arch_idx]
        # refine4 = self.refine4[self.arch_idx]
        # refine2 = self.refine2[self.arch_idx]
        # refine1 = self.refine1[self.arch_idx]
        # reconstruct = self.reconstruct[self.arch_idx]


        alphas0 = F.softmax(getattr(self, self._arch_names[self.arch_idx]["alphas"][0]), dim=-1)
        alphas1 = F.softmax(getattr(self, self._arch_names[self.arch_idx]["alphas"][1]), dim=-1)
        alphas2 = F.softmax(getattr(self, self._arch_names[self.arch_idx]["alphas"][2]), dim=-1)
        alphas = [alphas0, alphas1, alphas2]
        betas1 = F.softmax(getattr(self, self._arch_names[self.arch_idx]["betas"][0]), dim=-1)
        betas2 = F.softmax(getattr(self, self._arch_names[self.arch_idx]["betas"][1]), dim=-1)
        betas = [None, betas1, betas2]
        if self.prun_mode is not None:
            ratios = self.sample_prun_ratio(mode=self.prun_mode)
        else:
            ratios = self.sample_prun_ratio(mode=self._prun_modes[self.arch_idx])

        out_prev = [[stem(input), None]] # stem: one cell
        # i: layer | j: scale
        mu_var_dic = {}
        for i, cells in enumerate(self.cells):
            # layers
            out = []
            for j, cell in enumerate(cells):
                # scales
                # out,down -- 0: from down; 1: from keep
                out0 = None; out1 = None
                down0 = None; down1 = None
                alpha = alphas[j][i-j]
                # ratio: (in, out, down)
                # int: force #channel; tensor: arch_ratio; float(<=1): force width
                if i == 0 and j == 0:
                    # first cell
                    ratio = (self._stem_head_width[self.arch_idx][0], ratios[j][i-j], ratios[j+1][i-j])
                elif i == self._layers - 1:
                    # cell in last layer
                    if j == 0:
                        ratio = (ratios[j][i-j-1], self._stem_head_width[self.arch_idx][1], None)
                    else:
                        ratio = (ratios[j][i-j], self._stem_head_width[self.arch_idx][1], None)
                elif j == 2:
                    # cell in last scale: no down ratio "None"
                    ratio = (ratios[j][i-j], ratios[j][i-j+1], None)
                else:
                    if j == 0:
                        ratio = (ratios[j][i-j-1], ratios[j][i-j], ratios[j+1][i-j])
                    else:
                        ratio = (ratios[j][i-j], ratios[j][i-j+1], ratios[j+1][i-j])
                # out,down -- 0: from down; 1: from keep
                if j == 0:
                    out1, down1 = cell(out_prev[0][0], alpha, ratio)
                    out.append((out1, down1))
                    # mu_var_dic["{}_{}".format(i,j)] = [mu, var]
                else:
                    if i == j:
                        out0, down0 = cell(out_prev[j-1][1], alpha, ratio)
                        out.append((out0, down0))
                        # mu_var_dic["{}_{}".format(i, j)] = [mu, var]
                    else:
                        if betas[j][i-j-1][0] > 0:
                            out0, down0 = cell(out_prev[j-1][1], alpha, ratio)
                            # mu_var_dic["{}_{}".format(i, j)] = [mu, var]
                        if betas[j][i-j-1][1] > 0:
                            out1, down1 = cell(out_prev[j][0], alpha, ratio)
                            # mu_var_dic["{}_{}".format(i, j)] = [mu, var]
                        out.append((
                            sum(w * out for w, out in zip(betas[j][i-j-1], [out0, out1])),
                            sum(w * down if down is not None else 0 for w, down in zip(betas[j][i-j-1], [down0, down1])),
                            ))
                        # mu = sum(w * mu for w, mu in zip(betas[j][i-j-1], [mu0, mu1]))
                        # var = sum(w * var for w, var in zip(betas[j][i-j-1], [var0, var1]))
                        # mu_var_dic["{}_{}".format(i, j)] = [mu, var]
            out_prev = out
        ###################################
        # out0 = None; out1 = None; out2 = None
        out_flat32 = out[2][0].view(-1, int(384*2*2))
        latent_mu32, latent_var32 = self.mean_layer32(out_flat32), self.var_layer32(out_flat32)
        latent_var32 = F.softplus(latent_var32) + 1e-8
        # latent_mu, latent_var = mu_var_dic["{}_{}".format(self._layers-1, 2)]
        latent32 = sample_gaussian(latent_mu32, latent_var32)
        predict32 = F.log_softmax(self.classifier32(latent32), dim=1)
        # print(predict)
        # predict_test32 = F.log_softmax(self.classifier32(latent_mu), dim=1)
        yh32 = self.one_hot32(label_en)

        out_flat16 = out[1][0].view(-1, int(192 * 4 * 4))
        latent_mu16, latent_var16 = self.mean_layer16(out_flat16), self.var_layer16(out_flat16)
        latent_var16 = F.softplus(latent_var16) + 1e-8
        # latent_mu, latent_var = mu_var_dic["{}_{}".format(self._layers-1, 2)]
        latent16 = sample_gaussian(latent_mu16, latent_var16)
        predict16 = F.log_softmax(self.classifier16(latent16), dim=1)
        # print(predict)
        # predict_test16 = F.log_softmax(self.classifier16(latent_mu), dim=1)
        yh16 = self.one_hot16(label_en)

        out_flat8 = out[0][0].view(-1, int(96 * 8 * 8))
        latent_mu8, latent_var8 = self.mean_layer8(out_flat8), self.var_layer8(out_flat8)
        latent_var8 = F.softplus(latent_var8) + 1e-8
        # latent_mu, latent_var = mu_var_dic["{}_{}".format(self._layers-1, 2)]
        latent8 = sample_gaussian(latent_mu8, latent_var8)
        predict8 = F.log_softmax(self.classifier8(latent8), dim=1)
        # print(predict)
        # predict_test8 = F.log_softmax(self.classifier8(latent_mu), dim=1)
        yh8 = self.one_hot8(label_en)


        qmu5_1 = None
        qvar5_1 = None
        qmu4_1 = None
        qvar4_1 = None

        # structural ladder structure
        dec5_1, mu_dn5_1, var_dn5_1 = self.TCONV5_2.decode(latent32)
        prec_up5_1 = latent_var16 ** (-1)
        prec_dn5_1 = var_dn5_1 ** (-1)
        qmu5_1 = (latent_mu16 * prec_up5_1 + mu_dn5_1 * prec_dn5_1) / (prec_up5_1 + prec_dn5_1)
        qvar5_1 = (prec_up5_1 + prec_dn5_1) ** (-1)
        de_latent5_1 = sample_gaussian(qmu5_1, qvar5_1)

        dec4_1, mu_dn4_1, var_dn4_1 = self.TCONV4_2.decode(de_latent5_1)
        prec_up4_1 = latent_var8 ** (-1)
        prec_dn4_1 = var_dn4_1 ** (-1)
        qmu4_1 = (latent_mu8 * prec_up4_1 + mu_dn4_1 * prec_dn4_1) / (prec_up4_1 + prec_dn4_1)
        qvar4_1 = (prec_up4_1 + prec_dn4_1) ** (-1)
        de_latent4_1 = sample_gaussian(qmu4_1, qvar4_1)

        dec3_1 = self.TCONV3_2.final_decode(de_latent4_1)
        # print(dec3_1.shape)
        dec2_1 = self.TCONV2_2.final_decode2(dec3_1)
        # print(dec2_1.shape)
        dec1_1 = self.TCONV1_2.final_decode2(dec2_1)
        # print(dec1_1.shape)
        reconstructed = self.TCONV1_1.final_decode2(dec1_1)
        # print(reconstructed.shape)

        pred_final = F.log_softmax(self.fc3(self.fc2(self.fc1(torch.cat((predict8, predict16, predict32), dim=1)))),
                                   dim=1)


        latent_mu = [latent_mu32,latent_mu16,latent_mu8]
        latent_var = [latent_var32,latent_var16,latent_var8]
        yh = [yh32,yh16,yh8]
        predict = [predict32,predict16,predict8, pred_final]

        return latent_mu, latent_var, yh, predict, reconstructed, [qmu5_1, qvar5_1], [qmu4_1, qvar4_1]
        ###################################
    
    def forward_latency(self, size, alpha=True, beta=True, ratio=True):
        # out_prev: cell-state
        # index 0: keep; index 1: down
        stem = self.stem[self.arch_idx]

        if alpha:
            alphas0 = F.softmax(getattr(self, self._arch_names[self.arch_idx]["alphas"][0]), dim=-1)
            alphas1 = F.softmax(getattr(self, self._arch_names[self.arch_idx]["alphas"][1]), dim=-1)
            alphas2 = F.softmax(getattr(self, self._arch_names[self.arch_idx]["alphas"][2]), dim=-1)
            alphas = [alphas0, alphas1, alphas2]
        else:
            alphas = [
                torch.ones_like(getattr(self, self._arch_names[self.arch_idx]["alphas"][0])).cuda() * 1./len(PRIMITIVES),
                torch.ones_like(getattr(self, self._arch_names[self.arch_idx]["alphas"][1])).cuda() * 1./len(PRIMITIVES),
                torch.ones_like(getattr(self, self._arch_names[self.arch_idx]["alphas"][2])).cuda() * 1./len(PRIMITIVES)]
        if beta:
            betas1 = F.softmax(getattr(self, self._arch_names[self.arch_idx]["betas"][0]), dim=-1)
            betas2 = F.softmax(getattr(self, self._arch_names[self.arch_idx]["betas"][1]), dim=-1)
            betas = [None, betas1, betas2]
        else:
            betas = [
                None,
                torch.ones_like(getattr(self, self._arch_names[self.arch_idx]["betas"][0])).cuda() * 1./2,
                torch.ones_like(getattr(self, self._arch_names[self.arch_idx]["betas"][1])).cuda() * 1./2]
        if ratio:
            # ratios = self.sample_prun_ratio(mode='arch_ratio')
            if self.prun_mode is not None:
                ratios = self.sample_prun_ratio(mode=self.prun_mode)
            else:
                ratios = self.sample_prun_ratio(mode=self._prun_modes[self.arch_idx])
        else:
            ratios = self.sample_prun_ratio(mode='max')

        stem_latency = 0
        latency, size = stem[0].forward_latency(size); stem_latency = stem_latency + latency
        latency, size = stem[1].forward_latency(size); stem_latency = stem_latency + latency
        latency, size = stem[2].forward_latency(size); stem_latency = stem_latency + latency
        out_prev = [[size, None]] # stem: one cell
        latency_total = [[stem_latency, 0], [0, 0], [0, 0]] # (out, down)

        # i: layer | j: scale
        for i, cells in enumerate(self.cells):
            # layers
            out = []
            latency = []
            for j, cell in enumerate(cells):
                # scales
                # out,down -- 0: from down; 1: from keep
                out0 = None; out1 = None
                down0 = None; down1 = None
                alpha = alphas[j][i-j]
                # ratio: (in, out, down)
                # int: force #channel; tensor: arch_ratio; float(<=1): force width
                if i == 0 and j == 0:
                    # first cell
                    ratio = (self._stem_head_width[self.arch_idx][0], ratios[j][i-j], ratios[j+1][i-j])
                elif i == self._layers - 1:
                    # cell in last layer
                    if j == 0:
                        ratio = (ratios[j][i-j-1], self._stem_head_width[self.arch_idx][1], None)
                    else:
                        ratio = (ratios[j][i-j], self._stem_head_width[self.arch_idx][1], None)
                elif j == 2:
                    # cell in last scale
                    ratio = (ratios[j][i-j], ratios[j][i-j+1], None)
                else:
                    if j == 0:
                        ratio = (ratios[j][i-j-1], ratios[j][i-j], ratios[j+1][i-j])
                    else:
                        ratio = (ratios[j][i-j], ratios[j][i-j+1], ratios[j+1][i-j])
                # out,down -- 0: from down; 1: from keep
                if j == 0:
                    out1, down1 = cell.forward_latency(out_prev[0][0], alpha, ratio)
                    out.append((out1[1], down1[1] if down1 is not None else None))
                    latency.append([out1[0], down1[0] if down1 is not None else None])
                else:
                    if i == j:
                        out0, down0 = cell.forward_latency(out_prev[j-1][1], alpha, ratio)
                        out.append((out0[1], down0[1] if down0 is not None else None))
                        latency.append([out0[0], down0[0] if down0 is not None else None])
                    else:
                        if betas[j][i-j-1][0] > 0:
                            # from down
                            out0, down0 = cell.forward_latency(out_prev[j-1][1], alpha, ratio)
                        if betas[j][i-j-1][1] > 0:
                            # from keep
                            out1, down1 = cell.forward_latency(out_prev[j][0], alpha, ratio)
                        assert (out0 is None and out1 is None) or out0[1] == out1[1]
                        assert (down0 is None and down1 is None) or down0[1] == down1[1]
                        out.append((out0[1], down0[1] if down0 is not None else None))
                        latency.append([
                            sum(w * out for w, out in zip(betas[j][i-j-1], [out0[0], out1[0]])),
                            sum(w * down if down is not None else 0 for w, down in zip(betas[j][i-j-1], [down0[0] if down0 is not None else None, down1[0] if down1 is not None else None])),
                        ])
            out_prev = out
            for ii, lat in enumerate(latency):
                # layer: i | scale: ii
                if ii == 0:
                    # only from keep
                    if lat[0] is not None: latency_total[ii][0] = latency_total[ii][0] + lat[0]
                    if lat[1] is not None: latency_total[ii][1] = latency_total[ii][0] + lat[1]
                else:
                    if i == ii:
                        # only from down
                        if lat[0] is not None: latency_total[ii][0] = latency_total[ii-1][1] + lat[0]
                        if lat[1] is not None: latency_total[ii][1] = latency_total[ii-1][1] + lat[1]
                    else:
                        if lat[0] is not None: latency_total[ii][0] = betas[j][i-j-1][1] * latency_total[ii][0] + betas[j][i-j-1][0] * latency_total[ii-1][1] + lat[0]
                        if lat[1] is not None: latency_total[ii][1] = betas[j][i-j-1][1] * latency_total[ii][0] + betas[j][i-j-1][0] * latency_total[ii-1][1] + lat[1]
        ###################################
        latency0 = latency_total[0][0]
        latency1 = latency_total[1][0]
        latency2 = latency_total[2][0]
        latency = sum([latency0, latency1, latency2])
        return latency
        ###################################

    def latent_space(self, num_class=10, epoch=0, vis = False):
        class_list = []
        for i in range(num_class):
            temp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            temp[i] = 1
            class_list.append(temp)
        class_list = torch.FloatTensor(class_list)
        # print(class_list)
        # encoded = torch.Tensor(num_class,num_class)
        # encoded.zero_()
        # encoded.scatter_(1, class_list.view(-1, 1), 1)
        encoded = class_list.cuda()
        class_latent = self.one_hot32(encoded)
        class_latent = torch.Tensor.cpu(class_latent).detach().numpy()
        pca = PCA(n_components=2)
        pca_fea = pca.fit_transform(class_latent)
        if vis:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter([x[0] for x in pca_fea], [x[1] for x in pca_fea], s=10)
            plt.savefig(os.path.join("train_img", "{}.jpg".format(epoch)))
        return class_latent


    def forward(self, input, target, target_de, pretrain=False):
        # print(target)
        re_loss = 0
        ce_loss = 0
        kl_loss = 0
        beta = 0.5
        lamda = 100
        theta = 1
        if pretrain is not True:
            # "random width": sampled by gambel softmax
            self.prun_mode = None
            for idx in range(len(self._arch_names)):
                self.arch_idx = idx
                latent_mu, latent_var, yh, predict, reconstructed, up32, up16 = self._forward(input, target, target_de)

                re_loss = re_loss + reconstruction_function(input, reconstructed)
                ce_loss = ce_loss + nllloss(predict[0], target)
                if up32[0] != None:
                    # for i in range(3):
                    #     ce_loss = ce_loss + 0.2 * nllloss(predict[i], target)
                    # ce_loss = ce_loss + nllloss(predict[0], target)
                    for i in range(3):
                        pm, pv = torch.zeros(latent_mu[i].shape).cuda(), torch.ones(latent_var[i].shape).cuda()
                        # ce_loss = ce_loss + nllloss(predict[i], target)
                        kl_loss = kl_loss + kl_normal(latent_mu[i], latent_var[i], pm, pv, yh[i])
                    # pm, pv = torch.zeros(latent_mu[0].shape).cuda(), torch.ones(latent_var[0].shape).cuda()
                    # kl32 = kl_normal(latent_mu[0], latent_var[0], pm, pv, yh[0])
                    kl16 = kl_normal(up32[0], up32[1], latent_mu[1], latent_var[1], 0)
                    kl8 = kl_normal(up16[0], up16[1], latent_mu[2], latent_var[2], 0)
                    kl_loss = kl_loss + kl16
                    kl_loss = kl_loss + kl8
                else:
                    for i in range(3):
                        pm, pv = torch.zeros(latent_mu[i].shape).cuda(), torch.ones(latent_var[i].shape).cuda()
                        # ce_loss = ce_loss + nllloss(predict[i], target)
                        kl_loss = kl_loss + kl_normal(latent_mu[i], latent_var[i], pm, pv, yh[i])
        if len(self._width_mult_list) > 1:
            self.prun_mode = "max"
            latent_mu, latent_var, yh, predict, reconstructed, up32, up16 = self._forward(input, target, target_de)

            re = torch.Tensor.cpu(reconstructed).detach().numpy()
            # ori = torch.Tensor.cpu(input).detach().numpy()
            temp = re[0]
            temp = temp * 0.5 + 0.5
            temp = temp * 255
            temp = temp.transpose(1, 2, 0)
            temp = temp.astype(np.uint8)
            img = Image.fromarray(temp)
            img.save(os.path.join("train_img", "check.jpeg"))

            latent_distance = 0
            # latent_space_values = self.latent_space()
            # latent_space_values = torch.FloatTensor(latent_space_values).cuda()
            # # print(latent_space_values.shape)
            # latent_distance = 9999999
            # for i in range(9):
            #     for j in range(i+1,10):
            #         temp_dist = torch.sum((latent_space_values[i] - latent_space_values[j]).pow(2),dim=-1).sqrt()
            #         if latent_distance > temp_dist:
            #             latent_distance = temp_dist
            # # latent_distance = latent_distance / 45
            # # print(latent_distance)

            re_loss = re_loss + reconstruction_function(input, reconstructed)
            ce_loss = ce_loss + nllloss(predict[0], target)
            if up32[0] != None:
                # for i in range(3):
                #     ce_loss = ce_loss + 0.2 * nllloss(predict[i], target)
                # ce_loss = ce_loss + nllloss(predict[0], target)
                for i in range(3):
                    pm, pv = torch.zeros(latent_mu[i].shape).cuda(), torch.ones(latent_var[i].shape).cuda()
                    # ce_loss = ce_loss + nllloss(predict[i], target)
                    kl_loss = kl_loss + kl_normal(latent_mu[i], latent_var[i], pm, pv, yh[i])
                # pm, pv = torch.zeros(latent_mu[0].shape).cuda(), torch.ones(latent_var[0].shape).cuda()
                # kl32 = kl_normal(latent_mu[0], latent_var[0], pm, pv, yh[0])
                kl16 = kl_normal(up32[0], up32[1], latent_mu[1], latent_var[1], 0)
                kl8 = kl_normal(up16[0], up16[1], latent_mu[2], latent_var[2], 0)
                kl_loss = kl_loss + kl16
                kl_loss = kl_loss + kl8
            else:
                for i in range(3):
                    pm, pv = torch.zeros(latent_mu[i].shape).cuda(), torch.ones(latent_var[i].shape).cuda()
                    # ce_loss = ce_loss + nllloss(predict[i], target)
                    kl_loss = kl_loss + kl_normal(latent_mu[i], latent_var[i], pm, pv, yh[i])

            self.prun_mode = "min"
            latent_mu, latent_var, yh, predict, reconstructed, up32, up16 = self._forward(input, target, target_de)
            re_loss = re_loss + reconstruction_function(input, reconstructed)
            ce_loss = ce_loss + nllloss(predict[0], target)
            if up32[0] != None:
                # for i in range(3):
                #     ce_loss = ce_loss + 0.2 * nllloss(predict[i], target)
                # ce_loss = ce_loss + nllloss(predict[0], target)
                for i in range(3):
                    pm, pv = torch.zeros(latent_mu[i].shape).cuda(), torch.ones(latent_var[i].shape).cuda()
                    # ce_loss = ce_loss + nllloss(predict[i], target)
                    kl_loss = kl_loss + kl_normal(latent_mu[i], latent_var[i], pm, pv, yh[i])
                # pm, pv = torch.zeros(latent_mu[0].shape).cuda(), torch.ones(latent_var[0].shape).cuda()
                # kl32 = kl_normal(latent_mu[0], latent_var[0], pm, pv, yh[0])
                kl16 = kl_normal(up32[0], up32[1], latent_mu[1], latent_var[1], 0)
                kl8 = kl_normal(up16[0], up16[1], latent_mu[2], latent_var[2], 0)
                kl_loss = kl_loss + kl16
                kl_loss = kl_loss + kl8
            else:
                for i in range(3):
                    pm, pv = torch.zeros(latent_mu[i].shape).cuda(), torch.ones(latent_var[i].shape).cuda()
                    # ce_loss = ce_loss + nllloss(predict[i], target)
                    kl_loss = kl_loss + kl_normal(latent_mu[i], latent_var[i], pm, pv, yh[i])
            if pretrain == True:
                self.prun_mode = "random"
                latent_mu, latent_var, yh, predict, reconstructed, up32, up16 = self._forward(input, target, target_de)
                re_loss = re_loss + reconstruction_function(input, reconstructed)
                ce_loss = ce_loss + nllloss(predict[0], target)
                if up32[0] != None:
                    # for i in range(3):
                    #     ce_loss = ce_loss + 0.2 * nllloss(predict[i], target)
                    # ce_loss = ce_loss + nllloss(predict[0], target)
                    for i in range(3):
                        pm, pv = torch.zeros(latent_mu[i].shape).cuda(), torch.ones(latent_var[i].shape).cuda()
                        # ce_loss = ce_loss + nllloss(predict[i], target)
                        kl_loss = kl_loss + kl_normal(latent_mu[i], latent_var[i], pm, pv, yh[i])
                    # pm, pv = torch.zeros(latent_mu[0].shape).cuda(), torch.ones(latent_var[0].shape).cuda()
                    # kl32 = kl_normal(latent_mu[0], latent_var[0], pm, pv, yh[0])
                    kl16 = kl_normal(up32[0], up32[1], latent_mu[1], latent_var[1], 0)
                    kl8 = kl_normal(up16[0], up16[1], latent_mu[2], latent_var[2], 0)
                    kl_loss = kl_loss + kl16
                    kl_loss = kl_loss + kl8
                else:
                    for i in range(3):
                        pm, pv = torch.zeros(latent_mu[i].shape).cuda(), torch.ones(latent_var[i].shape).cuda()
                        # ce_loss = ce_loss + nllloss(predict[i], target)
                        kl_loss = kl_loss + kl_normal(latent_mu[i], latent_var[i], pm, pv, yh[i])
                self.prun_mode = "random"
                latent_mu, latent_var, yh, predict, reconstructed, up32, up16 = self._forward(input, target, target_de)
                re_loss = re_loss + reconstruction_function(input, reconstructed)
                ce_loss = ce_loss + nllloss(predict[0], target)
                if up32[0] != None:
                    # for i in range(3):
                    #     ce_loss = ce_loss + 0.2 * nllloss(predict[i], target)
                    # ce_loss = ce_loss + nllloss(predict[0], target)
                    for i in range(3):
                        pm, pv = torch.zeros(latent_mu[i].shape).cuda(), torch.ones(latent_var[i].shape).cuda()
                        # ce_loss = ce_loss + nllloss(predict[i], target)
                        kl_loss = kl_loss + kl_normal(latent_mu[i], latent_var[i], pm, pv, yh[i])
                    # pm, pv = torch.zeros(latent_mu[0].shape).cuda(), torch.ones(latent_var[0].shape).cuda()
                    # kl32 = kl_normal(latent_mu[0], latent_var[0], pm, pv, yh[0])
                    kl16 = kl_normal(up32[0], up32[1], latent_mu[1], latent_var[1], 0)
                    kl8 = kl_normal(up16[0], up16[1], latent_mu[2], latent_var[2], 0)
                    kl_loss = kl_loss + kl16
                    kl_loss = kl_loss + kl8
                else:
                    for i in range(3):
                        pm, pv = torch.zeros(latent_mu[i].shape).cuda(), torch.ones(latent_var[i].shape).cuda()
                        # ce_loss = ce_loss + nllloss(predict[i], target)
                        kl_loss = kl_loss + kl_normal(latent_mu[i], latent_var[i], pm, pv, yh[i])
        elif pretrain == True and len(self._width_mult_list) == 1:
            self.prun_mode = "max"
            latent_mu, latent_var, yh, predict, reconstructed, up32, up16 = self._forward(input, target, target_de)
            re_loss = re_loss + reconstruction_function(input, reconstructed)
            ce_loss = ce_loss + nllloss(predict[0], target)
            if up32[0] != None:
                # for i in range(3):
                #     ce_loss = ce_loss + 0.2 * nllloss(predict[i], target)
                # ce_loss = ce_loss + nllloss(predict[0], target)
                for i in range(3):
                    pm, pv = torch.zeros(latent_mu[i].shape).cuda(), torch.ones(latent_var[i].shape).cuda()
                    # ce_loss = ce_loss + nllloss(predict[i], target)
                    kl_loss = kl_loss + kl_normal(latent_mu[i], latent_var[i], pm, pv, yh[i])
                # pm, pv = torch.zeros(latent_mu[0].shape).cuda(), torch.ones(latent_var[0].shape).cuda()
                # kl32 = kl_normal(latent_mu[0], latent_var[0], pm, pv, yh[0])
                kl16 = kl_normal(up32[0], up32[1], latent_mu[1], latent_var[1], 0)
                kl8 = kl_normal(up16[0], up16[1], latent_mu[2], latent_var[2], 0)
                kl_loss = kl_loss + kl16
                kl_loss = kl_loss + kl8

            else:
                for i in range(3):
                    pm, pv = torch.zeros(latent_mu[i].shape).cuda(), torch.ones(latent_var[i].shape).cuda()
                    # ce_loss = ce_loss + nllloss(predict[i], target)
                    kl_loss = kl_loss + kl_normal(latent_mu[i], latent_var[i], pm, pv, yh[i])
        # print("The reconstruction loss is: {}".format(theta * re_loss))
        # print("The cross entropy loss is: {}".format(lamda * ce_loss))
        # print("The kl divergence loss is: {}".format(beta * kl_loss))
        # loss = torch.mean(theta * re_loss + lamda * ce_loss + beta * kl_loss)
        # loss = ce_loss
        self.prun_mode = "random"
        _, _, _, predict, _, _, _ = self._forward(input, target, target_de)
        # print("ce_loss: {}".format(ce_loss.shape))
        # print("re_loss: {}".format(re_loss.shape))
        # print("kl_loss: {}".format(kl_loss.shape))

        return ce_loss, re_loss, torch.mean(kl_loss), predict

    def _build_arch_parameters(self, idx):
        num_ops = len(PRIMITIVES)

        # define names
        alphas = [ "alpha_"+str(idx)+"_"+str(scale) for scale in [0, 1, 2] ]
        betas = [ "beta_"+str(idx)+"_"+str(scale) for scale in [1, 2] ]

        setattr(self, alphas[0], nn.Parameter(Variable(1e-3*torch.ones(self._layers, num_ops).cuda(), requires_grad=True)))
        setattr(self, alphas[1], nn.Parameter(Variable(1e-3*torch.ones(self._layers-1, num_ops).cuda(), requires_grad=True)))
        setattr(self, alphas[2], nn.Parameter(Variable(1e-3*torch.ones(self._layers-2, num_ops).cuda(), requires_grad=True)))
        # betas are now in-degree probs
        # 0: from down; 1: from keep
        setattr(self, betas[0], nn.Parameter(Variable(1e-3*torch.ones(self._layers-2, 2).cuda(), requires_grad=True)))
        setattr(self, betas[1], nn.Parameter(Variable(1e-3*torch.ones(self._layers-3, 2).cuda(), requires_grad=True)))

        ratios = [ "ratio_"+str(idx)+"_"+str(scale) for scale in [0, 1, 2] ]
        if self._prun_modes[idx] == 'arch_ratio':
            # prunning ratio
            num_widths = len(self._width_mult_list)
        else:
            num_widths = 1
        setattr(self, ratios[0], nn.Parameter(Variable(1e-3*torch.ones(self._layers-1, num_widths).cuda(), requires_grad=True)))
        setattr(self, ratios[1], nn.Parameter(Variable(1e-3*torch.ones(self._layers-1, num_widths).cuda(), requires_grad=True)))
        setattr(self, ratios[2], nn.Parameter(Variable(1e-3*torch.ones(self._layers-2, num_widths).cuda(), requires_grad=True)))
        return {"alphas": alphas, "betas": betas, "ratios": ratios}, [getattr(self, name) for name in alphas] + [getattr(self, name) for name in betas] + [getattr(self, name) for name in ratios]

    def _reset_arch_parameters(self, idx):
        num_ops = len(PRIMITIVES)
        if self._prun_modes[idx] == 'arch_ratio':
            # prunning ratio
            num_widths = len(self._width_mult_list)
        else:
            num_widths = 1

        getattr(self, self._arch_names[idx]["alphas"][0]).data = Variable(1e-3*torch.ones(self._layers, num_ops).cuda(), requires_grad=True)
        getattr(self, self._arch_names[idx]["alphas"][1]).data = Variable(1e-3*torch.ones(self._layers-1, num_ops).cuda(), requires_grad=True)
        getattr(self, self._arch_names[idx]["alphas"][2]).data = Variable(1e-3*torch.ones(self._layers-2, num_ops).cuda(), requires_grad=True)
        getattr(self, self._arch_names[idx]["betas"][0]).data = Variable(1e-3*torch.ones(self._layers-2, 2).cuda(), requires_grad=True)
        getattr(self, self._arch_names[idx]["betas"][1]).data = Variable(1e-3*torch.ones(self._layers-3, 2).cuda(), requires_grad=True)
        getattr(self, self._arch_names[idx]["ratios"][0]).data = Variable(1e-3*torch.ones(self._layers-1, num_widths).cuda(), requires_grad=True)
        getattr(self, self._arch_names[idx]["ratios"][1]).data = Variable(1e-3*torch.ones(self._layers-1, num_widths).cuda(), requires_grad=True)
        getattr(self, self._arch_names[idx]["ratios"][2]).data = Variable(1e-3*torch.ones(self._layers-2, num_widths).cuda(), requires_grad=True)
