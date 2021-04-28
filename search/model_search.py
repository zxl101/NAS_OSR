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

class DeterministicWarmup(object):
    def __init__(self, n=100, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.inc = 1 / n

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc

        self.t = self.t_max if t > self.t_max else t  # 0->1
        return self.t

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
    z = m + (v**0.5)*sample
    return z

def kl_normal(qm, qv, pm, pv, yh):
	element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm - yh).pow(2) / pv - 1)
	kl = element_wise.sum(-1)
	return kl

class TCONV(nn.Module):
    def __init__(self, t_in_ch, t_out_ch, t_kernel, t_padding, t_stride, outpadding = 0):
        super(TCONV, self).__init__()
        self.t_in_ch = t_in_ch
        self.t_out_ch = t_out_ch
        self.t_kernel = t_kernel
        self.t_stride = t_stride
        self.t_padding = t_padding
        self.weight = 1
        self.bias = 0
        self.outpadding = outpadding

        self.net = nn.Sequential(

            nn.ConvTranspose2d(t_in_ch, t_out_ch, kernel_size=t_kernel, padding=t_padding, stride=t_stride, output_padding=self.outpadding),  # (w-k+2p)/s+1
            nn.BatchNorm2d(t_out_ch),
            nn.PReLU(),
        )

    def decode(self, x):
        h = self.net(x)
        # mu, var = ut.gaussian_parameters(h, dim=1)
        return h


class FCONV(nn.Module):
    def __init__(self, t_in_ch, t_out_ch, t_kernel, t_padding, t_stride):
        super(FCONV, self).__init__()

        self.t_in_ch = t_in_ch
        self.t_out_ch = t_out_ch
        self.t_kernel = t_kernel
        self.t_stride = t_stride
        self.t_padding = t_padding


        self.final = nn.Sequential(
            nn.PReLU(),
            nn.ConvTranspose2d(t_in_ch, t_out_ch, kernel_size=t_kernel, padding=t_padding, stride=t_stride),  # (w-k+2p)/s+1
            #nn.Sigmoid()
            nn.Tanh()
        )

    def final_decode(self,x):
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
    def __init__(self, num_classes=10, in_channel=3, layers=16, criterion=nn.CrossEntropyLoss(ignore_index=-1), Fch=16,
                 width_mult_list=[1.,], prun_modes=['arch_ratio',], stem_head_width=[(1., 1.),], latent_dim32 = 32*1,
                 latent_dim64=64*1, latent_dim128=128*1, z_dim=10, temperature=1, beta=1, lamda=1, beta_z=1,
                 total_epoch=50, img_size=32, down_scale_last=4, skip_connect=1):
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
        self.z_dim = z_dim
        self.temperature = temperature
        self.beta = DeterministicWarmup(total_epoch, beta)
        self.lamda = lamda
        self.beta_z = beta_z
        self.skip_connect = skip_connect
        self.img_size = img_size
        self.down_scale_last = down_scale_last
        self.last_size = self.img_size // (2 ** self.down_scale_last)

        self.one_hot32 = nn.Linear(self._num_classes, self.latent_dim32)

        self.mean_layer32 = nn.Sequential(
            nn.Linear(int(1024 * self.last_size * self.last_size), self.latent_dim32 + self.z_dim)
        )
        self.var_layer32 = nn.Sequential(
            nn.Linear(int(1024 * self.last_size * self.last_size), self.latent_dim32 + self.z_dim)
        )


        # self.stem = nn.ModuleList([
        #     nn.Sequential(
        #         ConvNorm(self.in_channel, self.num_filters(4, stem_ratio)*2, kernel_size=3, stride=2, padding=1, bias=False, groups=1, slimmable=False),
        #         BasicResidual2x(self.num_filters(4, stem_ratio)*2, self.num_filters(8, stem_ratio)*2, kernel_size=3, stride=2, groups=1, slimmable=False),
        #         BasicResidual2x(self.num_filters(8, stem_ratio)*2, self.num_filters(16, stem_ratio), kernel_size=3, stride=2, groups=1, slimmable=False)
        #     ) for stem_ratio, _ in self._stem_head_width ])
        self.down1 = ConvNorm(self.in_channel, self.num_filters(4, 1), kernel_size=1, stride=1, padding=0, bias=False, groups=1, slimmable=False)
        self.down2 = BasicResidual2x(self.num_filters(4, 1), self.num_filters(8, 1), kernel_size=3, stride=2, groups=1, slimmable=False)
        self.down4 = BasicResidual2x(self.num_filters(8, 1), self.num_filters(16, 1), kernel_size=3, stride=2, groups=1, slimmable=False)


        self.cells = nn.ModuleList()
        for l in range(layers):
            cells = nn.ModuleList()
            if l == 0:
                # first node has only one input (prev cell's output)
                cells.append(Cell(self.num_filters(16), width_mult_list=width_mult_list, latent_dim=128, flat_dim=4))
            elif l == 1:
                cells.append(Cell(self.num_filters(16), width_mult_list=width_mult_list, latent_dim=128, flat_dim=4))
                cells.append(Cell(self.num_filters(32), width_mult_list=width_mult_list, latent_dim=64, flat_dim=2))
            elif l < layers - 1:
                cells.append(Cell(self.num_filters(16), width_mult_list=width_mult_list, latent_dim=128, flat_dim=4))
                cells.append(Cell(self.num_filters(32), width_mult_list=width_mult_list, latent_dim=64, flat_dim=2))
                cells.append(Cell(self.num_filters(64), down=False, width_mult_list=width_mult_list, latent_dim=32, flat_dim=1))
            else:
                cells.append(Cell(self.num_filters(16), down=False, width_mult_list=width_mult_list, latent_dim=128, flat_dim=4))
                cells.append(Cell(self.num_filters(32), down=False, width_mult_list=width_mult_list, latent_dim=64, flat_dim=2))
                cells.append(Cell(self.num_filters(64), down=False, width_mult_list=width_mult_list, latent_dim=32, flat_dim=1))
            self.cells.append(cells)


        self.dec32 = nn.Linear(self.latent_dim32 + self.z_dim, 1024*self.last_size*self.last_size)
        if self.skip_connect == 0:
            self.up32 = TCONV(1024, 512, t_kernel=3, t_stride=2, t_padding=1, outpadding=1)
            self.up16 = TCONV(512, 256, t_kernel=3, t_stride=2, t_padding=1, outpadding=1)
            self.up8 = TCONV(256, 128, t_kernel=3, t_stride=2, t_padding=1, outpadding=1)
            self.up4 = TCONV(128, 64, t_kernel=3, t_stride=2, t_padding=1, outpadding=1)
            self.up2 = TCONV(64, 32, t_kernel=1, t_stride=1, t_padding=0, outpadding=0)
            self.refine1 = FCONV(32, self.in_channel, t_kernel=1, t_stride=1, t_padding=0)
        else:
            self.up32 = TCONV(2048, 512, t_kernel=3, t_stride=2, t_padding=1, outpadding=1)
            self.up16 = TCONV(1024, 256, t_kernel=3, t_stride=2, t_padding=1, outpadding=1)
            self.up8 = TCONV(512, 128, t_kernel=3, t_stride=2, t_padding=1, outpadding=1)
            if self.skip_connect == 1:
                self.up4 = TCONV(256, 64, t_kernel=3, t_stride=2, t_padding=1, outpadding=1)
                self.up2 = TCONV(128, 32, t_kernel=1, t_stride=1, t_padding=0, outpadding=0)
                self.refine1 = FCONV(32, self.in_channel, t_kernel=1, t_stride=1, t_padding=0)
            else:
                self.up4 = TCONV(128, 64, t_kernel=3, t_stride=2, t_padding=1, outpadding=1)
                self.up2 = TCONV(64, 32, t_kernel=1, t_stride=1, t_padding=0, outpadding=0)
                self.refine1 = FCONV(32, self.in_channel, t_kernel=1, t_stride=1, t_padding=0)

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

        self.classifier = nn.Linear(self.latent_dim32, self._num_classes)

    
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

    def _forward(self, input, label_en):
        # out_prev: cell-state
        # index 0: keep; index 1: down
        # stem = self.stem[self.arch_idx]


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

        enc2 = self.down1(input)
        enc4 = self.down2(enc2)
        enc8 = self.down4(enc4)
        out_prev = [[enc8, None]] # stem: one cell
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

        latent_mu = self.mean_layer32(out[2][0].view(-1,1024*self.last_size*self.last_size))
        latent_var = self.var_layer32(out[2][0].view(-1,1024*self.last_size*self.last_size))
        latent_var = F.softplus(latent_var) + 1e-8

        # print(latent_mu.shape)
        z_mu, y_mu = torch.split(latent_mu, [self.z_dim, self.latent_dim32], dim=1)
        z_var, y_var = torch.split(latent_var, [self.z_dim, self.latent_dim32], dim=1)
        # z_var = F.softplus(z_var) + 1e-8
        # y_var = F.softplus(y_var) + 1e-8

        y_latent = sample_gaussian(y_mu, y_var)
        latent = sample_gaussian(latent_mu, latent_var)
        # print(latent_var)
        predict = F.log_softmax(self.classifier(y_latent), dim=1)
        predict_test = F.log_softmax(self.classifier(y_mu), dim=1)
        yh = self.one_hot32(label_en)

        decoded = self.dec32(latent)
        decoded = decoded.view(-1, 1024, self.last_size, self.last_size)

        # print(decoded.shape)
        # print(out[2].shape)
        if self.skip_connect == 0:
            out32 = out[2][0]
            out16 = self.up32.decode(out32)
            out8 = self.up16.decode(out16)
            out4 = self.up8.decode(out8)
            out2 = self.up4.decode(out4)
        else:
            out32 = torch.cat((decoded, out[2][0]), dim=1)
            out16 = torch.cat((self.up32.decode(out32), out[1][0]), dim=1)
            out8 = torch.cat((self.up16.decode(out16), out[0][0]), dim=1)
            if self.skip_connect == 1:
                out4 = torch.cat((self.up8.decode(out8), enc4), dim=1)
                out2 = torch.cat((self.up4.decode(out4), enc2), dim=1)
            else:
                out4 = self.up8.decode(out8)
                out2 = self.up4.decode(out4)
        out1 = self.up2.decode(out2)
        reconstructed = self.refine1.final_decode(out1)

        return latent, latent_mu, latent_var, \
               predict, predict_test, yh, \
               reconstructed, [enc2, enc4, out[0][0], out[1][0], out[2][0]]
    
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


    def forward(self, input, target, target_de, pretrain=False, test=False):
        # print(target)
        re_loss = 0
        ce_loss = 0
        kl_loss = 0
        contras_loss = 0
        beta = next(self.beta)
        img_index = 1
        if pretrain is not True:
            # "random width": sampled by gambel softmax
            self.prun_mode = None
            for idx in range(len(self._arch_names)):
                self.arch_idx = idx
                # latent_mu, latent_var, yh, predict, reconstructed, up32, up16 = self._forward(input, target, target_de)
                latent, latent_mu, latent_var, \
                predict, predict_test, yh, \
                reconstructed, out = self._forward(input, target_de)
                rec = reconstruction_function(reconstructed, input)
                re_loss += rec
                # print(rec)
                z_latent_mu, y_latent_mu = torch.split(latent_mu, [self.z_dim, self.latent_dim32], dim=1)
                z_latent_var, y_latent_var = torch.split(latent_var, [self.z_dim, self.latent_dim32], dim=1)
                pm_z, pv_z = torch.zeros(z_latent_mu.shape).cuda(), torch.ones(z_latent_var.shape).cuda()
                pm, pv = torch.zeros(y_latent_mu.shape).cuda(), torch.ones(y_latent_var.shape).cuda()
                kl_latent = kl_normal(y_latent_mu, y_latent_var, pm, pv, yh)
                kl_z = kl_normal(z_latent_mu, z_latent_var, pm_z, pv_z, 0)
                kl_loss += beta * (kl_latent + self.beta_z * kl_z)

                ce = nllloss(predict, target)
                ce_loss += self.lamda * ce

                contras = self.contrastive_loss(input, latent_mu, out, target, reconstructed)
                contras_loss += contras
        if len(self._width_mult_list) > 1:
            self.prun_mode = "max"
            # latent_mu, latent_var, yh, predict, reconstructed, up32, up16 = self._forward(input, target, target_de)
            latent, latent_mu, latent_var, \
            predict, predict_test, yh, \
            reconstructed, out = self._forward(input, target_de)
            rec = reconstruction_function(reconstructed, input)
            re_loss += rec

            z_latent_mu, y_latent_mu = torch.split(latent_mu, [self.z_dim, self.latent_dim32], dim=1)
            z_latent_var, y_latent_var = torch.split(latent_var, [self.z_dim, self.latent_dim32], dim=1)
            pm_z, pv_z = torch.zeros(z_latent_mu.shape).cuda(), torch.ones(z_latent_var.shape).cuda()
            pm, pv = torch.zeros(y_latent_mu.shape).cuda(), torch.ones(y_latent_var.shape).cuda()
            kl_latent = kl_normal(y_latent_mu, y_latent_var, pm, pv, yh)
            kl_z = kl_normal(z_latent_mu, z_latent_var, pm_z, pv_z, 0)
            kl_loss += beta * (kl_latent + self.beta_z * kl_z)

            ce = nllloss(predict, target)
            ce_loss += self.lamda * ce

            contras = self.contrastive_loss(input, latent_mu, out, target, reconstructed)
            contras_loss += contras

            # if pretrain == True:
            #     self.prun_mode = "random"

        elif pretrain == True and len(self._width_mult_list) == 1:
            self.prun_mode = "max"
            latent, latent_mu, latent_var, \
            predict, predict_test, yh, \
            reconstructed, out = self._forward(input, target_de)
            rec = reconstruction_function(reconstructed, input)
            re_loss += rec

            z_latent_mu, y_latent_mu = torch.split(latent_mu, [self.z_dim, self.latent_dim32], dim=1)
            z_latent_var, y_latent_var = torch.split(latent_var, [self.z_dim, self.latent_dim32], dim=1)
            pm_z, pv_z = torch.zeros(z_latent_mu.shape).cuda(), torch.ones(z_latent_var.shape).cuda()
            pm, pv = torch.zeros(y_latent_mu.shape).cuda(), torch.ones(y_latent_var.shape).cuda()
            kl_latent = kl_normal(y_latent_mu, y_latent_var, pm, pv, yh)
            kl_z = kl_normal(z_latent_mu, z_latent_var, pm_z, pv_z, 0)
            kl_loss += beta * (kl_latent + self.beta_z * kl_z)

            ce = nllloss(predict, target)
            ce_loss += self.lamda * ce

            contras = self.contrastive_loss(input, latent_mu, out, target, reconstructed)
            contras_loss += contras
        self.prun_mode = "max"
        # _, _, _, predict, _, _, _ = self._forward(input, target, target_de)
        _, latent_mu, _, \
        predict, predict_test, _, \
        reconstructed, _ = self._forward(input, target_de)
        # print("ce_loss: {}".format(ce_loss.shape))
        # print("re_loss: {}".format(re_loss.shape))
        # print("kl_loss: {}".format(kl_loss.shape))

        # return ce_loss, re_loss, torch.mean(kl_loss), predict
        z_latent_mu, y_latent_mu = torch.split(latent_mu, [self.z_dim, self.latent_dim32], dim=1)
        return ce_loss, torch.mean(kl_loss), re_loss, contras_loss, predict, predict_test, y_latent_mu, reconstructed

    def test(self, input, target, target_de, pretrain=False):
        self.prun_mode = "max"
        # _, _, _, predict, _, _, _ = self._forward(input, target, target_de)
        _, latent_mu, _, \
        predict, predict_test, _, \
        reconstructed, _ = self._forward(input, target_de)
        # print("ce_loss: {}".format(ce_loss.shape))
        # print("re_loss: {}".format(re_loss.shape))
        # print("kl_loss: {}".format(kl_loss.shape))

        # return ce_loss, re_loss, torch.mean(kl_loss), predict
        z_latent_mu, y_latent_mu = torch.split(latent_mu, [self.z_dim, self.latent_dim32], dim=1)
        return predict_test, y_latent_mu, reconstructed

    def get_yh(self, y_de):
        yh = self.one_hot32(y_de)
        return yh

    def contrastive_loss(self, x, latent_mu, out, target, rec_x):
        """
        z : batchsize * 10
        """
        bs = x.size(0)
        ### get current yh for each class
        target_en = torch.eye(self._num_classes)
        class_yh = self.get_yh(target_en.cuda())  # 6*32
        yh_size = class_yh.size(1)

        neg_class_num = self._num_classes - 1
        y_neg = torch.zeros((bs, neg_class_num, yh_size)).cuda()
        for i in range(bs):
            y_sample = [idx for idx in range(self._num_classes) if idx != torch.argmax(target[i])]
            y_neg[i] = class_yh[y_sample]
        # zy_neg = torch.cat([z_neg, y_neg], dim=2).view(bs*neg_class_num, z.size(1)+yh_size)

        rec_x_neg = self.generate_cf(x, latent_mu, out, target, y_neg)
        # print(rec_x.shape)
        # print(rec_x_neg.shape)
        rec_x_all = torch.cat([rec_x.unsqueeze(1), rec_x_neg], dim=1)

        x_expand = x.unsqueeze(1).repeat_interleave(self._num_classes, dim=1)
        neg_dist = -((x_expand - rec_x_all) ** 2).mean((2, 3, 4)) * self.temperature  # N*(K+1)
        label = torch.zeros(bs).cuda().long()
        contrastive_loss_euclidean = nn.CrossEntropyLoss()(neg_dist, label)

        return contrastive_loss_euclidean

    def generate_cf(self, x, latent_mu, out, y_de, mean_y):
        """
        :param x:
        :param mean_y: list, the class-wise feature y
        """
        if mean_y.dim() == 2:
            class_num = mean_y.size(0)
        elif mean_y.dim() == 3:
            class_num = mean_y.size(1)
        bs = latent_mu.size(0)

        z_latent_mu, y_latent_mu = torch.split(latent_mu, [self.z_dim, self.latent_dim32], dim=1)

        z_latent_mu = z_latent_mu.unsqueeze(1).repeat_interleave(class_num, dim=1)
        if mean_y.dim() == 2:
            y_mu =mean_y.unsqueeze(0).repeat_interleave(bs, dim=0)
        elif mean_y.dim() == 3:
            y_mu = mean_y
        latent_zy = torch.cat([z_latent_mu, y_mu], dim=2).view(bs*class_num, latent_mu.size(1))

        # latent = ut.sample_gaussian(mu_latent, var_latent)

        # partially downwards
        decoded = self.dec32(latent_zy)
        decoded = decoded.view(-1, 1024, self.last_size, self.last_size)
        # print(decoded.shape)
        out32 = torch.cat((decoded, out[4].repeat_interleave(class_num, dim=0)), dim=1)
        out16 = torch.cat((self.up32.decode(out32), out[3].repeat_interleave(class_num, dim=0)), dim=1)
        out8 = torch.cat((self.up16.decode(out16), out[2].repeat_interleave(class_num, dim=0)), dim=1)
        out4 = torch.cat((self.up8.decode(out8), out[1].repeat_interleave(class_num, dim=0)), dim=1)
        out2 = torch.cat((self.up4.decode(out4), out[0].repeat_interleave(class_num, dim=0)), dim=1)
        out1 = self.up2.decode(out2)
        x_re = self.refine1.final_decode(out1)

        return x_re.view(bs, class_num, *x.size()[1:])

    def rec_loss_cf(self, feature_y_mean, val_loader, test_loader, args):
        rec_loss_cf_all = []
        class_num = feature_y_mean.size(0)
        for data_test, target_test in val_loader:
            target_test_en = torch.Tensor(target_test.shape[0], args.num_classes)
            target_test_en.zero_()
            target_test_en.scatter_(1, target_test.view(-1, 1), 1)  # one-hot encoding
            target_test_en = target_test_en.cuda()
            if args.cuda:
                data_test, target_test = data_test.cuda(), target_test.cuda()
            with torch.no_grad():
                data_test, target_test = Variable(data_test), Variable(target_test)


            _, latent_mu, _, _, _, _, _, outputs = self._forward(data_test, target_test_en)

            re_test = self.generate_cf(data_test, latent_mu, outputs, target_test_en, feature_y_mean)
            data_test_cf = data_test.unsqueeze(1).repeat_interleave(class_num, dim=1)
            rec_loss = (re_test - data_test_cf).pow(2).sum((2, 3, 4))
            rec_loss_cf = rec_loss.min(1)[0]
            rec_loss_cf_all.append(rec_loss_cf)

        for data_test, target_test in test_loader:
            target_test_en = torch.Tensor(target_test.shape[0], args.num_classes)
            target_test_en.zero_()
            # target_test_en.scatter_(1, target_test.view(-1, 1), 1)  # one-hot encoding
            target_test_en = target_test_en.cuda()
            if args.cuda:
                data_test, target_test = data_test.cuda(), target_test.cuda()
            with torch.no_grad():
                data_test, target_test = Variable(data_test), Variable(target_test)

            _, latent_mu, _, _, _, _, _, outputs = self._forward(data_test, target_test_en)

            re_test = self.generate_cf(data_test, latent_mu, outputs, target_test_en, feature_y_mean)
            data_test_cf = data_test.unsqueeze(1).repeat_interleave(class_num, dim=1)
            rec_loss = (re_test - data_test_cf).pow(2).sum((2, 3, 4))
            rec_loss_cf = rec_loss.min(1)[0]
            rec_loss_cf_all.append(rec_loss_cf)

        rec_loss_cf_all = torch.cat(rec_loss_cf_all, 0)
        return rec_loss_cf_all

    def rec_loss_cf_train(self, feature_y_mean, train_loader, args):
        rec_loss_cf_all = []
        class_num = feature_y_mean.size(0)
        for data_train, target_train in train_loader:
            target_train_en = torch.Tensor(target_train.shape[0], args.num_classes)
            target_train_en.zero_()
            target_train_en.scatter_(1, target_train.view(-1, 1), 1)  # one-hot encoding
            target_train_en = target_train_en.cuda()
            if args.cuda:
                data_train, target_train = data_train.cuda(), target_train.cuda()
            with torch.no_grad():
                data_train, target_train = Variable(data_train), Variable(target_train)

            _, latent_mu, _, _, _, _, _, outputs = self._forward(data_train, target_train_en)

            re_train = self.generate_cf(data_train, latent_mu, outputs, target_train_en, feature_y_mean)
            data_train_cf = data_train.unsqueeze(1).repeat_interleave(class_num, dim=1)
            rec_loss = (re_train - data_train_cf).pow(2).sum((2, 3, 4))
            rec_loss_cf = rec_loss.min(1)[0]
            rec_loss_cf_all.append(rec_loss_cf)

        rec_loss_cf_all = torch.cat(rec_loss_cf_all, 0)
        return rec_loss_cf_all

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
