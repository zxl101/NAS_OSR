import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from operations import *
from genotypes import PRIMITIVES
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from PIL import Image
from pdb import set_trace as bp

BatchNorm2d = nn.BatchNorm2d

def kl_normal(qm, qv, pm, pv, yh):
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm - yh).pow(2) / pv - 1)
    kl = element_wise.sum(-1)
    return torch.mean(kl)

def sample_gaussian(m, v):
    sample = torch.randn(m.shape).to(torch.device("cuda"))
    m = m.cuda()
    v = v.cuda()
    z = m + (v ** 0.5) * sample
    return z


def softmax(x):
    return np.exp(x) / (np.exp(x).sum() + np.spacing(1))

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

def path2downs(path):
    '''
    0 same 1 down
    '''
    downs = []
    prev = path[0]
    for node in path[1:]:
        assert (node - prev) in [0, 1]
        if node > prev:
            downs.append(1)
        else:
            downs.append(0)
        prev = node
    downs.append(0)
    return downs


def downs2path(downs):
    path = [0]
    for down in downs[:-1]:
        if down == 0:
            path.append(path[-1])
        elif down == 1:
            path.append(path[-1] + 1)
    return path


def alphas2ops_path_width(alphas, path, widths):
    '''
    alphas: [alphas0, ..., alphas3]
    '''
    assert len(path) == len(widths) + 1, "len(path) %d, len(widths) %d" % (len(path), len(widths))
    ops = []
    path_compact = []
    widths_compact = []
    pos2alpha_skips = []  # (pos, alpha of skip) to be prunned
    min_len = int(np.round(len(path) / 3.)) + path[-1] * 2
    # keep record of position(s) of skip_connect
    for i in range(len(path)):
        scale = path[i]
        op = alphas[scale][i - scale].argmax()
        if op == 0 and (i == len(path) - 1 or path[i] == path[i + 1]):
            # alpha not softmax yet
            pos2alpha_skips.append((i, F.softmax(alphas[scale][i - scale], dim=-1)[0]))

    pos_skips = [pos for pos, alpha in pos2alpha_skips]
    pos_downs = [pos for pos in range(len(path) - 1) if path[pos] < path[pos + 1]]
    if len(pos_downs) > 0:
        pos_downs.append(len(path))
        for i in range(len(pos_downs) - 1):
            # cannot be all skip_connect between each downsample-pair
            # including the last down to the path-end
            pos1 = pos_downs[i];
            pos2 = pos_downs[i + 1]
            if pos1 + 1 in pos_skips and pos2 - 1 in pos_skips and pos_skips.index(pos2 - 1) - pos_skips.index(
                    pos1 + 1) == (pos2 - 1) - (pos1 + 1):
                min_skip = [1, -1]  # score, pos
                for j in range(pos1 + 1, pos2):
                    scale = path[j]
                    score = F.softmax(alphas[scale][j - scale], dim=-1)[0]
                    if score <= min_skip[0]:
                        min_skip = [score, j]
                alphas[path[min_skip[1]]][min_skip[1] - path[min_skip[1]]][0] = -float('inf')

    if len(pos2alpha_skips) > len(path) - min_len:
        pos2alpha_skips = sorted(pos2alpha_skips, key=lambda x: x[1], reverse=True)[:len(path) - min_len]
    pos_skips = [pos for pos, alpha in pos2alpha_skips]
    for i in range(len(path)):
        scale = path[i]
        if i < len(widths): width = widths[i]
        op = alphas[scale][i - scale].argmax()
        if op == 0:
            if i in pos_skips:
                # remove the last width if the last layer (skip_connect) is to be prunned
                if i == len(path) - 1: widths_compact = widths_compact[:-1]
                continue
            else:
                alphas[scale][i - scale][0] = -float('inf')
                op = alphas[scale][i - scale].argmax()
        path_compact.append(scale)
        if i < len(widths): widths_compact.append(width)
        ops.append(op)
    return ops, path_compact, widths_compact


def betas2path(betas, last, layers):

    downs = [0] * layers
    # betas1 is of length layers-2; beta2: layers-3; beta3: layers-4
    if last == 1:
        # print(betas[1].shape)
        if betas[1].shape[0] == 1:
            downs[0] = 1
        else:
            down_idx = np.argmax([beta[0] for beta in betas[1][1:-1].cpu().numpy()]) + 1
            downs[down_idx] = 1
    elif last == 2:
        if betas[2].shape[0] <= 1:
            downs[0] = 1
            downs[1] = 1
        else:
            max_prob = 0;
            max_ij = (0, 1)
            for j in range(layers - 4):
                for i in range(1, j - 1):
                    prob = betas[1][i][0] * betas[2][j][0]
                    if prob > max_prob:
                        max_ij = (i, j)
                        max_prob = prob
            downs[max_ij[0] + 1] = 1;
            downs[max_ij[1] + 2] = 1
    path = downs2path(downs)
    # print(path)
    assert path[-1] == last
    return path


def path2widths(path, ratios, width_mult_list):
    widths = []
    for layer in range(1, len(path)):
        scale = path[layer]
        if scale == 0:
            widths.append(width_mult_list[ratios[scale][layer - 1].argmax()])
        else:
            widths.append(width_mult_list[ratios[scale][layer - scale].argmax()])
    return widths


def network_metas(alphas, betas, ratios, width_mult_list, layers, last):
    betas[1] = F.softmax(betas[1], dim=-1)
    betas[2] = F.softmax(betas[2], dim=-1)
    path = betas2path(betas, last, layers)
    widths = path2widths(path, ratios, width_mult_list)
    ops, path, widths = alphas2ops_path_width(alphas, path, widths)
    assert len(ops) == len(path) and len(path) == len(widths) + 1, "op %d, path %d, width%d" % (
    len(ops), len(path), len(widths))
    downs = path2downs(path)  # 0 same 1 down
    return ops, path, downs, widths


class MixedOp(nn.Module):
    def __init__(self, C_in, C_out, op_idx, stride=1):
        super(MixedOp, self).__init__()
        self._op = OPS[PRIMITIVES[op_idx]](C_in, C_out, stride, slimmable=False, width_mult_list=[1.])

    def forward(self, x):
        return self._op(x)

    def forward_latency(self, size):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        latency, size_out = self._op.forward_latency(size)
        return latency, size_out


class Cell(nn.Module):
    def __init__(self, op_idx, C_in, C_out, down, latent_dim=None, flat_dim=None):
        super(Cell, self).__init__()
        self._C_in = C_in
        self._C_out = C_out
        self._down = down
        self.latent_dim = latent_dim
        self.flat_dim = flat_dim

        if self._down:
            self._op = MixedOp(C_in, C_out, op_idx, stride=2)
        else:
            self._op = MixedOp(C_in, C_out, op_idx)

        # self.mean_layer = nn.Sequential(
        #     nn.Linear(self.flat_dim * self.flat_dim * self._C_out, self.latent_dim)
        # )
        # self.var_layer = nn.Sequential(
        #     nn.Linear(self.flat_dim * self.flat_dim * self._C_out, self.latent_dim)
        # )

    def forward(self, input):
        out = self._op(input)
        # out_flat = out.view(-1, self._C_out * self.flat_dim * self.flat_dim)
        # mu, var = self.mean_layer(out_flat), self.var_layer(out_flat)
        # var = F.softplus(var) + 1e-8
        return out

    def forward_latency(self, size):
        # ratios: (in, out, down)
        out = self._op.forward_latency(size)
        return out


class Network_Multi_Path_Infer(nn.Module):
    def __init__(self, alphas, betas, ratios, num_classes=6, in_channel=3, layers=6,
                 criterion=nn.CrossEntropyLoss(ignore_index=-1),
                 Fch=16, width_mult_list=[1., ], stem_head_width=(1., 1.), latent_dim32=32*1,
                 latent_dim64=64*1, latent_dim128=128*1, temperature=1, z_dim=10, img_size=64, down_scale_last=4,
                 skip_connect=1, wcontras=1):
        super(Network_Multi_Path_Infer, self).__init__()
        self._num_classes = num_classes
        assert layers >= 2
        self._layers = layers
        self._criterion = criterion
        self._Fch = Fch
        self.in_channel = in_channel
        self.latent_dim32 = latent_dim32
        self.latent_dim64 = latent_dim64
        self.latent_dim128 = latent_dim128
        self.temperature = temperature
        self.z_dim = z_dim
        self.img_size = img_size
        self.down_scale_last = down_scale_last
        self.last_size = self.img_size // (2 ** self.down_scale_last)
        self.wcontras = wcontras

        self.skip_connect = skip_connect

        if ratios[0].size(1) == 1:
            self._width_mult_list = [1., ]
        else:
            self._width_mult_list = width_mult_list
        self._stem_head_width = stem_head_width
        self.latency = 0

        self.down1 = ConvNorm(self.in_channel, self.num_filters(4, 1), kernel_size=1, stride=1, padding=0, bias=False,
                              groups=1, slimmable=False)
        self.down2 = BasicResidual2x(self.num_filters(4, 1), self.num_filters(8, 1), kernel_size=3, stride=2, groups=1,
                                     slimmable=False)
        self.down4 = BasicResidual2x(self.num_filters(8, 1), self.num_filters(16, 1), kernel_size=3, stride=2, groups=1,
                                     slimmable=False)

        self.ops0, self.path0, self.downs0, self.widths0 = network_metas(alphas, betas, ratios, self._width_mult_list,
                                                                         layers, 0)
        self.ops1, self.path1, self.downs1, self.widths1 = network_metas(alphas, betas, ratios, self._width_mult_list,
                                                                         layers, 1)
        self.ops2, self.path2, self.downs2, self.widths2 = network_metas(alphas, betas, ratios, self._width_mult_list,
                                                                         layers, 2)

    def num_filters(self, scale, width=1.0):
        return int(np.round(scale * self._Fch * width))

    def build_structure(self, lasts):
        self._branch = len(lasts)
        self.lasts = lasts
        self.ops = [getattr(self, "ops%d" % last) for last in lasts]
        self.paths = [getattr(self, "path%d" % last) for last in lasts]
        self.downs = [getattr(self, "downs%d" % last) for last in lasts]
        self.widths = [getattr(self, "widths%d" % last) for last in lasts]
        self.branch_groups, self.cells = self.get_branch_groups_cells(self.ops, self.paths, self.downs, self.widths,
                                                                      self.lasts)
        self.build_arm_ffm_head()

    def build_arm_ffm_head(self):


        self.dec32 = nn.Linear(self.latent_dim32 + self.z_dim, 1024 * self.last_size * self.last_size)
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

        self.classifier = nn.Linear(self.latent_dim32, self._num_classes)

        self.one_hot32 = nn.Linear(self._num_classes, self.latent_dim32)

        self.mean_layer32 = nn.Sequential(
            nn.Linear(int(1024 * self.last_size * self.last_size), self.latent_dim32 + self.z_dim)
        )
        self.var_layer32 = nn.Sequential(
            nn.Linear(int(1024 * self.last_size * self.last_size), self.latent_dim32 + self.z_dim)
        )


    def get_branch_groups_cells(self, ops, paths, downs, widths, lasts):
        num_branch = len(ops)
        layers = max([len(path) for path in paths])
        groups_all = []
        self.ch_16 = 0;
        self.ch_8_2 = 0;
        self.ch_8_1 = 0
        cells = nn.ModuleDict()  # layer-branch: op
        branch_connections = np.ones(
            (num_branch, num_branch))  # maintain connections of heads of branches of different scales
        # all but the last layer
        # we determine branch-merging by comparing their next layer: if next-layer differs, then the "down" of current layer must differ
        for l in range(layers):
            connections = np.ones((num_branch, num_branch))  # if branch i/j share same scale & op in this layer
            for i in range(num_branch):
                for j in range(i + 1, num_branch):
                    # we also add constraint on ops[i][l] != ops[j][l] since some skip-connect may already be shrinked/compacted => layers of branches may no longer aligned in terms of alphas
                    # last layer won't merge
                    if len(paths[i]) <= l + 1 or len(paths[j]) <= l + 1 or paths[i][l + 1] != paths[j][l + 1] or ops[i][
                        l] != ops[j][l] or widths[i][l] != widths[j][l]:
                        connections[i, j] = connections[j, i] = 0
            branch_connections *= connections
            branch_groups = []
            # build branch_group for processing
            for branch in range(num_branch):
                # also accept if this is the last layer of branch (len(paths[branch]) == l+1)
                if len(paths[branch]) < l + 1: continue
                inserted = False
                for group in branch_groups:
                    if branch_connections[group[0], branch] == 1:
                        group.append(branch)
                        inserted = True
                        continue
                if not inserted:
                    branch_groups.append([branch])
            for group in branch_groups:
                # branch in the same group must share the same op/scale/down/width
                if len(group) >= 2: assert ops[group[0]][l] == ops[group[1]][l] and paths[group[0]][l + 1] == \
                                           paths[group[1]][l + 1] and downs[group[0]][l] == downs[group[1]][l] and \
                                           widths[group[0]][l] == widths[group[1]][l]
                if len(group) == 3: assert ops[group[1]][l] == ops[group[2]][l] and paths[group[1]][l + 1] == \
                                           paths[group[2]][l + 1] and downs[group[1]][l] == downs[group[2]][l] and \
                                           widths[group[1]][l] == widths[group[2]][l]
                op = ops[group[0]][l]
                scale = 2 ** (paths[group[0]][l] + 3) *2
                down = downs[group[0]][l]
                if l < len(paths[group[0]]) - 1: assert down == paths[group[0]][l + 1] - paths[group[0]][l]
                assert down in [0, 1]
                if l == 0:
                    cell = Cell(op, self.num_filters(scale, self._stem_head_width[0]),
                                self.num_filters(scale * (down + 1), widths[group[0]][l]), down)
                elif l == len(paths[group[0]]) - 1:
                    # last cell for this branch
                    assert down == 0
                    cell = Cell(op, self.num_filters(scale, widths[group[0]][l - 1]),
                                self.num_filters(scale, self._stem_head_width[1]), down)
                else:
                    cell = Cell(op, self.num_filters(scale, widths[group[0]][l - 1]),
                                self.num_filters(scale * (down + 1), widths[group[0]][l]), down)
                # For Feature Fusion: keep record of dynamic #channel of last 1/16 and 1/8 of "1/32 branch"; last 1/8 of "1/16 branch"
                if 2 in self.lasts and self.lasts.index(2) in group and down and scale == 16: self.ch_16 = cell._C_in
                if 2 in self.lasts and self.lasts.index(2) in group and down and scale == 8: self.ch_8_2 = cell._C_in
                if 1 in self.lasts and self.lasts.index(1) in group and down and scale == 8: self.ch_8_1 = cell._C_in
                for branch in group:
                    cells[str(l) + "-" + str(branch)] = cell
            groups_all.append(branch_groups)
        return groups_all, cells

    def agg_ffm(self, outputs):
        global check_vec
        outputs32 = outputs[4]
        outputs16 = outputs[3]
        outputs8 = outputs[2]
        outputs4 = outputs[1]
        outputs2 = outputs[0]
        latent_mu = self.mean_layer32(outputs32.view(-1, 1024 * self.last_size * self.last_size))
        latent_var = self.var_layer32(outputs32.view(-1, 1024 * self.last_size * self.last_size))
        latent_var = F.softplus(latent_var) + 1e-8


        z_mu, y_mu = torch.split(latent_mu, [self.z_dim, self.latent_dim32], dim=1)
        z_var, y_var = torch.split(latent_var, [self.z_dim, self.latent_dim32], dim=1)
        # z_var = F.softplus(z_var) + 1e-8
        # y_var = F.softplus(y_var) + 1e-8

        y_latent = sample_gaussian(y_mu, y_var)
        latent = sample_gaussian(latent_mu, latent_var)
        predict = F.log_softmax(self.classifier(y_latent), dim=1)
        predict_test = F.log_softmax(self.classifier(y_mu), dim=1)

        decoded = self.dec32(latent_mu)
        decoded = decoded.view(-1, 1024, self.last_size, self.last_size)


        if self.skip_connect == 0:
            out32 = outputs32
            out16 = self.up32.decode(out32)
            out8 = self.up16.decode(out16)
            out4 = self.up8.decode(out8)
            out2 = self.up4.decode(out4)
        else:
            out32 = torch.cat((decoded, outputs32), dim=1)
            out16 = torch.cat((self.up32.decode(out32), outputs16), dim=1)
            out8 = torch.cat((self.up16.decode(out16), outputs8), dim=1)
            if self.skip_connect == 1:
                out4 = torch.cat((self.up8.decode(out8), outputs4), dim=1)
                out2 = torch.cat((self.up4.decode(out4), outputs2), dim=1)
            else:
                out4 = self.up8.decode(out8)
                out2 = self.up4.decode(out4)
        out1 = self.up2.decode(out2)
        reconstructed = self.refine1.final_decode(out1)

        # print("The latent vector is")
        # print(torch.Tensor.cpu(reconstructed[0]).detach().numpy())
        check_vec = reconstructed[0]
        return latent, latent_mu, latent_var, \
               predict, predict_test,\
               reconstructed, outputs


    def forward(self, input):
        _, _, H, W = input.size()
        enc2 = self.down1(input)
        enc4 = self.down2(enc2)
        enc8 = self.down4(enc4)
        outputs = [enc8] * self._branch
        # print(enc8.shape)
        for layer in range(len(self.branch_groups)):
            for group in self.branch_groups[layer]:
                output = self.cells[str(layer) + "-" + str(group[0])](outputs[group[0]])
                scale = int(H // output.size(2))
                for branch in group:
                    outputs[branch] = output
                    if scale == 4:
                        outputs8 = output
                    elif scale == 8:
                        outputs16 = output
                    elif scale == 16:
                        outputs32 = output
        # print(enc8)
        # print(outputs32)
        latent, latent_mu, latent_var, \
        predict, predict_test, \
        reconstructed, outputs = self.agg_ffm([enc2, enc4, outputs8, outputs16, outputs32])
        # print("The latent vector of the reconstructed image is")
        # print(torch.Tensor.cpu(latent_mu[0][10:]).detach().numpy())
        # print("The reconstructed image is")
        # print(torch.Tensor.cpu(reconstructed[0]).detach().numpy())
        return latent, latent_mu, latent_var, predict, predict_test, reconstructed, outputs

    def get_yh(self, y_de):
        yh = self.one_hot32(y_de)
        return yh

    def contrastive_loss(self, x, latent_mu, out, target, rec_x, img_index=None, save_folder="cf_img"):
        """
        z : batchsize * 10
        """
        bs = x.size(0)
        ### get current yh for each class
        target_en = torch.eye(self._num_classes)
        class_yh = self.get_yh(target_en.cuda())  # 6*32

        yh_size = class_yh.size(1)
        yh = torch.zeros(bs, yh_size).cuda()
        y_all = torch.zeros(bs, self._num_classes, yh_size).cuda()
        for i in range(bs):
            y_all[i] = class_yh
            # print(target[i])
            # if target[i] < self._num_classes:
            yh[i] = class_yh[target[i]]

        rec_x_all = self.generate_cf(x, latent_mu, out, y_all)

        x_expand = x.unsqueeze(1).repeat_interleave(self._num_classes, dim=1)
        neg_dist = -((x_expand - rec_x_all) ** 2).mean((2, 3, 4)) * self.temperature  # N*(K+1)
        # neg_dist[:, target] = neg_dist[:, target] - 0.005 * self.temperature
        contrastive_loss_euclidean = nn.CrossEntropyLoss()(neg_dist, target)

        if img_index != None:
            if img_index < 10:
                for i in range(rec_x_all.shape[1]):
                    neg = torch.Tensor.cpu(y_all[0]).detach().numpy()
                    c_yh = torch.Tensor.cpu(class_yh).detach().numpy()
                    dist = torch.Tensor.cpu(neg_dist).detach().numpy()
                    with open('cf_img/train_yh.txt', 'ab') as f:
                        np.savetxt(f, c_yh[:, 0], fmt='%f', delimiter=' ', newline='\r')
                        f.write(b'\n')
                        np.savetxt(f, neg[:, 0], fmt='%f', delimiter=' ', newline='\r')
                        f.write(b'\n')
                    with open('cf_img/train_cf_re_diff.txt', 'ab') as f:
                        # np.savetxt(f, c_yh, fmt='%f', delimiter=' ', newline='\r')
                        np.savetxt(f, dist, fmt='%f', delimiter=' ', newline='\r')
                        f.write(b'\n')

                    temp = rec_x_all[0][i]
                    temp = torch.Tensor.cpu(temp).detach().numpy()
                    temp = temp.transpose(1, 2, 0)
                    temp = temp * (0.2023, 0.1994, 0.2010) + (0.4914, 0.4822, 0.4465)
                    # temp = temp * 0.3081 + 0.1307
                    # temp = np.reshape(temp, (32, 32))
                    temp = temp * 255
                    temp = temp.astype(np.uint8)
                    img = Image.fromarray(temp)
                    img.save(os.path.join(save_folder, "{}_{}.jpeg".format(img_index, range(self._num_classes)[i])))
        # print(yh)
        return contrastive_loss_euclidean, yh

    def generate_cf(self, x, latent_mu, out, mean_y, img_index=None, save_folder="cf_img"):
        """
        :param x:
        :param mean_y: list, the class-wise feature y
        """
        global check_vec
        if mean_y.dim() == 2:
            class_num = mean_y.size(0)
        elif mean_y.dim() == 3:
            class_num = mean_y.size(1)
        bs = latent_mu.size(0)

        z_latent_mu, y_latent_mu = torch.split(latent_mu, [self.z_dim, self.latent_dim32], dim=1)

        z_latent_mu = z_latent_mu.unsqueeze(1).repeat_interleave(class_num, dim=1)
        if mean_y.dim() == 2:
            y_mu = mean_y.unsqueeze(0).repeat_interleave(bs, dim=0)
        elif mean_y.dim() == 3:
            y_mu = mean_y
        latent_zy = torch.cat([z_latent_mu, y_mu], dim=2).view(bs*class_num, latent_mu.size(1))
        # for i in range(6):
        #     print("The vector of the {} image is".format(i))
        #     print(torch.Tensor.cpu(latent_zy[i][10:]).detach().numpy())

        decoded = self.dec32(latent_zy)
        decoded = decoded.view(-1, 1024, self.last_size, self.last_size)
        if self.skip_connect == 0:
            out32 = decoded
            out16 = self.up32.decode(out32)
            out8 = self.up16.decode(out16)
            out4 = self.up8.decode(out8)
            out2 = self.up4.decode(out4)
        else:
            out32 = torch.cat((decoded, out[4].repeat_interleave(class_num, dim=0)), dim=1)
            out16 = torch.cat((self.up32.decode(out32), out[3].repeat_interleave(class_num, dim=0)), dim=1)
            out8 = torch.cat((self.up16.decode(out16), out[2].repeat_interleave(class_num, dim=0)), dim=1)
            if self.skip_connect == 1:
                out4 = torch.cat((self.up8.decode(out8), out[1].repeat_interleave(class_num, dim=0)), dim=1)
                out2 = torch.cat((self.up4.decode(out4), out[0].repeat_interleave(class_num, dim=0)), dim=1)
            else:
                out4 = self.up8.decode(out8)
                out2 = self.up4.decode(out4)
        out1 = self.up2.decode(out2)
        x_re = self.refine1.final_decode(out1)
        # for i in range(6):
        #     # print("The {} image is".format(i))
        #     # print(torch.Tensor.cpu(x_re[i]).detach().numpy())
        #     print("The latent vector of {} image is".format(i))
        #     print(torch.Tensor.cpu(x_re[i]).detach().numpy())
        # for i in range(6):
        #     diff = torch.mean((check_vec-x_re[i]).pow(2))
        #     print("The difference between {} image is".format(i))
        #     print(torch.Tensor.cpu(diff).detach().numpy())
        rec_x_all = x_re.view(bs, class_num, *x.size()[1:])
        if img_index != None:
            if img_index < 50:
                for i in range(rec_x_all.shape[1]):
                    temp = rec_x_all[0][i]
                    temp = torch.Tensor.cpu(temp).detach().numpy()
                    temp = temp.transpose(1, 2, 0)
                    temp = temp * (0.2023, 0.1994, 0.2010) + (0.4914, 0.4822, 0.4465)
                    # temp = temp * 0.3081 + 0.1307
                    # temp = np.reshape(temp, (32, 32))
                    temp = temp * 255
                    temp = temp.astype(np.uint8)
                    img = Image.fromarray(temp)
                    img.save(os.path.join(save_folder, "{}_{}.jpeg".format(img_index, range(self._num_classes)[i])))
        return rec_x_all

    def rec_loss_cf(self, feature_y_mean, val_loader, test_loader, args):
        rec_loss_cf_all = []
        class_num = feature_y_mean.size(0)

        for data_test, target_test in val_loader:
            if args.cuda:
                data_test, target_test = data_test.cuda(), target_test.cuda()
            with torch.no_grad():
                data_test, target_test = Variable(data_test), Variable(target_test)

            _, latent_mu, _, _, _, _, outputs = self.forward(data_test)

            re_test = self.generate_cf(data_test, latent_mu, outputs, feature_y_mean)
            data_test_cf = data_test.unsqueeze(1).repeat_interleave(class_num, dim=1)
            rec_loss = (re_test - data_test_cf).pow(2).sum((2, 3, 4))
            rec_loss_cf = rec_loss.min(1)[0]
            rec_loss_cf_all.append(rec_loss_cf)

        args.img_index = 0
        for data_test, target_test in test_loader:

            if args.cuda:
                data_test, target_test = data_test.cuda(), target_test.cuda()
            with torch.no_grad():
                data_test, target_test = Variable(data_test), Variable(target_test)

            _, latent_mu, _, _, _, reconstructed, outputs = self.forward(data_test)

            re_test = self.generate_cf(data_test, latent_mu, outputs, feature_y_mean, args.img_index, "unknown_cf")
            data_test_cf = data_test.unsqueeze(1).repeat_interleave(class_num, dim=1)
            rec_loss = (re_test - data_test_cf).pow(2).sum((2, 3, 4))
            rec_loss_cf = rec_loss.min(1)[0]
            rec_loss_cf_all.append(rec_loss_cf)

            if args.img_index < 50:
                re = torch.Tensor.cpu(reconstructed[0]).detach().numpy()
                ori = torch.Tensor.cpu(data_test[0]).detach().numpy()
                temp = re
                temp = temp.transpose(1, 2, 0)
                temp = temp * (0.2023, 0.1994, 0.2010) + (0.4914, 0.4822, 0.4465)
                # temp = temp * 0.3081 + 0.1307
                temp = temp * 255
                temp = temp.astype(np.uint8)
                img = Image.fromarray(temp)
                img.save(os.path.join("unknown_cf", "{}_re.jpeg".format(args.img_index)))

                ori = ori
                ori = ori.transpose(1, 2, 0)
                ori = ori * (0.2023, 0.1994, 0.2010) + (0.4914, 0.4822, 0.4465)
                # ori = ori * 0.3081 + 0.1307
                ori = ori * 255
                ori = ori.astype(np.uint8)
                ori = Image.fromarray(ori)
                ori.save(os.path.join("unknown_cf", "{}_ori.jpeg".format(args.img_index)))
                args.img_index += 1
        rec_loss_cf_all = torch.cat(rec_loss_cf_all, 0)
        return rec_loss_cf_all

    def rec_loss_cf_train(self, feature_y_mean, train_loader, args):
        rec_loss_cf_all = []
        class_num = feature_y_mean.size(0)
        for data_train, target_train in train_loader:
            if args.cuda:
                data_train, target_train = data_train.cuda(), target_train.cuda()
            with torch.no_grad():
                data_train, target_train = Variable(data_train), Variable(target_train)

            _, latent_mu, _, _, _, _, outputs = self.forward(data_train)

            re_train = self.generate_cf(data_train, latent_mu, outputs, feature_y_mean)
            data_train_cf = data_train.unsqueeze(1).repeat_interleave(class_num, dim=1)
            rec_loss = (re_train - data_train_cf).pow(2).sum((2, 3, 4))
            rec_loss_cf = rec_loss.min(1)[0]
            rec_loss_cf_all.append(rec_loss_cf)

        rec_loss_cf_all = torch.cat(rec_loss_cf_all, 0)
        return rec_loss_cf_all
