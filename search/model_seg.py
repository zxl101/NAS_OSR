import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from operations import *
from genotypes import PRIMITIVES
from pdb import set_trace as bp

# from seg_oprs import FeatureFusion, Head

BatchNorm2d = nn.BatchNorm2d


def sample_gaussian(m, v):
    sample = torch.randn(m.shape).to(torch.device("cuda"))
    # sample = torch.randn(m.shape)
    m = m.cuda()
    v = v.cuda()
    z = m + (v ** 0.5) * sample
    return z


def softmax(x):
    return np.exp(x) / (np.exp(x).sum() + np.spacing(1))

class TCONV(nn.Module):
    def __init__(self, in_size, unflat_dim, t_in_ch, t_out_ch, t_kernel, t_padding, t_stride, out_dim, t_latent_dim):
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

        self.fc = nn.Linear(in_size, t_in_ch * unflat_dim * unflat_dim)
        self.net = nn.Sequential(
            nn.PReLU(),
            nn.ConvTranspose2d(t_in_ch, t_out_ch, kernel_size=t_kernel, padding=t_padding, stride=t_stride),  # (w-k+2p)/s+1
            nn.BatchNorm2d(t_out_ch),
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
        h = self.net(x)
        h_flat = h.view(-1, self.t_out_ch * self.out_dim * self.out_dim)
        mu, var = self.mean_layer(h_flat), self.var_layer(h_flat)
        var = F.softplus(var) + 1e-8
        # mu, var = ut.gaussian_parameters(h, dim=1)
        return h, mu, var

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
            nn.PReLU(),
            nn.ConvTranspose2d(t_in_ch, t_out_ch, kernel_size=t_kernel, padding=t_padding, stride=t_stride),  # (w-k+2p)/s+1
            #nn.Sigmoid()
            # nn.Tanh()
        )

    def final_decode(self,x):
        x = self.fc_final(x)
        x = x.view(-1, self.t_in_ch, self.unflat_dim, self.unflat_dim)
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
    assert len(path_compact) >= min_len
    return ops, path_compact, widths_compact


def betas2path(betas, last, layers):
    downs = [0] * layers
    # betas1 is of length layers-2; beta2: layers-3; beta3: layers-4
    if last == 1:
        down_idx = np.argmax([beta[0] for beta in betas[1][1:-1].cpu().numpy()]) + 1
        downs[down_idx] = 1
    elif last == 2:
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
    def __init__(self, alphas, betas, ratios, num_classes=10, in_channel=3, layers=9,
                 criterion=nn.CrossEntropyLoss(ignore_index=-1),
                 Fch=12, width_mult_list=[1., ], stem_head_width=(1., 1.), latent_dim32=32,
                 latent_dim64=64, latent_dim128=128):
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
        if ratios[0].size(1) == 1:
            self._width_mult_list = [1., ]
        else:
            self._width_mult_list = width_mult_list
        self._stem_head_width = stem_head_width
        self.latency = 0

        self.stem = nn.Sequential(
            ConvNorm(self.in_channel, self.num_filters(2, stem_head_width[0]) * 2, kernel_size=3, stride=2, padding=1,
                     bias=False, groups=1, slimmable=False),
            BasicResidual2x(self.num_filters(2, stem_head_width[0]) * 2, self.num_filters(4, stem_head_width[0]) * 2,
                            kernel_size=3, stride=2, groups=1, slimmable=False),
            BasicResidual2x(self.num_filters(4, stem_head_width[0]) * 2, self.num_filters(8, stem_head_width[0]),
                            kernel_size=3, stride=2, groups=1, slimmable=False)
        )

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

        # self.refine32 = nn.ModuleList([
        #         ConvNorm(self.num_filters(32, self._stem_head_width[1]), self.num_filters(16, self._stem_head_width[1]),
        #                  kernel_size=1, bias=False,
        #                  groups=1, slimmable=False),
        #         ConvNorm(self.num_filters(32, self._stem_head_width[1]), self.num_filters(16, self._stem_head_width[1]),
        #                  kernel_size=3, padding=1,
        #                  bias=False, groups=1, slimmable=False),
        #         ConvNorm(self.num_filters(16, self._stem_head_width[1]), self.num_filters(8, self._stem_head_width[1]),
        #                  kernel_size=1, bias=False,
        #                  groups=1, slimmable=False),
        #         ConvNorm(self.num_filters(16, self._stem_head_width[1]), self.num_filters(8, self._stem_head_width[1]),
        #                  kernel_size=3, padding=1,
        #                  bias=False, groups=1, slimmable=False)])
        # self.refine16 = nn.ModuleList([
        #         ConvNorm(self.num_filters(16, self._stem_head_width[1]), self.num_filters(8, self._stem_head_width[1]),
        #                  kernel_size=1, bias=False,
        #                  groups=1, slimmable=False),
        #         ConvNorm(self.num_filters(16, self._stem_head_width[1]), self.num_filters(8, self._stem_head_width[1]),
        #                  kernel_size=3, padding=1,
        #                  bias=False, groups=1, slimmable=False)])
        # self.refine8 = nn.ModuleList([
        #     ConvNorm(self.num_filters(8, self._stem_head_width[1]), self.num_filters(4, self._stem_head_width[1]),
        #              kernel_size=3,
        #              padding=1, bias=False, groups=1, slimmable=False)])
        # self.refine4 = nn.ModuleList([
        #     ConvNorm(self.num_filters(4, self._stem_head_width[1]), self.num_filters(2, self._stem_head_width[1]),
        #              kernel_size=3,
        #              padding=1, bias=False, groups=1, slimmable=False)])
        # self.refine2 = nn.ModuleList([
        #     ConvNorm(self.num_filters(2, self._stem_head_width[1]), self.num_filters(1, self._stem_head_width[1]),
        #              kernel_size=3,
        #              padding=1, bias=False, groups=1, slimmable=False)])
        # self.refine1 = nn.ModuleList([
        #     ConvNorm(self.num_filters(1, self._stem_head_width[1]), self.num_filters(1, self._stem_head_width[1]),
        #              kernel_size=3,
        #              padding=1, bias=False, groups=1, slimmable=False)])
        # self.reconstruct = nn.ModuleList([
        #     ConvNorm(self.num_filters(1, self._stem_head_width[1]), 3, kernel_size=3,
        #              padding=1, bias=False, groups=1, slimmable=False)])

        self.fc1 = nn.Linear(30, 30)
        self.fc2 = nn.Linear(30, 15)
        self.fc3 = nn.Linear(15,10)

        self.TCONV5_2 = TCONV(32, 2, 512, 512, 2,
                              0, 2, 4, 64)
        self.TCONV5_1 = TCONV(32, 4, 512, 512, 1,
                              0, 1, 4, 64)

        self.TCONV4_2 = TCONV(64, 4, 512, 512, 2,
                              0, 2, 8, 128)
        self.TCONV4_1 = TCONV(64, 8, 512, 256, 1,
                              0, 1, 8, 128)

        self.TCONV3_2 = TCONV(128, 8, 256, 256, 2,
                              0, 2, 16, 256)
        self.TCONV3_1 = TCONV(128, 16, 256, 128, 1,
                              0, 1, 16, 256)
        self.TCONV2_2 = TCONV(256, 16, 128, 128, 2,
                              0, 2, 32, 512)
        self.TCONV1_2 = TCONV(512, 32, 64, 64, 2,
                              0, 2, 64, 512)
        self.TCONV1_1 = FCONV(512, 64, 64, 3, 1,
                              0, 1)


        # self.decoder32_1 = nn.Sequential(
        #     nn.PReLU(),
        #     nn.ConvTranspose2d(32, 512, kernel_size=2, padding=0, stride=2),  # (w-k+2p)/s+1
        #     nn.BatchNorm2d(512, affine=False),
        # )
        # self.decoder32_2 = nn.Sequential(
        #     nn.PReLU(),
        #     nn.ConvTranspose2d(512, 512, kernel_size=1, padding=0, stride=1),  # (w-k+2p)/s+1
        #     nn.BatchNorm2d(512, affine=False),
        # )
        # self.decoder16_1 = nn.Sequential(
        #     nn.PReLU(),
        #     nn.ConvTranspose2d(16, 256, kernel_size=2, padding=0, stride=2),  # (w-k+2p)/s+1
        #     nn.BatchNorm2d(512, affine=False),
        # )
        # self.decoder16_2 = nn.Sequential(
        #     nn.PReLU(),
        #     nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1, stride=1),  # (w-k+2p)/s+1
        #     nn.BatchNorm2d(256, affine=False),
        # )
        # self.decoder8_1 = nn.Sequential(
        #     nn.PReLU(),
        #     nn.ConvTranspose2d(8, 128, kernel_size=2, padding=0, stride=2),  # (w-k+2p)/s+1
        #     nn.BatchNorm2d(512, affine=False),
        # )
        # self.decoder8_2 = nn.Sequential(
        #     nn.PReLU(),
        #     nn.ConvTranspose2d(8, 128, kernel_size=3, padding=1, stride=1),  # (w-k+2p)/s+1
        #     nn.BatchNorm2d(256, affine=False),
        # )

        # if 2 in self.lasts:
        #     self.arms32 = nn.ModuleList([
        #         ConvNorm(self.num_filters(32, self._stem_head_width[1]), self.num_filters(16, self._stem_head_width[1]), 1, 1, 0, slimmable=False),
        #         ConvNorm(self.num_filters(16, self._stem_head_width[1]), self.num_filters(8, self._stem_head_width[1]), 1, 1, 0, slimmable=False),
        #     ])
        #     self.refines32 = nn.ModuleList([
        #         ConvNorm(self.num_filters(16, self._stem_head_width[1])+self.ch_16, self.num_filters(16, self._stem_head_width[1]), 3, 1, 1, slimmable=False),
        #         ConvNorm(self.num_filters(8, self._stem_head_width[1])+self.ch_8_2, self.num_filters(8, self._stem_head_width[1]), 3, 1, 1, slimmable=False),
        #     ])
        # if 1 in self.lasts:
        #     self.arms16 = ConvNorm(self.num_filters(16, self._stem_head_width[1]), self.num_filters(8, self._stem_head_width[1]), 1, 1, 0, slimmable=False)
        #     self.refines16 = ConvNorm(self.num_filters(8, self._stem_head_width[1])+self.ch_8_1, self.num_filters(8, self._stem_head_width[1]), 3, 1, 1, slimmable=False)
        # self.ffm = FeatureFusion(self.num_filters(8, self._stem_head_width[1]) * self._branch, self.num_filters(8, self._stem_head_width[1]) * self._branch, reduction=1, Fch=self._Fch, scale=8, branch=self._branch, norm_layer=BatchNorm2d)

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
                scale = 2 ** (paths[group[0]][l] + 3)
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

    def agg_ffm(self, outputs8, outputs16, outputs32, label_en):
        # pred32 = []; pred16 = []; pred8 = [] # order of predictions is not important
        for branch in range(self._branch):
            last = self.lasts[branch]
            # if last == 2:
            #     pred32 = outputs32[branch]
            #     # out = self.arms32[0](outputs32[branch])
            #     # out = F.interpolate(out, size=(int(out.size(2))*2, int(out.size(3))*2), mode='bilinear', align_corners=True)
            #     # out = self.refines32[0](torch.cat([out, outputs16[branch]], dim=1))
            #     # if self.training: pred16.append(outputs16[branch])
            #     # out = self.arms32[1](out)
            #     # out = F.interpolate(out, size=(int(out.size(2))*2, int(out.size(3))*2), mode='bilinear', align_corners=True)
            #     # out = self.refines32[1](torch.cat([out, outputs8[branch]], dim=1))
            #     # pred8.append(out)
            # elif last == 1:
            #     pred16 = outputs16[branch]
            #     # out = self.arms16(outputs16[branch])
            #     # out = F.interpolate(out, size=(int(out.size(2))*2, int(out.size(3))*2), mode='bilinear', align_corners=True)
            #     # out = self.refines16(torch.cat([out, outputs8[branch]], dim=1))
            #     # pred8.append(out)
            # elif last == 0:
            #     pred8 = outputs8[branch]

        out_flat32 = outputs32.view(-1, int(384 * 2 * 2))
        latent_mu32, latent_var32 = self.mean_layer32(out_flat32), self.var_layer32(out_flat32)
        latent_var32 = F.softplus(latent_var32) + 1e-8
        # latent_mu, latent_var = mu_var_dic["{}_{}".format(self._layers-1, 2)]
        latent32 = sample_gaussian(latent_mu32, latent_var32)
        predict32 = F.log_softmax(self.classifier32(latent32), dim=1)
        # print(predict)
        # predict_test32 = F.log_softmax(self.classifier32(latent_mu), dim=1)
        yh32 = self.one_hot32(label_en)

        out_flat16 = outputs16.view(-1, int(192 * 4 * 4))
        latent_mu16, latent_var16 = self.mean_layer16(out_flat16), self.var_layer16(out_flat16)
        latent_var16 = F.softplus(latent_var16) + 1e-8
        # latent_mu, latent_var = mu_var_dic["{}_{}".format(self._layers-1, 2)]
        latent16 = sample_gaussian(latent_mu16, latent_var16)
        predict16 = F.log_softmax(self.classifier16(latent16), dim=1)
        # print(predict)
        # predict_test16 = F.log_softmax(self.classifier16(latent_mu), dim=1)
        yh16 = self.one_hot16(label_en)

        out_flat8 = outputs8.view(-1, int(96 * 8 * 8))
        latent_mu8, latent_var8 = self.mean_layer8(out_flat8), self.var_layer8(out_flat8)
        latent_var8 = F.softplus(latent_var8) + 1e-8
        # latent_mu, latent_var = mu_var_dic["{}_{}".format(self._layers-1, 2)]
        latent8 = sample_gaussian(latent_mu8, latent_var8)
        predict8 = F.log_softmax(self.classifier8(latent8), dim=1)
        # print(predict)
        # predict_test8 = F.log_softmax(self.classifier8(latent_mu), dim=1)
        yh8 = self.one_hot8(label_en)


        # Simple decoder
        # out32 = outputs32
        # # print(out32.shape)
        # out16 = F.interpolate(self.refine32[0](out32), scale_factor=2, mode="bilinear", align_corners=True)
        # out16 = self.refine32[1](torch.cat([out16, outputs16], dim=1))
        # out8 = F.interpolate(self.refine16[0](out16), scale_factor=2, mode="bilinear", align_corners=True)
        # out8 = self.refine16[1](torch.cat([out8, outputs8], dim=1))
        # out4 = F.interpolate(self.refine8[0](out8), scale_factor=2, mode="bilinear", align_corners=True)
        # out2 = F.interpolate(self.refine4[0](out4), scale_factor=2, mode="bilinear", align_corners=True)
        # out1 = F.interpolate(self.refine2[0](out2), scale_factor=2, mode="bilinear", align_corners=True)
        # reconstructed = self.reconstruct[0](self.refine1[0](out1))

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

        dec3_1, mu_dn3_1, var_dn3_1 = self.TCONV3_2.decode(de_latent4_1)
        de_latent3_1 = sample_gaussian(mu_dn3_1, var_dn3_1)

        dec2_1, mu_dn2_1, var_dn2_1 = self.TCONV2_2.decode(de_latent3_1)
        de_latent2_1 = sample_gaussian(mu_dn2_1, var_dn2_1)
        dec1_1, mu_dn1_1, var_dn1_1 = self.TCONV1_2.decode(de_latent2_1)
        de_latent1_1 = sample_gaussian(mu_dn1_1, var_dn1_1)
        # print(dec1_1.shape)
        # print(de_latent1_1.shape)
        reconstructed = self.TCONV1_1.final_decode(de_latent1_1)



        pred_final = F.log_softmax(self.fc3(self.fc2(self.fc1(torch.cat((predict8,predict16,predict32),dim=1)))),dim=1)

        return predict32, predict16, predict8, pred_final, reconstructed, \
               [latent_mu32, latent_mu16, latent_mu8], [latent_var32, latent_var16, latent_var8], [yh32, yh16, yh8], [qmu5_1,qvar5_1],[qmu4_1,qvar4_1]
        # if len(pred32) > 0:
        #     pred32 = self.heads32(torch.cat(pred32, dim=1))
        # else:
        #     pred32 = None
        # if len(pred16) > 0:
        #     pred16 = self.heads16(torch.cat(pred16, dim=1))
        # else:
        #     pred16 = None
        # pred8 = self.heads8(self.ffm(torch.cat(pred8, dim=1)))
        # if self.training:
        #     return pred8, pred16, pred32
        # else:
        #     return pred8

    def forward(self, input, label_en):
        _, _, H, W = input.size()
        stem = self.stem(input)

        # store the last feature map w. corresponding scale of each branch
        # outputs8 = [stem] * self._branch
        # outputs16 = [stem] * self._branch
        # outputs32 = [stem] * self._branch
        outputs = [stem] * self._branch

        for layer in range(len(self.branch_groups)):
            for group in self.branch_groups[layer]:
                output = self.cells[str(layer) + "-" + str(group[0])](outputs[group[0]])
                scale = int(H // output.size(2))
                for branch in group:
                    outputs[branch] = output
                    # if scale == 8: outputs8[branch] = output
                    # elif scale == 16: outputs16[branch] = output
                    # elif scale == 32: outputs32[branch] = output
                    if scale == 8:
                        outputs8 = output
                    elif scale == 16:
                        outputs16 = output
                    elif scale == 32:
                        outputs32 = output

        # if self.training:
        #     pred8, pred16, pred32, reconstructed = self.agg_ffm(outputs8, outputs16, outputs32)
        #     pred8 = F.interpolate(pred8, scale_factor=8, mode='bilinear', align_corners=True)
        #     if pred16 is not None: pred16 = F.interpolate(pred16, scale_factor=16, mode='bilinear', align_corners=True)
        #     if pred32 is not None: pred32 = F.interpolate(pred32, scale_factor=32, mode='bilinear', align_corners=True)
        #     return pred8, pred16, pred32
        # else:
        #     pred8 = self.agg_ffm(outputs8, outputs16, outputs32)
        #     out = F.interpolate(pred8, size=(int(pred8.size(2))*8, int(pred8.size(3))*8), mode='bilinear', align_corners=True)
        output8 = outputs8.cuda()
        output16 = outputs16.cuda()
        output32 = outputs32.cuda()
        pred8, pred16, pred32, pred_final, reconstructed,\
            latent_mu, latent_var, yh, up32, up16 = self.agg_ffm(outputs8, outputs16, outputs32, label_en)
        return pred8, pred16, pred32, pred_final, reconstructed, latent_mu, latent_var, yh, up32, up16

    def forward_latency(self, size):
        _, H, W = size
        latency_total = 0
        latency, size = self.stem[0].forward_latency(size);
        latency_total += latency
        latency, size = self.stem[1].forward_latency(size);
        latency_total += latency
        latency, size = self.stem[2].forward_latency(size);
        latency_total += latency

        # store the last feature map w. corresponding scale of each branch
        outputs8 = [size] * self._branch
        outputs16 = [size] * self._branch
        outputs32 = [size] * self._branch
        outputs = [size] * self._branch

        for layer in range(len(self.branch_groups)):
            for group in self.branch_groups[layer]:
                latency, size = self.cells[str(layer) + "-" + str(group[0])].forward_latency(outputs[group[0]])
                latency_total += latency
                scale = int(H // size[1])
                for branch in group:
                    outputs[branch] = size
                    if scale == 8:
                        outputs8[branch] = size
                    elif scale == 16:
                        outputs16[branch] = size
                    elif scale == 32:
                        outputs32[branch] = size

        for branch in range(self._branch):
            last = self.lasts[branch]
            if last == 2:
                latency, size = self.arms32[0].forward_latency(outputs32[branch]);
                latency_total += latency
                latency, size = self.refines32[0].forward_latency((size[0] + self.ch_16, size[1] * 2, size[2] * 2));
                latency_total += latency
                latency, size = self.arms32[1].forward_latency(size);
                latency_total += latency
                latency, size = self.refines32[1].forward_latency((size[0] + self.ch_8_2, size[1] * 2, size[2] * 2));
                latency_total += latency
                out_size = size
            elif last == 1:
                latency, size = self.arms16.forward_latency(outputs16[branch]);
                latency_total += latency
                latency, size = self.refines16.forward_latency((size[0] + self.ch_8_1, size[1] * 2, size[2] * 2));
                latency_total += latency
                out_size = size
            elif last == 0:
                out_size = outputs8[branch]
        latency, size = self.ffm.forward_latency((out_size[0] * self._branch, out_size[1], out_size[2]));
        latency_total += latency
        latency, size = self.heads8.forward_latency(size);
        latency_total += latency
        return latency_total, size

