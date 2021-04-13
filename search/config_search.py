# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
import numpy as np
from easydict import EasyDict as edict

C = edict()
config = C
cfg = C

C.seed = 12345

"""please config ROOT_dir and user when u first using"""
C.repo_name = 'NAS_OSR'
C.abs_dir = osp.realpath(".")
C.this_dir = C.abs_dir.split(osp.sep)[-1]
# print(C.abs_dir)
# print(C.abs_dir.index(C.repo_name) + len(C.repo_name))
# print(C.abs_dir[:38])
C.root_dir = C.abs_dir[: C.abs_dir.index(C.repo_name) + len(C.repo_name)]
# C.root_dir = "/media/neuron/Elements/openset/NAS_OSR"

"""Weight Hyperparameters"""
C.z_dim = 10
C.temperature = 1
C.beta_z = 6
C.beta = 1
C.lamda = 100
C.val_interval = 5
"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.join(C.root_dir, 'tools'))

"""Image Config"""
C.num_classes = 10
C.in_channel = 3
# C.background = -1
# C.image_mean = np.array([0.485, 0.456, 0.406])
# C.image_std = np.array([0.229, 0.224, 0.225])
# C.down_sampling = 2 # use downsampled images during search. In dataloader the image will first be down_sampled then cropped
# C.image_height = 64 # crop height after down_sampling in dataloader
# C.image_width = 64 # crop width after down_sampling in dataloader
# C.gt_down_sampling = 8 # model's default output size without final upsampling
C.num_train_imgs = 20000 # number of training images
C.num_eval_imgs = 10000 # number of validation images

""" Settings for network, this would be different for each kind of model"""
C.bn_eps = 1e-5 # a value added to the BN denominator for numerical stability
C.bn_momentum = 0.1 # value used for the running_mean and running_var computation

"""Train Config"""
C.lr = 0.0005 # learning rate for updating supernet weight (NOT arch params)
C.momentum = 0.9 # SGD momentum
C.weight_decay = 5e-4 # SGD weight decay
C.num_workers = 4 # workers for dataloader
C.train_scale_array = [0.75, 1, 1.25] # scale factors for augmentation during training
C.nepochs = 300
C.pretrain_nepochs = 100
"""Eval Config"""
C.eval_stride_rate = 5 / 6 # stride for crop based evaluation. Not used in this repo
C.eval_scale_array = [1, ] # multi-scale evaluation
C.eval_flip = False # flipping for evaluation
C.eval_height = 1024 # real image height
C.eval_width = 2048 # real image width


""" Search Config """
C.grad_clip = 5 # grad clip for search
C.train_portion = 0.5 # use how much % of training data for search
C.arch_learning_rate = 3e-4 # learning rate for updating arch params
C.arch_weight_decay = 0
C.layers = 3 # layers (cells) for supernet
C.branch = 3 # number of output branches

# C.pretrain = True
C.pretrain = "search-pretrain-2x2_F16.L3_batch50_100_1_6_lr0-20210312-192255"
########################################
# C.prun_modes = ['max', 'arch_ratio',] # channel pruning mode for [teacher, student], i.e. by default teacher will use max channel number, and student will sample the channel number based on arch_ratio
C.prun_modes = ['max',]
C.Fch = 16 # base channel number
C.width_mult_list = [1.] # selection choices for channel pruning
C.stem_head_width = [(1, 1),] # width ratio (#channel / Fch) for [teacher, student]
# C.FPS_min = [0, 155.] # minimum FPS required for [teacher, student]
# C.FPS_max = [0, 175.] # maximum FPS allowed for [teacher, student]
if C.pretrain == True:
    C.batch_size =  10
    # C.niters_per_epoch = max(C.num_train_imgs // 2 // C.batch_size, 400)
    C.niters_per_epoch = C.num_train_imgs // 2 // C.batch_size
    C.lr = 0.0005
    C.latency_weight = [0, 0] # weight of latency penalty loss
    C.image_height =2 # this size is after down_sampling
    C.image_width = 2

    C.save = "pretrain-%dx%d_F%d.L%d_batch%d"%(C.image_height, C.image_width, C.Fch, C.layers, C.batch_size)
else:
    C.batch_size = 64
    # C.niters_per_epoch = max(C.num_train_imgs // 2 // C.batch_size, 400)
    C.niters_per_epoch = C.num_train_imgs // 2 // C.batch_size
    C.latency_weight = [0, 1e-2,]
    C.image_height = 2 # this size is after down_sampling
    C.image_width = 2
    C.nepochs = 150
    C.save = "%dx%d_F%d.L%d_batch%d"%(C.image_height, C.image_width, C.Fch, C.layers, C.batch_size)
########################################
# assert len(C.latency_weight) == len(C.stem_head_width) and len(C.stem_head_width) == len(C.FPS_min) and len(C.FPS_min) == len(C.FPS_max)

C.unrolled = False # for DARTS v2
