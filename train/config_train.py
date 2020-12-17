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
C.repo_name = 'FasterSeg'
C.abs_dir = osp.realpath(".")
C.this_dir = C.abs_dir.split(osp.sep)[-1]
C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]
C.log_dir = osp.abspath(osp.join(C.root_dir, 'log', C.this_dir))

"""Data Dir"""
C.dataset_path = "/ssd1/chenwy/cityscapes/"
C.img_root_folder = C.dataset_path
C.gt_root_folder = C.dataset_path
C.train_source = osp.join(C.dataset_path, "cityscapes_train_fine.txt")
C.train_eval_source = osp.join(C.dataset_path, "cityscapes_train_val_fine.txt")
C.eval_source = osp.join(C.dataset_path, "cityscapes_val_fine.txt")
C.test_source = osp.join(C.dataset_path, "cityscapes_test.txt")

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.join(C.root_dir, 'tools'))

"""Image Config"""
C.num_classes = 19
C.background = -1
C.image_mean = np.array([0.485, 0.456, 0.406])
C.image_std = np.array([0.229, 0.224, 0.225])
C.target_size = 1024
C.down_sampling = 1 # use downsampled images during search. In dataloader the image will first be down_sampled then cropped
C.gt_down_sampling = 1 # model's default output size without final upsampling
C.num_train_imgs = 2975 # number of training images
C.num_eval_imgs = 500 # number of validation images

""" Settings for network, this would be different for each kind of model"""
C.bn_eps = 1e-5 # a value added to the BN denominator for numerical stability
C.bn_momentum = 0.1 # value used for the running_mean and running_var computation

"""Train Config"""
C.lr = 0.01 # learning rate for updating supernet weight (NOT arch params)
C.momentum = 0.9 # SGD momentum
C.weight_decay = 5e-4 # SGD weight decay
C.nepochs = 600 # how many epochs to train
C.niters_per_epoch = 1000 # how many batches per epoch
C.num_workers = 6 # workers for dataloader
C.train_scale_array = [0.75, 1, 1.25] # scale factors for augmentation during training

"""Eval Config"""
C.eval_stride_rate = 5 / 6 # stride for crop based evaluation. Not used in this repo
C.eval_scale_array = [1, ] # multi-scale evaluation
C.eval_flip = False # flipping for evaluation
C.eval_base_size = 1024
C.eval_crop_size = 1024
C.eval_height = 1024 # real image height
C.eval_width = 2048 # real image width


C.layers = 16 # layers (cells) for network
""" Train Config """
C.mode = "student" # "teacher" or "student"
if C.mode == "teacher":
    ##### train teacher model only ####################################
    C.arch_idx = [0] # 0 for teacher
    C.branch = [2] # number of output branches
    C.width_mult_list = [4./12, 6./12, 8./12, 10./12, 1.,] # selection choices for channel pruning
    C.stem_head_width = [(1, 1)] # width ratio (#channel / Fch) for [teacher, student]
    C.load_path = "fasterseg" # path to the searched directory
    C.load_epoch = "last" # "last" or "int" (e.g. "30"): which epoch to load from the searched architecture
    C.batch_size = 12
    C.Fch = 12 # base channel number
    C.image_height = 512 # crop size for training
    C.image_width = 1024
    C.save = "%dx%d_teacher_batch%d"%(C.image_height, C.image_width, C.batch_size)
elif C.mode == "student":
    ##### train student with KL distillation from teacher ##############
    C.arch_idx = [0, 1] # 0 for teacher, 1 for student
    C.branch = [2, 2]
    C.width_mult_list = [4./12, 6./12, 8./12, 10./12, 1.,]
    C.stem_head_width = [(1, 1), (8./12, 8./12),]
    C.load_path = "fasterseg" # path to the searched directory
    C.teacher_path = "fasterseg" # where to load the pretrained teacher's weight
    C.load_epoch = "last" # "last" or "int" (e.g. "30")
    C.batch_size = 12
    C.Fch = 12
    C.image_height = 512
    C.image_width = 1024
    C.save = "%dx%d_student_batch%d"%(C.image_height, C.image_width, C.batch_size)

########################################
C.is_test = False # if True, prediction files for the test set will be generated
C.is_eval = False # if True, the train.py will only do evaluation for once
C.eval_path = "fasterseg" # path to pretrained directory to be evaluated
