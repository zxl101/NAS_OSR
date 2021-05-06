# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
from easydict import EasyDict as edict

C = edict()
config = C
cfg = C
C.seed = 123456

"""please config ROOT_dir and user when u first using"""
C.repo_name = 'NAS_OSR'
C.abs_dir = osp.realpath(".")
C.this_dir = C.abs_dir.split(osp.sep)[-1]
C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]
C.log_dir = osp.abspath(osp.join(C.root_dir, 'log', C.this_dir))

"""Weight Hyperparameters"""
C.z_dim = 10
C.temperature = 1
C.beta_z = 6
C.beta = 1
C.lamda = 100
C.val_interval = 5
C.use_cuda = True
C.cuda = True

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.join(C.root_dir, 'tools'))

"""Data Config"""
C.num_classes = 10
C.unseen_num = 50

""" Settings for network, this would be different for each kind of model"""
C.bn_eps = 1e-5 # a value added to the BN denominator for numerical stability
C.bn_momentum = 0.1 # value used for the running_mean and running_var computation

"""Train Config"""
C.lr = 0.0005 # learning rate for updating supernet weight (NOT arch params)
C.momentum = 0.9 # SGD momentum
C.weight_decay = 5e-4 # SGD weight decay
C.nepochs = 300 # how many epochs to train
C.num_workers = 6 # workers for dataloader
# C.train_scale_array = [0.75, 1, 1.25] # scale factors for augmentation during training
C.layers = 6 # layers (cells) for network

""" Train Config """
C.mode = "teacher" # "teacher" or "student"
if C.mode == "teacher":
    ##### train teacher model only ####################################
    C.arch_idx = [0] # 0 for teacher
    C.branch = [3] # number of output branches
    C.width_mult_list = [1.,] # selection choices for channel pruning
    C.stem_head_width = [(1, 1)] # width ratio (#channel / Fch) for [teacher, student]
    C.load_path = "search-2x2_F16.L6_batch64_80_5_10_lr0-20210327-060637" # path to the searched directory
    C.load_epoch = "last" # "last" or "int" (e.g. "30"): which epoch to load from the searched architecture
    C.batch_size = 64
    # C.niters_per_epoch = min(1000, C.num_train_imgs // C.batch_size)  # how many batches per epoch
    C.Fch = 16 # base channel number
    C.image_height = 64
    C.image_width = 64
    C.save = "%dx%d_teacher_batch%d"%(C.image_height, C.image_width, C.batch_size)
# elif C.mode == "student":
#     ##### train student with KL distillation from teacher ##############
#     C.arch_idx = [0, 1] # 0 for teacher, 1 for student
#     C.branch = [3, 3]
#     C.width_mult_list = [4./12, 6./12, 8./12, 10./12, 1.,]
#     C.stem_head_width = [(1, 1), (8./12, 8./12),]
#     C.load_path = "fasterseg" # path to the searched directory
#     C.teacher_path = "fasterseg" # where to load the pretrained teacher's weight
#     C.load_epoch = "last" # "last" or "int" (e.g. "30")
#     C.batch_size = 12
#     C.Fch = 12
#     C.image_height = 512
#     C.image_width = 1024
#     C.save = "%dx%d_student_batch%d"%(C.image_height, C.image_width, C.batch_size)

########################################
C.is_train = True
C.is_test = False # if True, prediction files for the test set will be generated
C.is_eval = False # if True, the train.py will only do evaluation for once
C.eval_path = "best_model/CIFARAddN/47"
