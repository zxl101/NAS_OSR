from __future__ import division
import os
import sys
import time
import glob
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from thop import profile

from config_search import config
# from dataloader import get_train_loader
# from datasets import Cityscapes

from utils.init_func import init_weight
# from seg_opr.loss_opr import ProbOhemCrossEntropy2d
# from eval import SegEvaluator

from architect import Architect
from utils.darts_utils import create_exp_dir, save, plot_op, plot_path_width, objective_acc_lat
from model_search import Network_Multi_Path as Network
from model_seg import Network_Multi_Path_Infer

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score
import PIL
import argparse

parser = argparse.ArgumentParser(description='PyTorch OSR Example')
parser.add_argument('--batch_size', type=int, default=None, help='input batch size for training (default: 64)')
parser.add_argument('--num_classes', type=int, default=None, help='number of classes')
parser.add_argument('--nepochs', type=int, default=None, help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=None, help='learning rate (default: 1e-3)')
parser.add_argument('--layers', type=int, default=None)
parser.add_argument('--weight_decay', type=float, default=0.00005, help='weight decay')
# parser.add_argument('--momentum', type=float, default=0.01, help='momentum (default: 1e-3)')
# parser.add_argument('--decreasing_lr', default='60,100,150', help='decreasing strategy')
# parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
# parser.add_argument('--log_interval', type=int, default=20,
#                     help='how many batches to wait before logging training status')
# parser.add_argument('--val_interval', type=int, default=5, help='how many epochs to wait before another val')
# parser.add_argument('--test_interval', type=int, default=5, help='how many epochs to wait before another test')
# parser.add_argument('--lamda', type=int, default=100, help='lamda in loss function')
parser.add_argument('--wce', type=float, default=1)
parser.add_argument('--wre', type=float, default=1)
parser.add_argument('--wkl', type=float, default=1)
parser.add_argument('--wdist', type=float, default=1)
# parser.add_argument('--wkl', type=float, default=1)
args = parser.parse_args()


def main(pretrain=True):

    config.wce = args.wce
    config.wre = args.wre
    config.wkl = args.wkl
    config.wdist = args.wdist
    config.weight_decay = args.weight_decay
    if args.batch_size != None:
        config.batch_size = args.batch_size
        config.niters_per_epoch = min(config.num_train_imgs // 2 // config.batch_size, 400)
    if args.num_classes != None:
        config.num_classes = args.num_classes
    if args.nepochs != None:
        config.nepochs = args.nepochs
    if args.lr != None:
        config.lr = args.lr
    if args.layers != None:
        config.layers = args.layers

    if config.pretrain == True:
        config.save = "pretrain-%dx%d_F%d.L%d_batch%d_%d_%d_%d_lr%d"%(config.image_height, config.image_width,
                                                                      config.Fch, config.layers, config.batch_size,
                                                                      config.wce,config.wre,config.wkl,config.lr)
    else:
        config.save = "%dx%d_F%d.L%d_batch%d_%d_%d_%d_lr%d" % (config.image_height, config.image_width,
                                                                        config.Fch, config.layers, config.batch_size,
                                                                        config.wce, config.wre, config.wkl, config.lr)

    config.save = 'search-{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(config.save, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh'))
    logger = SummaryWriter(config.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    assert type(pretrain) == bool or type(pretrain) == str
    update_arch = True
    if pretrain == True:
        update_arch = False
    logging.info("args = %s", str(config))
    # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # config network and criterion ################
    # min_kept = int(config.batch_size * config.image_height * config.image_width // (16 * config.gt_down_sampling ** 2))
    # ohem_criterion = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7, min_kept=min_kept, use_weight=False)

    # Model #######################################
    model = Network(config.num_classes, config.in_channel, config.layers, Fch=config.Fch, width_mult_list=config.width_mult_list, prun_modes=config.prun_modes, stem_head_width=config.stem_head_width)
    # input_check = (torch.randn(1, 3, 64, 64,device=torch.device("cpu")),torch.randn(1, device=torch.device("cpu")), torch.randn(1,10,device=torch.device("cpu")))
    # # for item in input_check:
    # #     item.to(torch.device("cuda"))
    # flops, params = profile(model, inputs= input_check, verbose=False)
    # logging.info("params = %fMB, FLOPs = %fGB", params / 1e6, flops / 1e9)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if type(pretrain) == str:
        partial = torch.load(pretrain + "/weights.pt", map_location='cuda:0')
        state = model.state_dict()
        pretrained_dict = {k: v for k, v in partial.items() if k in state and state[k].size() == partial[k].size()}
        state.update(pretrained_dict)
        model.load_state_dict(state)
    else:
        init_weight(model, nn.init.kaiming_normal_, nn.BatchNorm2d, config.bn_eps, config.bn_momentum, mode='fan_in', nonlinearity='relu')
        # model.one_hot32.weight.data.uniform_(0,1)
        # model.one_hot32.bias.data.fill_(0)
        # model.one_hot16.weight.data.uniform_(0,1)
        # model.one_hot16.bias.data.fill_(0)
        # model.one_hot8.weight.data.uniform_(0,1)
        # model.one_hot8.bias.data.fill_(0)

    model = nn.DataParallel(model)
    model.to(device)
    architect = Architect(model, config)

    # Optimizer ###################################
    base_lr = config.lr
    parameters = []
    parameters += list(model.module.stem.parameters())
    parameters += list(model.module.cells.parameters())
    # parameters += list(model.module.refine32.parameters())
    # parameters += list(model.module.refine16.parameters())
    # parameters += list(model.module.refine8.parameters())
    # parameters += list(model.module.refine4.parameters())
    # parameters += list(model.module.refine2.parameters())
    # parameters += list(model.module.refine1.parameters())
    # parameters += list(model.module.reconstruct.parameters())
    parameters += list(model.module.mean_layer32.parameters())
    parameters += list(model.module.var_layer32.parameters())
    parameters += list(model.module.classifier32.parameters())
    parameters += list(model.module.one_hot32.parameters())
    parameters += list(model.module.mean_layer16.parameters())
    parameters += list(model.module.var_layer16.parameters())
    parameters += list(model.module.classifier16.parameters())
    parameters += list(model.module.one_hot16.parameters())
    parameters += list(model.module.mean_layer8.parameters())
    parameters += list(model.module.var_layer8.parameters())
    parameters += list(model.module.classifier8.parameters())
    parameters += list(model.module.one_hot8.parameters())
    parameters += list(model.module.fc1.parameters())
    parameters += list(model.module.fc2.parameters())
    parameters += list(model.module.fc3.parameters())
    parameters += list(model.module.TCONV5_2.parameters())
    # parameters += list(model.module.TCONV5_1.parameters())
    parameters += list(model.module.TCONV4_2.parameters())
    # parameters += list(model.module.TCONV4_1.parameters())
    parameters += list(model.module.TCONV3_2.parameters())
    # parameters += list(model.module.TCONV3_1.parameters())
    parameters += list(model.module.TCONV2_2.parameters())
    parameters += list(model.module.TCONV1_2.parameters())
    parameters += list(model.module.TCONV1_1.parameters())
    # optimizer = torch.optim.SGD(
    #     parameters,
    #     # model.parameters(),
    #     lr=base_lr,
    #     momentum=config.momentum,
    #     weight_decay=config.weight_decay)
    optimizer = torch.optim.Adam(
        params=parameters, lr=base_lr,weight_decay=config.weight_decay
    )

    # lr policy ##############################
    lr_policy = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.990)

    # data loader ###########################
    # data_setting = {'img_root': config.img_root_folder,
    #                 'gt_root': config.gt_root_folder,
    #                 'train_source': config.train_source,
    #                 'eval_source': config.eval_source,
    #                 'down_sampling': config.down_sampling}
    # train_loader_model = get_train_loader(config, Cityscapes, portion=config.train_portion)
    # train_loader_arch = get_train_loader(config, Cityscapes, portion=config.train_portion-1)
    #
    # evaluator = SegEvaluator(Cityscapes(data_setting, 'val', None), config.num_classes, config.image_mean,
    #                          config.image_std, model, config.eval_scale_array, config.eval_flip, 0, config=config,
    #                          verbose=False, save_path=None, show_image=False)
    train_dataset = datasets.CIFAR10('data/cifar10', download=False, train=True,
                                   transform=transforms.Compose([

                                       transforms.ToTensor(),
                                       transforms.Resize(64),
                                       transforms.ColorJitter(hue=.05, saturation=.05),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
                                       transforms.Normalize((0.5,), (0.5,))
                                   ]))
    train_dataset_model, train_dataset_arch = torch.utils.data.random_split(train_dataset,[25000,25000])
    # print(len(train_dataset_model))
    # print(len(train_datset_arch))
    # return None
    train_loader_model = DataLoader(train_dataset_model, batch_size=config.batch_size, shuffle=True, num_workers=4)

    # train_dataset_arch = datasets.CIFAR10('data/cifar10', download=False, train=True,
    #                                transform=transforms.Compose([
    #
    #                                    transforms.ToTensor(),
    #                                    transforms.Resize(64),
    #                                    transforms.ColorJitter(hue=.05, saturation=.05),
    #                                    transforms.RandomHorizontalFlip(),
    #                                    transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
    #                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
    train_loader_arch = DataLoader(train_dataset_arch, batch_size=config.batch_size, shuffle=True, num_workers=4)

    val_dataset = datasets.CIFAR10('data/cifar10', download=False, train=False,
                                 transform=transforms.Compose([

                                     transforms.ToTensor(),
                                     transforms.Resize(64),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    print("The size of the validation set is: {}".format(len(val_loader.dataset)))

    if update_arch:
        for idx in range(len(config.latency_weight)):
            logger.add_scalar("arch/latency_weight%d"%idx, config.latency_weight[idx], 0)
            logging.info("arch_latency_weight%d = "%idx + str(config.latency_weight[idx]))

    tbar = tqdm(range(config.nepochs), ncols=80)
    valid_loss_history = []; FPSs_history = [];
    latency_supernet_history = []; latency_weight_history = [];
    valid_names = ["8s", "16s", "32s", "8s_32s", "16s_32s"]
    arch_names = {0: "teacher", 1: "student"}
    for epoch in tbar:
        logging.info(pretrain)
        logging.info(config.save)
        logging.info("lr: " + str(optimizer.param_groups[0]['lr']))

        logging.info("update arch: " + str(update_arch))

        # training
        tbar.set_description("[Epoch %d/%d][train...]" % (epoch + 1, config.nepochs))
        # print("The learning rate of current epoch is {}".format(optimizer.lr))
        train(pretrain, train_loader_model, train_loader_arch, model, architect, optimizer, lr_policy, logger, epoch, update_arch=update_arch, config=config, device=device)
        torch.cuda.empty_cache()
        lr_policy.step()

        # validation
        # if epoch % 3 == 1:
        tbar.set_description("[Epoch %d/%d][validation...]" % (epoch + 1, config.nepochs))
        with torch.no_grad():
            if pretrain == True:
                model.module.prun_mode = "min"
                ce_loss, re_loss, kl_loss = infer(epoch, model, val_loader, logger, FPS=False, config=config)
                logger.add_scalar('ce_loss/val_min', ce_loss, epoch)
                logger.add_scalar('re_loss/val_min', re_loss, epoch)
                logger.add_scalar('kl_loss/val_min', kl_loss, epoch)
                # logger.add_scalar('latent_distance/val_search', latent_distance, epoch)
                logging.info("Epoch %d: valid_kl_loss_search %.3f" % (epoch, kl_loss))
                logging.info("Epoch %d: valid_ce_loss_search %.3f" % (epoch, ce_loss))
                logging.info("Epoch %d: valid_re_loss_search %.3f" % (epoch, re_loss))
                # logging.info("Epoch %d: valid_latent_distance %.3f" % (epoch, latent_distance))
                # if len(model.module._width_mult_list) > 1:
                #     model.module.prun_mode = "max"
                #     ce_loss, re_loss, kl_loss = infer(epoch, model, val_loader, logger, FPS=False, config=config, add_scalar=True)
                #     logger.add_scalar('ce_loss/val_max', ce_loss, epoch)
                #     logger.add_scalar('re_loss/val_max', re_loss, epoch)
                #     logger.add_scalar('kl_loss/val_max', kl_loss, epoch)
                #     logging.info("Epoch %d: valid_kl_loss_max %.3f" % (epoch, kl_loss))
                #     logging.info("Epoch %d: valid_ce_loss_max %.3f" % (epoch, ce_loss))
                #     logging.info("Epoch %d: valid_re_loss_max %.3f" % (epoch, re_loss))
                #     model.module.prun_mode = "random"
                #     ce_loss, re_loss, kl_loss = infer(epoch, model, val_loader, logger, FPS=False, config=config)
                #     logger.add_scalar('ce_loss/val_random', ce_loss, epoch)
                #     logger.add_scalar('re_loss/val_random', re_loss, epoch)
                #     logger.add_scalar('kl_loss/val_random', kl_loss, epoch)
                #     logging.info("Epoch %d: valid_kl_loss_random %.3f" % (epoch, kl_loss))
                #     logging.info("Epoch %d: valid_ce_loss_random %.3f" % (epoch, ce_loss))
                #     logging.info("Epoch %d: valid_re_loss_random %.3f" % (epoch, re_loss))
            else:
                valid_losses = []; FPSs = []
                model.prun_mode = None
                for idx in range(len(model.module._arch_names)):
                    # arch_idx
                    model.module.arch_idx = idx
                    # valid_loss, fps0, fps1 = infer(epoch, model.module, val_loader, logger, config=config)
                    ce_loss, re_loss, kl_loss = infer(epoch, model, val_loader, logger, config=config)
                    # valid_losses.append(valid_loss)
                    # FPSs.append([fps0, fps1])
                    # logger.add_scalar('mIoU/val_min', valid_loss, epoch)
                    # logging.info("Epoch %d: valid_loss_min %.3f" % (epoch, loss))
                    logger.add_scalar('ce_loss/val_search', ce_loss, epoch)
                    logger.add_scalar('re_loss/val_search', re_loss, epoch)
                    logger.add_scalar('kl_loss/val_search', kl_loss, epoch)
                    # logger.add_scalar('latent_distance/val_search', latent_distance, epoch)
                    logging.info("Epoch %d: valid_kl_loss_search %.3f" % (epoch, kl_loss))
                    logging.info("Epoch %d: valid_ce_loss_search %.3f" % (epoch, ce_loss))
                    logging.info("Epoch %d: valid_re_loss_search %.3f" % (epoch, re_loss))
                    # logging.info("Epoch %d: valid_latent_distance %.3f" % (epoch, latent_distance))
                    # if config.latency_weight[idx] > 0:
                    #     logger.add_scalar('Objective/val_%s_8s_32s'%arch_names[idx], objective_acc_lat(valid_mIoUs[3], 1000./fps0), epoch)
                    #     logging.info("Epoch %d: Objective_%s_8s_32s %.3f"%(epoch, arch_names[idx], objective_acc_lat(valid_mIoUs[3], 1000./fps0)))
                    #     logger.add_scalar('Objective/val_%s_16s_32s'%arch_names[idx], objective_acc_lat(valid_mIoUs[4], 1000./fps1), epoch)
                    #     logging.info("Epoch %d: Objective_%s_16s_32s %.3f"%(epoch, arch_names[idx], objective_acc_lat(valid_mIoUs[4], 1000./fps1)))
                # valid_loss_history.append(valid_losses)
                # FPSs_history.append(FPSs)
                if update_arch:
                    latency_supernet_history.append(architect.latency_supernet)
                latency_weight_history.append(architect.latency_weight)

        save(model, os.path.join(config.save, 'weights.pt'))
        if type(pretrain) == str:
            # contains arch_param names: {"alphas": alphas, "betas": betas, "gammas": gammas, "ratios": ratios}
            for idx, arch_name in enumerate(model.module._arch_names):
                state = {}
                for name in arch_name['alphas']:
                    state[name] = getattr(model.module, name)
                for name in arch_name['betas']:
                    state[name] = getattr(model.module, name)
                for name in arch_name['ratios']:
                    state[name] = getattr(model.module, name)
                # state["mIoU02"] = valid_mIoUs[3]
                # state["mIoU12"] = valid_mIoUs[4]
                state["ce_loss"] = ce_loss
                state["re_loss"] = re_loss
                state["kl_loss"] = kl_loss
                # if pretrain is not True:
                #     state["latency02"] = 1000. / fps0
                #     state["latency12"] = 1000. / fps1
                torch.save(state, os.path.join(config.save, "arch_%d_%d.pt"%(idx, epoch)))
                torch.save(state, os.path.join(config.save, "arch_%d.pt"%(idx)))

        # if update_arch:
        #     for idx in range(len(config.latency_weight)):
        #         if config.latency_weight[idx] > 0:
        #             if (int(FPSs[idx][0] >= config.FPS_max[idx]) + int(FPSs[idx][1] >= config.FPS_max[idx])) >= 1:
        #                 architect.latency_weight[idx] /= 2
        #             elif (int(FPSs[idx][0] <= config.FPS_min[idx]) + int(FPSs[idx][1] <= config.FPS_min[idx])) > 0:
        #                 architect.latency_weight[idx] *= 2
        #             logger.add_scalar("arch/latency_weight_%s"%arch_names[idx], architect.latency_weight[idx], epoch+1)
        #             logging.info("arch_latency_weight_%s = "%arch_names[idx] + str(architect.latency_weight[idx]))


def train(pretrain, train_loader_model, train_loader_arch, model, architect, optimizer, lr_policy, logger, epoch, update_arch=True, config = None, device = None):
    model.train()
    model.module.latent_space(epoch=epoch, vis=True)
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format, ncols=80)
    dataloader_model = iter(train_loader_model)
    dataloader_arch = iter(train_loader_arch)
    epoch_ce_loss = 0
    epoch_re_loss = 0
    epoch_kl_loss = 0
    count = 0
    c8 = 0
    c16 = 0
    c32 = 0
    cfinal = 0
    total = 0
    for step in pbar:
        optimizer.zero_grad()


        minibatch = dataloader_model.next()
        imgs = minibatch[0]
        imgs = minibatch[0]
        target = minibatch[1]
        target_en = torch.Tensor(target.shape[0], config.num_classes)
        target_en.zero_()
        target_en.scatter_(1, target.view(-1, 1), 1)  # one-hot encoding
        target_en = target_en.to(device)
        imgs = imgs.to(device)
        target = target.to(device)

        if update_arch:
            # get a random minibatch from the search queue with replacement
            pbar.set_description("[Arch Step %d/%d]" % (step + 1, len(train_loader_model)))
            minibatch = dataloader_arch.next()
            imgs_search = minibatch[0]
            target_search = minibatch[1]
            target_en_search = torch.Tensor(target_search.shape[0], config.num_classes)
            target_en_search.zero_()
            target_en_search.scatter_(1, target_search.view(-1, 1), 1)  # one-hot encoding
            target_en_search = target_en_search.to(device)
            imgs_search = imgs_search.to(device)
            target_search = target_search.to(device)

            loss_arch = architect.step(imgs, target, target_en, imgs_search, target_search, target_en_search)
            if (step+1) % 10 == 0:
                logger.add_scalar('loss_arch/train', loss_arch, epoch*len(pbar)+step)
                # logger.add_scalar('arch/latency_supernet', architect.latency_supernet, epoch*len(pbar)+step)

        ce_loss, re_loss, kl_loss, preds = model(input=imgs, target=target, target_de=target_en, pretrain=pretrain)
        # logger.add_graph(model.module,(imgs,target_en))
        # print(ce_loss)
        # print(re_loss)
        # print(kl_loss)
        c, batch_count = cal_acc(preds, target)
        c8 += c[2]
        c16 += c[1]
        c32 += c[0]
        cfinal += c[3]
        total += batch_count
        ce_loss = torch.mean(ce_loss)
        re_loss = torch.mean(re_loss)
        kl_loss = torch.mean(kl_loss)
        # latent_distance = torch.mean(latent_distance)
        epoch_ce_loss += ce_loss
        epoch_re_loss += re_loss
        epoch_kl_loss += kl_loss
        # print("The training loss of this batch is: {}".format(loss))
        logger.add_scalar('loss_step/train_ce', ce_loss, epoch*len(pbar)+step)
        logger.add_scalar('loss_step/train_re', re_loss, epoch * len(pbar) + step)
        logger.add_scalar('loss_step/train_kl', kl_loss, epoch * len(pbar) + step)
        # logger.add_scalar('loss_step/train_latent', latent_distance, epoch * len(pbar) + step)
        loss = config.wce * ce_loss + config.wre * re_loss + ((epoch + 1)/config.nepochs) * config.wkl * kl_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.module.parameters(), config.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        pbar.set_description("[Step %d/%d]" % (step + 1, len(train_loader_model)))
        count+=1
        # break
    logger.add_scalar('loss_epoch/train_ce', epoch_ce_loss/count, epoch)
    logger.add_scalar('loss_epoch/train_re', epoch_re_loss / count, epoch)
    logger.add_scalar('loss_epoch/train_kl', epoch_kl_loss / count, epoch)
    logger.add_scalar('train_acc/acc8', c8 / total, epoch)
    logger.add_scalar('train_acc/acc16', c16 / total, epoch)
    logger.add_scalar('train_acc/acc32', c32 / total, epoch)
    logger.add_scalar('train_acc/acc_final', cfinal / total, epoch)
    print("The training loss of this epoch is: {}".format((epoch_ce_loss+epoch_re_loss+kl_loss)/count))
    print("The average training accuracy of this epoch is: {}".format( cfinal/ total))
    torch.cuda.empty_cache()
    # del loss
    # if update_arch: del loss_arch

def infer(epoch, model, val_loader, logger, FPS=True, config=None, device=torch.device("cuda"), add_scalar = False):
    model.eval()
    total_ce_loss = 0
    total_re_loss = 0
    total_kl_loss = 0
    total_distance = 0
    c8 = 0
    c16 = 0
    c32 = 0
    cfinal = 0
    total = 0
    i = 0
    for data_val, target_val in val_loader:
        # print("Current working on {} batch".format(i))
        target_val_en = torch.Tensor(target_val.shape[0], config.num_classes)
        target_val_en.zero_()
        target_val_en.scatter_(1, target_val.view(-1, 1), 1)  # one-hot encoding
        target_val_en = target_val_en.to(device)
        data_val, target_val = data_val.to(device), target_val.to(device)
        ce_loss, re_loss, kl_loss, preds = model(data_val, target_val, target_val_en)
        c, batch_count = cal_acc(preds, target_val)
        c8 += c[2]
        c16 += c[1]
        c32 += c[0]
        cfinal += c[3]
        total += batch_count
        ce_loss = torch.mean(ce_loss)
        re_loss = torch.mean(re_loss)
        kl_loss = torch.mean(kl_loss)
        # latent_distance = torch.mean(latent_distance)
        # total_distance += latent_distance
        total_ce_loss += config.wce * ce_loss
        total_re_loss += config.wre * re_loss
        total_kl_loss += config.wkl * kl_loss
        i += 1
        # print(c_loss)
        # break
    total_ce_loss = total_ce_loss / i
    total_re_loss = total_re_loss / i
    total_kl_loss = total_kl_loss / i
    # total_distance = total_distance / i
    print("The validation ce loss is: {}".format(total_ce_loss))
    print("The validation re loss is: {}".format(total_re_loss))
    print("The validation kl loss is: {}".format(total_kl_loss))
    if add_scalar == True:
        logger.add_scalar('val_acc/acc8', c8 / total, epoch)
        logger.add_scalar('val_acc/acc16', c16 / total, epoch)
        logger.add_scalar('val_acc/acc32', c32 / total, epoch)
        logger.add_scalar('val_acc/acc_final', cfinal / total, epoch)
        print("The average validation accuracy of this epoch is: {}".format(cfinal / total))


    # if FPS:
    #     fps0, fps1 = arch_logging(model, config, logger, epoch)
    #     return total_ce_loss, fps0, fps1
    # else:
    #     return total_ce_loss, total_re_loss, total_kl_loss
    return total_ce_loss, total_re_loss, total_kl_loss

# def infer(epoch, model, val_loader, logger, FPS=True):
#     model.eval()
#     mIoUs = []
#     for idx in range(5):
#         evaluator.out_idx = idx
#         # _, mIoU = evaluator.run_online()
#         _, mIoU = evaluator.run_online_multiprocess()

#         mIoUs.append(mIoU)
#     if FPS:
#         fps0, fps1 = arch_logging(model, config, logger, epoch)
#         return mIoUs, fps0, fps1
#     else:
#         return mIoUs


def arch_logging(model, args, logger, epoch):
    input_size = (1, 3, 64, 64)
    net = Network_Multi_Path_Infer(
        [getattr(model, model._arch_names[model.arch_idx]["alphas"][0]).clone().detach(), getattr(model, model._arch_names[model.arch_idx]["alphas"][1]).clone().detach(), getattr(model, model._arch_names[model.arch_idx]["alphas"][2]).clone().detach()],
        [None, getattr(model, model._arch_names[model.arch_idx]["betas"][0]).clone().detach(), getattr(model, model._arch_names[model.arch_idx]["betas"][1]).clone().detach()],
        [getattr(model, model._arch_names[model.arch_idx]["ratios"][0]).clone().detach(), getattr(model, model._arch_names[model.arch_idx]["ratios"][1]).clone().detach(), getattr(model, model._arch_names[model.arch_idx]["ratios"][2]).clone().detach()],
        num_classes=model._num_classes, in_channel=model.in_channel, layers=model._layers, Fch=model._Fch, width_mult_list=model._width_mult_list, stem_head_width=model._stem_head_width[model.arch_idx])

    plot_op(net.ops0, net.path0, F_base=args.Fch).savefig("table.png", bbox_inches="tight")
    logger.add_image("arch/ops0_arch%d"%model.arch_idx, np.swapaxes(np.swapaxes(plt.imread("table.png"), 0, 2), 1, 2), epoch)
    plot_op(net.ops1, net.path1, F_base=args.Fch).savefig("table.png", bbox_inches="tight")
    logger.add_image("arch/ops1_arch%d"%model.arch_idx, np.swapaxes(np.swapaxes(plt.imread("table.png"), 0, 2), 1, 2), epoch)
    plot_op(net.ops2, net.path2, F_base=args.Fch).savefig("table.png", bbox_inches="tight")
    logger.add_image("arch/ops2_arch%d"%model.arch_idx, np.swapaxes(np.swapaxes(plt.imread("table.png"), 0, 2), 1, 2), epoch)

    # net.build_structure([2, 0])
    # net = net.cuda()
    # net.eval()
    # latency0, _ = net.forward_latency(input_size[1:])
    # logger.add_scalar("arch/fps0_arch%d"%model.arch_idx, 1000./latency0, epoch)
    # logger.add_figure("arch/path_width_arch%d_02"%model.arch_idx, plot_path_width([2, 0], [net.path2, net.path0], [net.widths2, net.widths0]), epoch)
    #
    # net.build_structure([2, 1])
    # net = net.cuda()
    # net.eval()
    # latency1, _ = net.forward_latency(input_size[1:])
    # logger.add_scalar("arch/fps1_arch%d"%model.arch_idx, 1000./latency1, epoch)
    # logger.add_figure("arch/path_width_arch%d_12"%model.arch_idx, plot_path_width([2, 1], [net.path2, net.path1], [net.widths2, net.widths1]), epoch)

    net.build_structure([2, 1, 0])
    net = net.to(device)
    net.eval()
    latency2, _ = net.forward_latency(input_size[1:])
    logger.add_scalar("arch/fps2_arch%d" % model.arch_idx, 1000. / latency2, epoch)
    logger.add_figure("arch/path_width_arch%d_012" % model.arch_idx,
                      plot_path_width([2, 1, 0], [net.path2, net.path1, net.path0], [net.widths2, net.widths1, net.widths0]), epoch)

    # return 1000./latency0, 1000./latency1, 1000./latency2
    return 1000./latency2

def cal_acc(preds, target):
    c = [0, 0, 0, 0]
    i = 0
    for pred in preds:
        pred = pred.data.max(1)[1]
        c[i] += pred.eq(target.view_as(pred)).sum().item()
        i += 1
    return c, len(target)

if __name__ == '__main__':
    main(pretrain=config.pretrain) 
