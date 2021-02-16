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
import torch.nn.functional as F
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter

import numpy as np
from thop import profile

from config_train import config
if config.is_eval:
    config.save = 'eval-{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
else:
    config.save = 'train-{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
from dataloader import get_train_loader
# from datasets import Cityscapes

from utils.init_func import init_weight
# from seg_opr.loss_opr import ProbOhemCrossEntropy2d
from eval import SegEvaluator
from test import SegTester

from utils.darts_utils import create_exp_dir, save, plot_op, plot_path_width, objective_acc_lat
from model_seg import Network_Multi_Path_Infer as Network
import seg_metrics

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import pickle
import PIL
from PIL import Image
import cv2
import matplotlib
import argparse

def sample_gaussian(m, v):
    sample = torch.randn(m.shape).to(torch.device("cuda"))
    # sample = torch.randn(m.shape)
    m = m.cuda()
    v = v.cuda()
    z = m + (v ** 0.5) * sample
    return z

# from torchviz import make_dot
# from torchsummary import summary

reconstruction_function = nn.L1Loss()
reconstruction_function.reduction = 'mean'
criterion = nn.NLLLoss(reduction='mean')

def kl_normal(qm, qv, pm, pv, yh):
	element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm - yh).pow(2) / pv - 1)
	# element_wise = 0.5 * (torch.log(pv) - torch.log(qv) - qv / pv - (qm - pm - yh).pow(2) / pv - 1)
	kl = element_wise.sum(-1)
	#print("log var1", qv)
	return torch.mean(kl)


def adjust_learning_rate(base_lr, power, optimizer, epoch, total_epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * power

def sq_difference_from_mean(data, class_mean, num_class):
    """ Calculates the squared difference from clas mean.
    """
    sq_diff_list = []
    for i in range(num_class):
        sq_diff_list.append(nn.MSELoss(data, class_mean[i]))

    return torch.stack(sq_diff_list)


def class_mean(data, label, num_class):
    class_mean = []
    for i in range(num_class):
        class_mean.append((data[label == i]).mean(0))
    return torch.stack(class_mean)

def intra_spread(data, label, num_class, class_mean):
    intra_diff = 0

    for j in range(data.shape[0]):
        for i in range(num_class):
            if label[j] == i:
                intra_diff += ((data - class_mean[i]).pow(2)).sum().sqrt()
                break
    return intra_diff / data.shape[0]

def all_pair_distance(A):
    r = torch.sum(A*A, 1)

    # turn r into column vector
    r = torch.reshape(r, [-1, 1])
    D = r - 2*torch.matmul(A, A.T) + r.T
    return D

def inter_spread(class_mean):
    ap_dist = all_pair_distance(class_mean)
    dim = class_mean.shape[0]
    not_diag_mask = torch.logical_not(torch.eye(dim) == 0)
    inter_separation = torch.min(ap_dist[not_diag_mask])

    return inter_separation
    # diff_list = []
    # for i in range(class_mean.shape[0]):
    #     for j in range(i,class_mean.shape[0]):
    #         diff_list.append((class_mean[i]-class_mean[j]).pow(2).sum().sqrt())
    # return np.min(diff_list)

def cal_ii_loss(data, label, num_class):
    c_mean = class_mean(data, label, num_class)
    # print("1111111111111111111111111111111")
    # print(c_mean)
    intra_diff = intra_spread(data, label, num_class, c_mean)
    # print("2222222222222222222222222222222")
    # print(intra_diff)
    inter_diff = inter_spread(c_mean)
    # print("3333333333333333333333333333333")
    print(inter_diff)
    loss = intra_diff - inter_diff

    return loss




parser = argparse.ArgumentParser(description='PyTorch OSR Example')
parser.add_argument('--batch_size', type=int, default=None, help='input batch size for training (default: 64)')
parser.add_argument('--num_classes', type=int, default=None, help='number of classes')
parser.add_argument('--nepochs', type=int, default=None, help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=None, help='learning rate (default: 1e-3)')
parser.add_argument('--load_path', type=str, default=None)
parser.add_argument('--load_epoch', type=str, default=None)
parser.add_argument('--layers', type=int, default=None)
parser.add_argument('--wce', type=float, default=1)
parser.add_argument('--wre', type=float, default=1)
parser.add_argument('--wkl', type=float, default=1)
parser.add_argument('--wii', type=float, default=1)
parser.add_argument('--ladder', type=int, default=0)
parser.add_argument('--val', type=str, default="cifar10")
parser.add_argument('--weight_decay', type=float, default=0.0005)
args = parser.parse_args()


def main():
    config.wce = args.wce
    config.wre = args.wre
    config.wkl = args.wkl
    config.wii = args.wii
    config.val = args.val
    config.weight_decay = args.weight_decay
    if args.ladder == 1:
        config.ladder = True
    else:
        config.ladder = False
    # config.ladder = False

    if args.lr != None:
        config.lr = args.lr
    if args.batch_size != None:
        config.batch_size = args.batch_size
        config.niters_per_epoch = config.num_train_imgs // 2 // config.batch_size
    if args.nepochs != None:
        config.nepochs = args.nepochs
    if args.load_path != None:
        config.load_path = args.load_path
    if args.load_epoch != None:
        config.load_epoch - args.load_epoch
    if args.layers != None:
        config.layers = args.layers

    create_exp_dir(config.save, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh'))
    logger = SummaryWriter(config.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info("args = %s", str(config))
    # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


    criterion = nn.NLLLoss()


    # train_dataset = datasets.MNIST('data/mnist', download=True, train=True,
    #                                transform=transforms.Compose([
    #
    #                                    transforms.ToTensor(),
    #                                    transforms.Resize(64),
    #                                    transforms.Normalize((0.5,), (0.5,)),
    #                                transforms.Lambda(lambda x: x.repeat(3, 1, 1) )]))

    # train_dataset = datasets.SVHN('data/svhn', download=True, split="train",
    #                               transform=transforms.Compose([
    #                                   transforms.ToTensor(),
    #                                   transforms.Resize(64)
    #                                   # transforms.Normalize((0.5,),(0.5,))
    #                                   # transforms.Lambda(lambda x: x.repeat(3,1,1))
    #                               ]))
    train_dataset = datasets.CIFAR10('data/cifar10', download=False, train=True,
                                     transform=transforms.Compose([

                                         transforms.ToTensor(),
                                         transforms.Resize(64),
                                         transforms.ColorJitter(hue=.05, saturation=.05),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
                                         transforms.Normalize((0.5,), (0.5,))
                                     ]))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)


    if args.val == "cifar10":
        val_dataset = datasets.CIFAR10('data/cifar10', download=False, train=False,
                                       transform=transforms.Compose([

                                           transforms.ToTensor(),
                                           transforms.Resize(64),
                                           # transforms.ColorJitter(hue=.05, saturation=.05),
                                           # transforms.RandomHorizontalFlip(),
                                           # transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
                                           transforms.Normalize((0.5,), (0.5,))
                                       ]))
    # val_dataset = datasets.MNIST('data/mnist', download=False, train=False,
    #                               transform=transforms.Compose([
    #
    #                                   transforms.ToTensor(),
    #                                   transforms.Resize(64),
    #                                   transforms.Normalize((0.5,), (0.5,)),
    #                               transforms.Lambda(lambda x: x.repeat(3, 1, 1) )]))
    # val_dataset = datasets.SVHN('data/svhn', download=False, split="test",
    #                               transform=transforms.Compose([
    #                                   transforms.ToTensor(),
    #                                   transforms.Resize(64)
    #                                   # transforms.Normalize((0.5,), (0.5,))
    #                                   # transforms.Lambda(lambda x: x.repeat(3,1,1))
    #                               ]))
    elif args.val == "cifar100":
        val_dataset = datasets.CIFAR100('data/cifar100', download=False, train=False,
                                   transform=transforms.Compose([

                                       transforms.ToTensor(),
                                       transforms.Resize(64),
                                       transforms.Normalize((0.5,), (0.5,))
                                   ]))
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)


    # Model #######################################
    models = []
    evaluators = []
    testers = []
    lasts = []
    for idx, arch_idx in enumerate(config.arch_idx):
        if config.load_epoch == "last":
            state = torch.load(os.path.join(config.load_path, "arch_%d.pt"%arch_idx))
        else:
            state = torch.load(os.path.join(config.load_path, "arch_%d_%d.pt"%(arch_idx, int(config.load_epoch))))

        model = Network(
            [state["alpha_%d_0"%arch_idx].detach(), state["alpha_%d_1"%arch_idx].detach(), state["alpha_%d_2"%arch_idx].detach()],
            [None, state["beta_%d_1"%arch_idx].detach(), state["beta_%d_2"%arch_idx].detach()],
            [state["ratio_%d_0"%arch_idx].detach(), state["ratio_%d_1"%arch_idx].detach(), state["ratio_%d_2"%arch_idx].detach()],
            num_classes=config.num_classes, layers=config.layers, Fch=config.Fch, width_mult_list=config.width_mult_list,
            stem_head_width=config.stem_head_width[idx], in_channel=3)

        last = [2,1,0]
        lasts.append(last)
        model.build_structure(last,config.ladder)
        logging.info("net: " + str(model))
        for b in last:
            if len(config.width_mult_list) > 1:
                plot_op(getattr(model, "ops%d"%b), getattr(model, "path%d"%b), width=getattr(model, "widths%d"%b), head_width=config.stem_head_width[idx][1], F_base=config.Fch).savefig(os.path.join(config.save, "ops_%d_%d.png"%(arch_idx,b)), bbox_inches="tight")
            else:
                plot_op(getattr(model, "ops%d"%b), getattr(model, "path%d"%b), F_base=config.Fch).savefig(os.path.join(config.save, "ops_%d_%d.png"%(arch_idx,b)), bbox_inches="tight")
        plot_path_width(model.lasts, model.paths, model.widths).savefig(os.path.join(config.save, "path_width%d.png"%arch_idx))
        plot_path_width([2, 1, 0], [model.path2, model.path1, model.path0], [model.widths2, model.widths1, model.widths0]).savefig(os.path.join(config.save, "path_width_all%d.png"%arch_idx))
        # flops, params = profile(model, inputs=(torch.randn(1, 3, 1024, 2048),), verbose=False)
        # logging.info("params = %fMB, FLOPs = %fGB", params / 1e6, flops / 1e9)
        logging.info("ops:" + str(model.ops))
        logging.info("path:" + str(model.paths))
        logging.info("last:" + str(model.lasts))
        model = nn.DataParallel(model)
        model = model.cuda()


        # init_weight(model.module, nn.init.kaiming_normal_, torch.nn.BatchNorm2d, config.bn_eps, config.bn_momentum, mode='fan_in', nonlinearity='relu')

        if arch_idx == 0 and len(config.arch_idx) > 1:
            partial = torch.load(os.path.join(config.teacher_path, "weights%d.pt"%arch_idx))
            state = model.state_dict()
            pretrained_dict = {k: v for k, v in partial.items() if k in state}
            state.update(pretrained_dict)
            model.load_state_dict(state)
        elif config.is_eval:
            # partial = torch.load(os.path.join(config.eval_path, "weights%d.pt"%arch_idx))
            # print(partial.keys())
            # state1 = model.state_dict()
            # print(state1.keys())
            # pretrained_dict = {k: v for k, v in partial.items() if k in state}
            # state.update(pretrained_dict)
            # model.module.load_state_dict(state)
            model.load_state_dict(torch.load(os.path.join(config.eval_path, "weights0.pt")))
            # state2 = model.state_dict()
            # print(list(set(partial.keys()) - set(state2.keys())))
            # print(state2)
        # print(model)


        # Optimizer ###################################
        base_lr = config.lr
        if arch_idx == 1 or len(config.arch_idx) == 1:
            # optimize teacher solo OR student (w. distill from teacher)
            # optimizer = torch.optim.SGD(model.module.parameters(), lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
            optimizer = torch.optim.Adam(model.module.parameters(), lr=base_lr, weight_decay=config.weight_decay)
        # print("--------------------------------------------------------------------")
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name)
        # print("--------------------------------------------------------------------")
        models.append(model)
    # return None


    if config.is_train:
        tbar = tqdm(range(config.nepochs), ncols=80)
        for epoch in tbar:
            logging.info(config.load_path)
            logging.info(config.save)
            logging.info("lr: " + str(optimizer.param_groups[0]['lr']))
            # training
            tbar.set_description("[Epoch %d/%d][train...]" % (epoch + 1, config.nepochs))
            train_Accs = train(train_loader, models, criterion, optimizer, logger, epoch)
            torch.cuda.empty_cache()
            for idx, arch_idx in enumerate(config.arch_idx):
                if arch_idx == 0:
                    # logger.add_scalar("Acc/train_8", train_Accs[idx][0], epoch)
                    # logging.info("layer8 train_Acc %.3f"%(train_Accs[idx][0]))
                    # logger.add_scalar("Acc/train_16", train_Accs[idx][1], epoch)
                    # logging.info("layer16 train_Acc %.3f" % (train_Accs[idx][1]))
                    logger.add_scalar("Acc/train_32", train_Accs[idx][2], epoch)
                    logging.info("layer32 train_Acc %.3f" % (train_Accs[idx][2]))
                    logger.add_scalar("Acc/train_final", train_Accs[idx][3], epoch)
                    logging.info("final train_Acc %.3f" % (train_Accs[idx][3]))
                    # logger.add_scalar("AUROC/train_final", train_Accs[idx][4], epoch)
                    # logging.info("final train_AUROC %.3f" % (train_Accs[idx][4]))
                else:
                    logger.add_scalar("mIoU/train_student", train_mIoUs[idx], epoch)
                    logging.info("student's train_mIoU %.3f"%(train_mIoUs[idx]))
            adjust_learning_rate(base_lr, 0.992, optimizer, epoch+1, config.nepochs)

            # validation
            if not config.is_test and ((epoch+1) % 1 == 0 or epoch == 0):
                tbar.set_description("[Epoch %d/%d][validation...]" % (epoch + 1, config.nepochs))
                with torch.no_grad():
                    acc8, acc16, acc32, acc_final = infer(models[0], val_loader, epoch=epoch, logger=logger)
                    for idx, arch_idx in enumerate(config.arch_idx):
                        if arch_idx == 0:
                            # logger.add_scalar("Val_Acc/val_8", acc8, epoch)
                            # logging.info("layer8 val_Acc %.3f" % (acc8))
                            # logger.add_scalar("Val_Acc/val_16", acc16, epoch)
                            # logging.info("layer16 val_Acc %.3f" % (acc16))
                            logger.add_scalar("Val_Acc/val_32", acc32, epoch)
                            logging.info("layer32 val_Acc %.3f" % (acc32))
                            logger.add_scalar("Val_Acc/val_final", acc_final, epoch)
                            logging.info("final val_Acc %.3f" % (acc_final))
                            # logger.add_scalar("Val_AUROC/val_final", auroc, epoch)
                            # logging.info("final val_AUROC %.3f" % (auroc))
                        else:
                            logger.add_scalar("mIoU/val_student", valid_mIoUs[idx], epoch)
                            logging.info("student's valid_mIoU %.3f"%(valid_mIoUs[idx]))
                        save(models[idx], os.path.join(config.save, "weights%d.pt"%arch_idx))

    if config.is_eval:

        write_features(model, train_loader, config, part="train", dataset="cifar10")
        write_features(model, val_loader, config, part="val", dataset="cifar100")

    for idx, arch_idx in enumerate(config.arch_idx):
        save(models[idx], os.path.join(config.save, "weights%d.pt"%arch_idx))


def train(train_loader, models, criterion, optimizer, logger, epoch):
    if len(models) == 1:
        # train teacher solo
        models[0].train()
    else:
        # train student (w. distill from teacher)
        models[0].eval()
        models[1].train()

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format, ncols=80)
    dataloader = iter(train_loader)

    metrics = [ seg_metrics.Cls_Metrics(n_classes=config.num_classes) for _ in range(len(models)) ]
    lamb = 0.2
    img_index = 1

    for step in pbar:
        optimizer.zero_grad()

        minibatch = dataloader.next()
        imgs = minibatch[0]
        target = minibatch[1]
        target_en = torch.Tensor(target.shape[0], config.num_classes)
        target_en.zero_()
        target_en.scatter_(1, target.view(-1, 1), 1)  # one-hot encoding
        target_en = target_en.cuda(non_blocking=True)
        imgs = imgs.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        logits_list = []

        description = ""
        ce_loss = 0
        re_loss = 0
        kl_loss = 0
        ii_loss = 0
        for idx, arch_idx in enumerate(config.arch_idx):
            model = models[idx]
            if arch_idx == 0 and len(models) > 1:
                with torch.no_grad():
                    logits8 = model(imgs, target_en, ladder = config.ladder)
                    logits_list.append(logits8)
            else:
                logits8, logits16, logits32, logits_final, reconstructed,\
                    latent_mu, latent_var, yh, up32, up16 = model(imgs, target_en, ladder = config.ladder)
                # logits32 = model(imgs, target_en, ladder = config.ladder)
                ce_loss = ce_loss + criterion(logits32, target)
                re_loss = re_loss + reconstruction_function(reconstructed, imgs)
                # ii_loss = ii_loss + cal_ii_loss(sample_gaussian(latent_mu[0],latent_var[0]),target,config.num_classes)
                if up32[0] != None:
                    pm, pv = torch.zeros(latent_mu[0].shape).cuda(), torch.ones(latent_var[0].shape).cuda()
                    kl32 = kl_normal(latent_mu[0], latent_var[0], pm, pv, yh[0])
                    kl16 = kl_normal(up32[0],up32[1],latent_mu[1],latent_var[1],0)
                    kl8 = kl_normal(up16[0], up16[1], latent_mu[2], latent_var[2], 0)
                    kl_loss = kl_loss + torch.mean(kl32 + kl16 + kl8)
                else:
                    for i in range(3):
                        pm, pv = torch.zeros(latent_mu[i].shape).cuda(), torch.ones(latent_var[i].shape).cuda()
                        kl_loss = kl_loss + kl_normal(latent_mu[i], latent_var[i], pm, pv, yh[i])

            re = torch.Tensor.cpu(reconstructed).detach().numpy()
            ori = torch.Tensor.cpu(imgs).detach().numpy()
            temp = re[0]
            temp = temp * 0.5 + 0.5
            temp = temp * 255
            temp = temp.transpose(1, 2, 0)
            temp = temp.astype(np.uint8)
            img = Image.fromarray(temp)
            img.save(os.path.join("train_img", "{}.jpeg".format(img_index)))

            ori = ori[0]
            ori = ori.transpose(1, 2, 0)
            ori = ori * 0.5 + 0.5
            ori = ori * 255
            ori = ori.astype(np.uint8)
            ori = Image.fromarray(ori)
            ori.save(os.path.join("train_img", "{}_ori.jpeg".format(img_index)))
            img_index += 1


            metrics[idx].update(logits8.data, logits16.data, logits32.data, logits_final.data, target)
            # metrics[idx].update(None,None,logits32.data,None,target)
            # description += "[Acc%d_8: %.3f]"%(arch_idx, metrics[idx].get_scores()[0])
            # description += "[Acc%d_16: %.3f]" % (arch_idx, metrics[idx].get_scores()[1])
            description += "[Acc%d_32: %.3f]" % (arch_idx, metrics[idx].get_scores()[2])
            description += "[Acc%d_final: %.3f]" % (arch_idx, metrics[idx].get_scores()[3])
            # description += "[AUROC_final: %.3f]" % (arch_idx, metrics[idx].get_scores()[4])

        pbar.set_description("[Step %d/%d]"%(step + 1, len(train_loader)) + description)
        logger.add_scalar('train/ce_loss', ce_loss, epoch * len(pbar) + step)
        logger.add_scalar('train/re_loss', re_loss, epoch * len(pbar) + step)
        logger.add_scalar('train/kl_loss', kl_loss, epoch * len(pbar) + step)
        logger.add_scalar('train/ii_loss', ii_loss, epoch * len(pbar) + step)

        loss = config.wce * ce_loss + config.wre * re_loss + config.wkl * kl_loss + config.wii * ii_loss

        # print(torch.Tensor.cpu(kl_loss).detach().numpy())
        ce_loss.backward()
        optimizer.step()
        # logger.add_graph(model.module,(imgs, target_en))
    return [ metric.get_scores() for metric in metrics ]


def infer(model, val_loader, device=torch.device("cuda"), epoch= 0, logger = None):
    model.eval()
    metrics = seg_metrics.Cls_Metrics()
    ce_loss = 0
    re_loss = 0
    kl_loss = 0
    ii_loss = 0
    lamb = 0.2
    total_num = 0
    for data_val, target_val in val_loader:
        # print("Current working on {} batch".format(i))
        total_num += len(target_val)
        target_val_en = torch.Tensor(target_val.shape[0], config.num_classes)
        target_val_en.zero_()
        target_val_en.scatter_(1, target_val.view(-1, 1), 1)  # one-hot encoding
        target_val_en = target_val_en.to(device)
        data_val, target_val = data_val.to(device), target_val.to(device)
        logits8, logits16, logits32, logits_final, reconstructed,\
            latent_mu, latent_var, yh, up32, up16 = model(data_val, target_val_en, ladder = config.ladder)
        # logits32 = model(data_val, target_val_en, ladder = config.ladder)
        metrics.update(logits8, logits16, logits32, logits_final, target_val)
        # metrics.update(None,None,logits32.data,None,target_val)
        ce_loss = ce_loss + criterion(logits32, target_val)
        re_loss = re_loss + reconstruction_function(reconstructed, data_val)
        # ii_loss = ii_loss + cal_ii_loss(sample_gaussian(latent_mu[0], latent_var[0]), target_val, config.num_classes)
        if up32[0] != None:
            pm, pv = torch.zeros(latent_mu[0].shape).cuda(), torch.ones(latent_var[0].shape).cuda()
            kl32 = kl_normal(latent_mu[0], latent_var[0], pm, pv, yh[0])
            kl16 = kl_normal(up32[0], up32[1], latent_mu[1], latent_var[1], 0)
            kl8 = kl_normal(up16[0], up16[1], latent_mu[2], latent_var[2], 0)
            kl_loss = kl_loss + torch.mean(kl32 + kl16 + kl8)
        else:
            for i in range(3):
                pm, pv = torch.zeros(latent_mu[i].shape).cuda(), torch.ones(latent_var[i].shape).cuda()
                kl_loss = kl_loss + kl_normal(latent_mu[i], latent_var[i], pm, pv, yh[i])

    # ce_loss = ce_loss
    # re_loss =re_loss
    # kl_loss = kl_loss
    logger.add_scalar('val/ce_loss', ce_loss, epoch)
    logger.add_scalar('val/re_loss', re_loss, epoch)
    logger.add_scalar('val/kl_loss', kl_loss, epoch)
    logger.add_scalar('val/ii_loss', ii_loss, epoch)
    return metrics.get_scores()


def write_features(model, train_loader, config, dataset="cifar10", part="train",  device=torch.device("cuda")):

    open('{}_fea/{}_mu32.txt'.format(part, dataset), 'w').close()
    open('{}_fea/{}_mu16.txt'.format(part, dataset), 'w').close()
    open('{}_fea/{}_mu8.txt'.format(part, dataset), 'w').close()
    open('{}_fea/{}_target.txt'.format(part, dataset), 'w').close()
    open('{}_fea/{}_logit.txt'.format(part, dataset), 'w').close()
    open('{}_fea/{}_pred.txt'.format(part, dataset), 'w').close()
    open('{}_fea/{}_re_loss.txt'.format(part, dataset), 'w').close()

    model.eval()

    img_index = 1
    img_dir = part + "_img"
    for data, target in train_loader:
        target_en = torch.Tensor(target.shape[0], config.num_classes)
        target_en.zero_()
        # target_en.scatter_(1, target.view(-1, 1), 1)  # one-hot encoding
        target_en = target_en.to(device)
        data, target = data.to(device), target.to(device)
        pred8, pred16, pred_32, pred_final, reconstructed, \
            latent_mu, latent_var, yh, _, _ = model(data, target_en, ladder = config.ladder)
        # print(reconstructed.shape)
        # make_dot(reconstructed, params = dict(list(model.module.named_parameters()))).render("model", format="png")
        # break
        re_loss = ((reconstructed - data)**2).view(config.batch_size, -1).mean(1)
        # print(re_loss.shape)
        # print(re_loss.shape)
        re = torch.Tensor.cpu(reconstructed).detach().numpy()
        ori = torch.Tensor.cpu(data).detach().numpy()
        # print(re.shape)

        temp = re[0]
        temp = temp * 0.5+0.5
        temp = temp * 255
        temp = temp.transpose(1,2,0)
        temp = temp.astype(np.uint8)
        img = Image.fromarray(temp, 'RGB')
        img.save(os.path.join(img_dir,"{}.jpeg".format(img_index)))

        ori = ori[0]
        ori = ori.transpose(1,2,0)
        ori = ori * 0.5 + 0.5
        ori = ori * 255
        ori = ori.astype(np.uint8)
        ori = Image.fromarray(ori)
        ori.save(os.path.join(img_dir,"{}_ori.jpeg".format(img_index)))
        img_index += 1
        # break

        pred = pred_32.max(1, keepdim=True)[1]
        # print(pred)
        pred_final = torch.Tensor.cpu(pred_32).detach().numpy()
        latent_mu32 = torch.Tensor.cpu(latent_mu[0]).detach().numpy()
        latent_mu16 = torch.Tensor.cpu(latent_mu[1]).detach().numpy()
        latent_mu8 = torch.Tensor.cpu(latent_mu[2]).detach().numpy()
        re_loss = torch.Tensor.cpu(re_loss).detach().numpy()
        target = torch.Tensor.cpu(target).detach().numpy()
        pred = torch.Tensor.cpu(pred).detach().numpy()
        if config.val != "cifar10" and part=="val":
            target = np.full(target.shape, config.num_classes)
        with open('{}_fea/{}_mu32.txt'.format(part, dataset), 'ab') as f_test:
            np.savetxt(f_test, latent_mu32, fmt='%f', delimiter=' ', newline='\r')
            f_test.write(b'\n')
        with open('{}_fea/{}_mu16.txt'.format(part, dataset), 'ab') as f_test:
            np.savetxt(f_test, latent_mu16, fmt='%f', delimiter=' ', newline='\r')
            f_test.write(b'\n')
        with open('{}_fea/{}_mu8.txt'.format(part, dataset), 'ab') as f_test:
            np.savetxt(f_test, latent_mu8, fmt='%f', delimiter=' ', newline='\r')
            f_test.write(b'\n')
        with open('{}_fea/{}_target.txt'.format(part, dataset), 'ab') as f_test:
            np.savetxt(f_test, target, fmt='%f', delimiter=' ', newline='\r')
            f_test.write(b'\n')
        with open('{}_fea/{}_logit.txt'.format(part, dataset), 'ab') as f_test:
            np.savetxt(f_test, pred_final, fmt='%f', delimiter=' ', newline='\r')
            f_test.write(b'\n')
        with open('{}_fea/{}_pred.txt'.format(part, dataset), 'ab') as f_test:
            np.savetxt(f_test, pred, fmt='%f', delimiter=' ', newline='\r')
            f_test.write(b'\n')
        with open('{}_fea/{}_re_loss.txt'.format(part, dataset), 'ab') as f_test:
            np.savetxt(f_test, re_loss, fmt='%f', delimiter=' ', newline='\r')
            f_test.write(b'\n')



def test(epoch, models, testers, logger):
    for idx, arch_idx in enumerate(config.arch_idx):
        if arch_idx == 0: continue
        model = models[idx]
        tester = testers[idx]
        os.system("mkdir %s"%os.path.join(os.path.join(os.path.realpath('.'), config.save, "test")))
        model.eval()
        tester.run_online()
        os.system("mv %s %s"%(os.path.join(os.path.realpath('.'), config.save, "test"), os.path.join(os.path.realpath('.'), config.save, "test_%d_%d"%(arch_idx, epoch))))


if __name__ == '__main__':
    main() 
