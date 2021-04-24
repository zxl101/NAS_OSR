from __future__ import division
import os
import sys
import time
import glob
import logging
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.utils
from tensorboardX import SummaryWriter
from torch.autograd import Variable
# from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from thop import profile

from config_search import config
from dataloader import MNIST_Dataset, CIFAR10_Dataset, SVHN_Dataset, CIFARAdd10_Dataset, CIFARAdd50_Dataset, CIFARAddN_Dataset, CIFAR100_Dataset, TinyImageNet_Dataset


from utils.init_func import init_weight
# from seg_opr.loss_opr import ProbOhemCrossEntropy2d
# from eval import SegEvaluator

from architect import Architect
from utils.darts_utils import create_exp_dir, save, plot_op, plot_path_width, objective_acc_lat
from model_search import Network_Multi_Path as Network
from model_seg import Network_Multi_Path_Infer

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from qmv import ocr_test
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
parser.add_argument('--lamda', type=float, default=None)
parser.add_argument('--beta', type=float, default=None)
parser.add_argument('--beta_z', type=float, default=None)
parser.add_argument('--seed_sampler', type=int, default=777)
parser.add_argument('--dataset', type=str, default="CIFAR10")
parser.add_argument('--use_model', action="store_true", default=False, help='If use model to get the train feature')
parser.add_argument('--cf', action="store_true", default=False, help='use counterfactual generation')
parser.add_argument('--cf_threshold', action="store_true", default=False, help='use counterfactual threshold in revise_cf')
parser.add_argument('--yh', action="store_true", default=False, help='use yh rather than feature_y_mean')
parser.add_argument('--use_model_gau', action="store_true", default=False, help='use feature by model in gau')
parser.add_argument('--threshold', type=float, default=0.5, help='threshold of gaussian model')
parser.add_argument('--temperature', type=float, default=None)
parser.add_argument('--pretrain_nepochs', type=int, default=None)
parser.add_argument('--pretrain_metric', type=str, default='f1_score')
parser.add_argument('--search_metric', type=str, default='f1_score')
parser.add_argument('--unseen_num', type=int, default=None)
parser.add_argument('--skip_connect', type=float, default=1, help='use skip connection')
args = parser.parse_args()

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def main(pretrain=True):
    if args.temperature != None:
        config.temperature = args.temperature
    if args.lamda != None:
        config.lamda = args.lamda
    if args.beta != None:
        config.beta = args.beta
    if args.beta_z != None:
        config.beta_z = args.beta_z
    if args.weight_decay != None:
        config.weight_decay = args.weight_decay
    if args.batch_size != None:
        config.batch_size = args.batch_size
        config.niters_per_epoch = config.num_train_imgs // 2 // config.batch_size
    if args.num_classes != None:
        config.num_classes = args.num_classes
    if args.nepochs != None:
        config.nepochs = args.nepochs
    if args.lr != None:
        config.lr = args.lr
    if args.layers != None:
        config.layers = args.layers
    if args.pretrain_nepochs != None:
        config.pretrain_nepochs = args.pretrain_nepochs
    if args.unseen_num != None:
        config.unseen_num = args.unseen_num
    config.use_model = args.use_model
    config.cf = args.cf
    config.cf_threshold = args.cf_threshold
    config.yh = args.yh
    config.use_model_gau = args.use_model_gau
    config.threshold = args.threshold
    config.cuda = True
    config.pretrain_metric = args.pretrain_metric
    config.search_metric = args.search_metric
    config.skip_connect = args.skip_connect

    if args.dataset == "MNIST":
        load_dataset = MNIST_Dataset()
        args.num_classes = 4
        in_channel = 1
    elif args.dataset == "CIFAR10":
        load_dataset = CIFAR10_Dataset()
        args.num_classes = 4
        in_channel = 3
    elif args.dataset == "SVHN":
        load_dataset = SVHN_Dataset()
        args.num_classes = 4
        in_channel = 3
    elif args.dataset == "CIFARAdd10":
        load_dataset = CIFARAdd10_Dataset()
        args.num_classes = 4
        in_channel = 3
    elif args.dataset == "CIFARAdd50":
        load_dataset = CIFARAdd50_Dataset()
        args.num_classes = 4
        in_channel = 3
    elif args.dataset == "CIFARAddN":
        load_dataset = CIFARAddN_Dataset()
        args.num_classes = 4
        in_channel = 3
    elif args.dataset == "CIFAR100":
        load_dataset = CIFAR100_Dataset()
        args.num_classes = 15
        in_channel = 3
    elif args.dataset == "TinyImageNet":
        load_dataset = TinyImageNet_Dataset()
        args.num_classes = 70
        in_channel = 3
    config.num_classes = args.num_classes
    config.in_channel = in_channel
    config.img_size = 32
    if args.dataset == "TinyImageNet":
        config.img_size = 64

    # pretrain
    # config.save = "pretrain-%dx%d_F%d.L%d_batch%d_%d_%d_%d_lr%d"%(config.image_height, config.image_width,
    #                                                               config.Fch, config.layers, config.batch_size,
    #                                                               config.lamda,config.beta,config.beta_z,config.lr)
    #
    # config.save = 'search-{}-{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"),random.randint(1000,9999))
    config.save = os.path.join(config.dataset, "pretrain")
    if not os.path.exists(config.save):
        create_exp_dir(config.save, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh'))
    logger = SummaryWriter(config.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    assert type(pretrain) == bool or type(pretrain) == str
    update_arch = False
    logging.info("args = %s", str(config))
    # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Model #######################################
    model = Network(config.num_classes, config.in_channel, config.layers, Fch=config.Fch, width_mult_list=config.width_mult_list,
                    prun_modes=config.prun_modes, stem_head_width=config.stem_head_width, z_dim=config.z_dim,
                    lamda=config.lamda, beta=config.beta, beta_z=config.beta_z, temperature=config.temperature,
                    img_size=config.img_size, skip_connect=config.skip_connect)

    use_cuda = torch.cuda.is_available() and True
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)

    init_weight(model, nn.init.kaiming_normal_, nn.BatchNorm2d, config.bn_eps, config.bn_momentum, mode='fan_in', nonlinearity='relu')

    model = nn.DataParallel(model)
    model.to(device)
    architect = Architect(model, config)

    # Optimizer ###################################
    base_lr = config.lr
    parameters = []
    # parameters += list(model.module.stem.parameters())
    parameters += list(model.module.cells.parameters())
    parameters += list(model.module.down1.parameters())
    parameters += list(model.module.down2.parameters())
    parameters += list(model.module.down4.parameters())
    parameters += list(model.module.dec32.parameters())
    parameters += list(model.module.up32.parameters())
    parameters += list(model.module.up16.parameters())
    parameters += list(model.module.up8.parameters())
    parameters += list(model.module.up4.parameters())
    parameters += list(model.module.up2.parameters())
    parameters += list(model.module.refine1.parameters())
    parameters += list(model.module.mean_layer32.parameters())
    parameters += list(model.module.var_layer32.parameters())
    parameters += list(model.module.classifier.parameters())
    parameters += list(model.module.one_hot32.parameters())
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
    lr_policy = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.970)

    seed_sampler = int(args.seed_sampler)
    train_dataset, val_dataset, pick_dataset, test_dataset = load_dataset.sampler_search(seed_sampler, args)
    if len(train_dataset) % 2 == 0 :
        train_dataset_model, train_dataset_arch = torch.utils.data.random_split(train_dataset,[len(train_dataset)//2, len(train_dataset)//2])
    else:
        train_dataset_model, train_dataset_arch = torch.utils.data.random_split(train_dataset, [len(train_dataset) // 2 + 1,
                                                                                                len(
                                                                                                    train_dataset) // 2])
    train_loader_model = DataLoader(train_dataset_model, batch_size=config.batch_size, shuffle=True, num_workers=0)
    train_loader_arch = DataLoader(train_dataset_arch, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    pick_loader = DataLoader(pick_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    tbar = tqdm(range(config.nepochs), ncols=80)
    best_val_loss = 10000000
    best_f1_score = 10
    best_acc = 0
    best_val_epoch = 0
    select_metric = config.pretrain_metric
    for epoch in tbar:
        if epoch == config.pretrain_nepochs:
            partial = torch.load(config.save + '/weights.pt', map_location='cuda:0')
            state = model.state_dict()
            pretrained_dict = {k: v for k, v in partial.items() if k in state and state[k].size() == partial[k].size()}
            state.update(pretrained_dict)
            model.load_state_dict(state)
            # config.save = "%dx%d_F%d.L%d_batch%d_%d_%d_%d_lr%d" % (config.image_height, config.image_width,
            #                                                        config.Fch, config.layers, config.batch_size,
            #                                                        config.lamda, config.beta, config.beta_z, config.lr)
            # config.save = 'search-{}-{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"),random.randint(1000,9999))
            config.save = os.path.join(config.save, "search")
            if not os.path.exists(config.save):
                create_exp_dir(config.save, scripts_to_save=glob.glob('*.py') + glob.glob('*.sh'))
            logger = SummaryWriter(config.save)

            log_format = '%(asctime)s %(message)s'
            logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
            fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
            fh.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(fh)
            update_arch = True
            pretrain = "111"
            logging.info("args = %s", str(config))
            best_f1_score = 10
            best_val_loss = 10000000
            best_acc = 0
            best_val_epoch = 0
            config.nepochs -= config.pretrain_nepochs
            select_metric = config.search_metric


        if epoch >= config.pretrain_nepochs:
            epoch -= config.pretrain_nepochs

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
                model.module.prun_mode = "max"
                # ce_loss, kl_loss, re_loss, contras_loss = infer(epoch, model, val_loader, logger, FPS=False, config=config)
                # total_val_loss = ce_loss + kl_loss + re_loss + contras_loss
                f1_score, loss, acc = test(model, train_loader_model, val_loader, pick_loader,epoch,logger)
            else:
                valid_losses = []; FPSs = []
                model.prun_mode = None
                for idx in range(len(model.module._arch_names)):
                    # arch_idx
                    model.module.arch_idx = idx

                    # ce_loss, kl_loss, re_loss, contras_loss = infer(epoch, model, val_loader, logger, config=config)
                    # total_val_loss = ce_loss + kl_loss + re_loss + contras_loss
                    f1_score, loss, acc = test(model, train_loader_model, val_loader, pick_loader, epoch, logger)
                # if update_arch:
                #     latency_supernet_history.append(architect.latency_supernet)
                # latency_weight_history.append(architect.latency_weight)
            if epoch % config.val_interval == 0 and epoch >= 0:

                if select_metric == 'loss':
                    if best_val_loss > loss:
                        best_val_loss = loss
                        best_val_epoch = epoch
                        save(model, os.path.join(config.save, 'weights.pt'))
                    else:
                        continue
                elif select_metric == 'acc':
                    if best_acc < acc:
                        best_acc = acc
                        best_val_epoch = epoch
                        save(model, os.path.join(config.save, 'weights.pt'))
                    else:
                        continue
                else:
                    if best_f1_score > f1_score:
                        best_f1_score = f1_score
                        best_val_epoch = epoch
                        save(model, os.path.join(config.save, 'weights.pt'))


        save(model, os.path.join(config.save, 'epoch{}_weights.pt'.format(epoch)))
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

                torch.save(state, os.path.join(config.save, "arch_%d_%d.pt"%(idx, epoch)))
                # torch.save(state, os.path.join(config.save, "arch_%d.pt"%(idx)))
                if best_val_epoch == epoch:
                    torch.save(state, os.path.join(config.save, "arch_%d.pt" % (idx)))


    # open('%s/train_fea.txt' % config.save, 'w').close()  # clear
    # np.savetxt('%s/train_fea.txt' % config.save, train_fea, delimiter=' ', fmt='%f')
    # open('%s/train_tar.txt' % config.save, 'w').close()
    # np.savetxt('%s/train_tar.txt' % config.save, train_tar, delimiter=' ', fmt='%d')
    # open('%s/train_rec.txt' % config.save, 'w').close()
    # np.savetxt('%s/train_rec.txt' % config.save, train_rec, delimiter=' ', fmt='%f')
    #
    # fea_omn = np.loadtxt('%s/test_fea.txt' % config.save)
    # tar_omn = np.loadtxt('%s/test_tar.txt' % config.save)
    # pre_omn = np.loadtxt('%s/test_pre.txt' % config.save)
    # rec_omn = np.loadtxt('%s/test_rec.txt' % config.save)
    #
    # open('%s/test_fea.txt' % config.save, 'w').close()  # clear
    # np.savetxt('%s/test_fea.txt' % config.save, fea_omn, delimiter=' ', fmt='%f')
    # open('%s/test_tar.txt' % config.save, 'w').close()
    # np.savetxt('%s/test_tar.txt' % config.save, tar_omn, delimiter=' ', fmt='%d')
    # open('%s/test_pre.txt' % config.save, 'w').close()
    # np.savetxt('%s/test_pre.txt' % config.save, pre_omn, delimiter=' ', fmt='%d')
    # open('%s/test_rec.txt' % config.save, 'w').close()
    # np.savetxt('%s/test_rec.txt' % config.save, rec_omn, delimiter=' ', fmt='%d')
    print("Best epoch is {}".format(best_val_epoch))
    print("Best loss is {}".format(best_val_loss))

def train(pretrain, train_loader_model, train_loader_arch, model, architect, optimizer, lr_policy, logger, epoch, update_arch=True, config = None, device = None):
    open('%s/train_fea.txt' % config.save, 'w').close()
    open('%s/train_tar.txt' % config.save, 'w').close()
    open('%s/train_rec.txt' % config.save, 'w').close()

    model.train()
    # model.module.latent_space(epoch=epoch, vis=True)
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format, ncols=80)
    dataloader_model = iter(cycle(train_loader_model))
    dataloader_arch = iter(cycle(train_loader_arch))
    epoch_ce_loss = 0
    epoch_re_loss = 0
    epoch_kl_loss = 0
    epoch_contras_loss = 0
    count = 0
    correct_train = 0
    for step in pbar:
        optimizer.zero_grad()

        minibatch = next(dataloader_model)
        imgs = minibatch[0]
        target = minibatch[1]
        target_en = torch.Tensor(target.shape[0], config.num_classes)
        target_en.zero_()
        target_en.scatter_(1, target.view(-1, 1), 1)  # one-hot encoding
        target_en = target_en.to(device)
        imgs = imgs.to(device)
        target = target.to(device)
        imgs, target = Variable(imgs), Variable(target)

        if update_arch:
            # print("Updating architecture")
            # get a random minibatch from the search queue with replacement
            pbar.set_description("[Arch Step %d/%d]" % (step + 1, len(train_loader_model)))
            minibatch = next(dataloader_arch)
            imgs_search = minibatch[0]
            target_search =minibatch[1]
            target_en_search = torch.Tensor(target_search.shape[0], config.num_classes)
            target_en_search.zero_()
            target_en_search.scatter_(1, target_search.view(-1, 1), 1)  # one-hot encoding
            target_en_search = target_en_search.to(device)
            imgs_search = imgs_search.to(device)
            target_search = target_search.to(device)
            imgs_search, target_search = Variable(imgs_search), Variable(target_search)

            loss_arch = architect.step(imgs, target, target_en, imgs_search, target_search, target_en_search)
            if (step+1) % 10 == 0:
                logger.add_scalar('loss_arch/train', loss_arch, epoch*len(pbar)+step)

        ce_loss, kl_loss, re_loss, contras_loss, predict, predict_test, y_mu, x_re = model(imgs, target, target_en, pretrain)

        rec_loss = (x_re - imgs).pow(2).sum((3, 2, 1))

        loss = ce_loss + re_loss + kl_loss + contras_loss
        loss = loss.sum()
        loss.backward()
        nn.utils.clip_grad_norm_(model.module.parameters(), config.grad_clip)
        optimizer.step()

        ce_loss = torch.mean(ce_loss)
        re_loss = torch.mean(re_loss)
        kl_loss = torch.mean(kl_loss)
        contras_loss = torch.mean(contras_loss)
        epoch_ce_loss += ce_loss
        epoch_re_loss += re_loss
        epoch_kl_loss += kl_loss
        epoch_contras_loss += contras_loss
        logger.add_scalar('loss_step/train_ce', ce_loss, epoch*len(pbar)+step)
        logger.add_scalar('loss_step/train_re', re_loss, epoch * len(pbar) + step)
        logger.add_scalar('loss_step/train_kl', kl_loss, epoch * len(pbar) + step)
        logger.add_scalar('loss_step/train_contras', contras_loss, epoch * len(pbar) + step)
        pbar.set_description("[Step %d/%d]" % (step + 1, len(train_loader_model)))
        count += len(target)

        outlabel = predict.data.max(1)[1]  # get the index of the max log-probability
        correct_train += outlabel.eq(target.view_as(outlabel)).sum().item()

        cor_fea = y_mu[(outlabel == target)]
        cor_tar = target[(outlabel == target)]
        cor_fea = torch.Tensor.cpu(cor_fea).detach().numpy()
        cor_tar = torch.Tensor.cpu(cor_tar).detach().numpy()
        rec_loss = torch.Tensor.cpu(rec_loss).detach().numpy()

        with open('%s/train_fea.txt' % config.save, 'ab') as f:
            np.savetxt(f, cor_fea, fmt='%f', delimiter=' ', newline='\r')
            f.write(b'\n')
        with open('%s/train_tar.txt' % config.save, 'ab') as t:
            np.savetxt(t, cor_tar, fmt='%d', delimiter=' ', newline='\r')
            t.write(b'\n')
        with open('%s/train_rec.txt' % config.save, 'ab') as m:
            np.savetxt(m, rec_loss, fmt='%f', delimiter=' ', newline='\r')
            m.write(b'\n')
    train_acc = float(100 * correct_train) / count
    logger.add_scalar('loss_epoch/train_ce', epoch_ce_loss/count, epoch)
    logger.add_scalar('loss_epoch/train_re', epoch_re_loss / count, epoch)
    logger.add_scalar('loss_epoch/train_kl', epoch_kl_loss / count, epoch)
    logger.add_scalar('loss_epoch/train_contras', epoch_contras_loss / count, epoch)
    logger.add_scalar('train_acc/train_acc', train_acc, epoch)
    print('Train_Acc: {}/{} ({:.2f}%)'.format(correct_train, count, train_acc))
    torch.cuda.empty_cache()
    # del loss
    # if update_arch: del loss_arch

def infer(epoch, model, val_loader, logger, FPS=True, config=None, device=torch.device("cuda"), add_scalar = True):
    model.eval()
    total_ce_loss = 0
    total_re_loss = 0
    total_kl_loss = 0
    total_contras_loss = 0
    correct_val = 0
    count = 0
    for data_val, target_val in val_loader:
        # print("Current working on {} batch".format(i))
        target_val_en = torch.Tensor(target_val.shape[0], config.num_classes)
        target_val_en.zero_()
        target_val_en.scatter_(1, target_val.view(-1, 1), 1)  # one-hot encoding
        target_val_en = target_val_en.to(device)
        data_val, target_val = data_val.to(device), target_val.to(device)
        data_val, target_val = Variable(data_val), Variable(target_val)
        count += len(target_val)

        ce_loss, kl_loss, re_loss, contras_loss, predict, predict_test, y_mu, x_re = model(data_val, target_val, target_val_en, config.pretrain)
        ce_loss = torch.mean(ce_loss)
        re_loss = torch.mean(re_loss)
        kl_loss = torch.mean(kl_loss)
        contras_loss = torch.mean(contras_loss)
        total_ce_loss += ce_loss
        total_re_loss += re_loss
        total_kl_loss += kl_loss
        total_contras_loss += contras_loss
    total_val_loss = total_ce_loss + total_kl_loss + total_re_loss + total_contras_loss
    val_loss = torch.mean(total_val_loss) / count
    val_rec = torch.mean(total_re_loss) / count
    val_kl = torch.mean(total_kl_loss) / count
    val_ce = torch.mean(total_ce_loss) / count
    val_contras = total_contras_loss / count
    vallabel = predict.data.max(1)[1]  # get the index of the max log-probability
    correct_val += vallabel.eq(target_val.view_as(vallabel)).sum().item()
    val_acc = float(100 * correct_val) / count
    print("The validation ce loss is: {}".format(total_ce_loss))
    print("The validation re loss is: {}".format(total_re_loss))
    print("The validation kl loss is: {}".format(total_kl_loss))
    print("The validation contras loss is: {}".format(total_contras_loss))
    print("The validation total loss is: {}".format(total_val_loss))
    if add_scalar == True:
        logger.add_scalar("val_loss/val_loss_all", val_loss, epoch)
        logger.add_scalar("val_loss/val_rec_loss", val_rec, epoch)
        logger.add_scalar("val_loss/val_kl_loss", val_kl, epoch)
        logger.add_scalar("val_loss/val_ce_loss", val_ce, epoch)
        logger.add_scalar("val_loss/val_contras_loss", val_contras, epoch)
        logger.add_scalar("Acc_val", val_acc, epoch)


    # if FPS:
    #     fps0, fps1 = arch_logging(model, config, logger, epoch)
    #     return total_ce_loss, fps0, fps1
    # else:
    #     return total_ce_loss, total_re_loss, total_kl_loss
    return total_ce_loss, total_re_loss, total_kl_loss, total_contras_loss

def test(model, train_loader, val_loader, test_loader, epoch, logger, device=torch.device('cuda')):
    open('%s/test_fea.txt' % config.save, 'w').close()
    open('%s/test_tar.txt' % config.save, 'w').close()
    open('%s/test_pre.txt' % config.save, 'w').close()
    open('%s/test_rec.txt' % config.save, 'w').close()

    model.eval()
    total_ce_loss = 0
    total_re_loss = 0
    total_kl_loss = 0
    total_contras_loss = 0
    correct_val = 0
    total_num = 0

    for data_val, target_val in val_loader:
        target_val_en = torch.Tensor(target_val.shape[0], config.num_classes)
        target_val_en.zero_()
        target_val_en.scatter_(1, target_val.view(-1, 1), 1)  # one-hot encoding
        target_val_en = target_val_en.to(device)
        data_val, target_val = data_val.cuda(), target_val.cuda()
        with torch.no_grad():
            data_val, target_val = Variable(data_val), Variable(target_val)

        ce_loss, kl_loss, re_loss, contras_loss, predict, output_val, mu_val, de_val = model(data_val, target_val, target_val_en, config.pretrain)

        total_num += len(target_val)

        total_re_loss += torch.mean(re_loss)
        total_ce_loss += torch.mean(ce_loss)
        total_kl_loss += torch.mean(kl_loss)
        total_contras_loss += torch.mean(contras_loss)

        vallabel = predict.data.max(1)[1]  # get the index of the max log-probability
        correct_val += vallabel.eq(target_val.view_as(vallabel)).sum().item()

        output_val = torch.exp(output_val)
        prob_val = output_val.max(1)[0]  # get the value of the max probability
        pre_val = output_val.max(1, keepdim=True)[1]  # get the index of the max log-probability
        rec_val = (de_val - data_val).pow(2).sum((3, 2, 1))
        # _, mu_val = torch.split(mu_val, [config.z_dim, config.latent_dim32], dim=1)
        mu_val = torch.Tensor.cpu(mu_val).detach().numpy()
        target_val_np = torch.Tensor.cpu(target_val).detach().numpy()
        pre_val = torch.Tensor.cpu(pre_val).detach().numpy()
        rec_val = torch.Tensor.cpu(rec_val).detach().numpy()

        with open('%s/test_fea.txt' % config.save, 'ab') as f_val:
            np.savetxt(f_val, mu_val, fmt='%f', delimiter=' ', newline='\r')
            f_val.write(b'\n')
        with open('%s/test_tar.txt' % config.save, 'ab') as t_val:
            np.savetxt(t_val, target_val_np, fmt='%d', delimiter=' ', newline='\r')
            t_val.write(b'\n')
        with open('%s/test_pre.txt' % config.save, 'ab') as p_val:
            np.savetxt(p_val, pre_val, fmt='%d', delimiter=' ', newline='\r')
            p_val.write(b'\n')
        with open('%s/test_rec.txt' % config.save, 'ab') as l_val:
            np.savetxt(l_val, rec_val, fmt='%f', delimiter=' ', newline='\r')
            l_val.write(b'\n')
    total_re_loss = torch.mean(total_re_loss)
    total_kl_loss = torch.mean(total_kl_loss)
    total_ce_loss = torch.mean(total_ce_loss)
    total_contras_loss = torch.mean(total_contras_loss) 
    total_val_loss = total_ce_loss + total_kl_loss + total_re_loss + total_contras_loss
    val_loss = total_val_loss / total_num
    val_rec = total_re_loss / total_num
    val_kl = total_kl_loss / total_num
    val_ce = total_ce_loss / total_num
    val_contras = total_contras_loss / total_num
    val_acc = float(100 * correct_val) / total_num
    print("The validation ce loss is: {}".format(total_ce_loss))
    print("The validation re loss is: {}".format(total_re_loss))
    print("The validation kl loss is: {}".format(total_kl_loss / config.beta))
    print("The validation contras loss is: {}".format(total_contras_loss))
    print("The validation total loss is: {}".format(total_val_loss))
    if True:
        logger.add_scalar("val_loss/val_loss_all", val_loss, epoch)
        logger.add_scalar("val_loss/val_rec_loss", val_rec, epoch)
        logger.add_scalar("val_loss/val_kl_loss", val_kl / config.beta, epoch)
        logger.add_scalar("val_loss/val_ce_loss", val_ce, epoch)
        logger.add_scalar("val_loss/val_contras_loss", val_contras, epoch)
        logger.add_scalar("Acc_val", val_acc, epoch)

    # img_index = 1
    for data_omn, target_omn in test_loader:
        tar_omn = torch.from_numpy(config.num_classes * np.ones(target_omn.shape[0]))
        data_omn = data_omn.cuda()
        with torch.no_grad():
            data_omn = Variable(data_omn)
        # print(data_omn.shape)
        output_omn, mu_omn, de_omn = model.module.test(data_omn, target_val, target_val_en)
        output_omn = torch.exp(output_omn)
        # prob_omn = output_omn.max(1)[0]  # get the value of the max probability
        pre_omn = output_omn.max(1, keepdim=True)[1]  # get the index of the max log-probability
        rec_omn = (de_omn - data_omn).pow(2).sum((3, 2, 1))
        mu_omn = torch.Tensor.cpu(mu_omn).detach().numpy()
        tar_omn = torch.Tensor.cpu(tar_omn).detach().numpy()
        pre_omn = torch.Tensor.cpu(pre_omn).detach().numpy()
        rec_omn = torch.Tensor.cpu(rec_omn).detach().numpy()

        with open('%s/test_fea.txt' % config.save, 'ab') as f_test:
            np.savetxt(f_test, mu_omn, fmt='%f', delimiter=' ', newline='\r')
            f_test.write(b'\n')
        with open('%s/test_tar.txt' % config.save, 'ab') as t_test:
            np.savetxt(t_test, tar_omn, fmt='%d', delimiter=' ', newline='\r')
            t_test.write(b'\n')
        with open('%s/test_pre.txt' % config.save, 'ab') as p_test:
            np.savetxt(p_test, pre_omn, fmt='%d', delimiter=' ', newline='\r')
            p_test.write(b'\n')
        with open('%s/test_rec.txt' % config.save, 'ab') as l_test:
            np.savetxt(l_test, rec_omn, fmt='%f', delimiter=' ', newline='\r')
            l_test.write(b'\n')

    perf = ocr_test(config, model, train_loader, val_loader, test_loader)
    print("The f1 score is {}".format(perf[-1]))
    logger.add_scalar("val_perf/val_f1", perf[-1], epoch)
    return perf[-1], total_val_loss, val_acc


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
    if len(preds) == 1:
        preds = preds[0].data.max(1)[1]
        c[0] += preds.eq(target.view_as(preds)).sum().item()
    else:
        for pred in preds:
            pred = pred.data.max(1)[1]
            c[i] += pred.eq(target.view_as(pred)).sum().item()
            i += 1
    return c, len(target)

if __name__ == '__main__':
    main(pretrain=config.pretrain) 
