from __future__ import division
import os
import sys
import time
import glob
import logging
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score

import numpy as np
from thop import profile
import random
from config_train import config
if config.is_eval:
    config.save = 'eval-{}-{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"),random.randint(1000,9999))
else:
    config.save = 'train-{}-{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"), random.randint(1000,9999))

from dataloader import MNIST_Dataset, CIFAR10_Dataset, SVHN_Dataset, CIFARAdd10_Dataset, CIFARAdd50_Dataset, CIFARAddN_Dataset, CIFAR100_Dataset, TinyImageNet_Dataset


from utils.darts_utils import create_exp_dir, save, plot_op, plot_path_width, objective_acc_lat
from utils.init_func import init_weight
from model_seg import Network_Multi_Path_Infer as Network

from torch.utils.data import DataLoader
from qmv import ocr_test, auroc_cal
from PIL import Image
import argparse
import shutil

class DeterministicWarmup(object):
    def __init__(self, n=100, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.inc = self.t_max / n

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc

        self.t = self.t_max if t > self.t_max else t  # 0->1
        return self.t

def sample_gaussian(m, v):
    sample = torch.randn(m.shape).to(torch.device("cuda"))
    # sample = torch.randn(m.shape)
    m = m.cuda()
    v = v.cuda()
    z = m + (v ** 0.5) * sample
    return z

# from torchviz import make_dot
# from torchsummary import summary

reconstruction_function = nn.MSELoss()
reconstruction_function.reduction = 'mean'
nllloss = nn.NLLLoss(reduction='mean')

def kl_normal(qm, qv, pm, pv, yh):
	element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm - yh).pow(2) / pv - 1)
	# element_wise = 0.5 * (torch.log(pv) - torch.log(qv) - qv / pv - (qm - pm - yh).pow(2) / pv - 1)
	kl = element_wise.sum(-1)
	#print("log var1", qv)
	return torch.mean(kl)


def adjust_learning_rate(base_lr, power, optimizer, epoch, warm_up=5, total= 24):
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


def cal_ii_loss(data, label, num_class):
    c_mean = class_mean(data, label, num_class)
    intra_diff = intra_spread(data, label, num_class, c_mean)
    inter_diff = inter_spread(c_mean)
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
parser.add_argument('--weight_decay', type=float, default=None)
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
parser.add_argument('--z_dim', type=int, default=None)
parser.add_argument('--latent_dim32', type=int, default=32)
parser.add_argument('--temperature', type=float, default=None)
parser.add_argument('--skip_connect', type=float, default=1, help='use skip connection')
parser.add_argument('--wre', type=float, default=1)
parser.add_argument('--wregroup', type=float, default=1)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--unseen_num', type=int, default=None)
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--wcontras', type=float, default=1)
parser.add_argument('--is_eval', action="store_true", default=False)
parser.add_argument('--auroc', action='store_true', default=False)
args = parser.parse_args()


def main():
    config.use_model = args.use_model
    config.cf = args.cf
    config.cf_threshold = args.cf_threshold
    config.yh = args.yh
    config.use_model_gau = args.use_model_gau
    config.threshold = args.threshold
    config.latent_dim32 = args.latent_dim32
    config.skip_connect = args.skip_connect
    config.wre = args.wre
    config.wcontras = args.wcontras
    config.wregroup = args.wregroup
    config.test = args.test
    config.is_eval = args.is_eval
    config.auroc = args.auroc
    config.re_threshold = 2
    if args.temperature != None:
        config.temperature = args.temperature
    if args.z_dim != None:
        config.z_dim = args.z_dim
    if args.load_path != None:
        config.load_path = args.load_path
    if args.load_epoch != None:
        config.load_epoch = args.load_epoch
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

    if args.num_classes != None:
        config.num_classes = args.num_classes
    if args.nepochs != None:
        config.nepochs = args.nepochs
    if args.lr != None:
        config.lr = args.lr
    if args.layers != None:
        config.layers = args.layers
    if args.unseen_num != None:
        config.unseen_num = args.unseen_num
    if args.load != None:
        config.load_path = args.load

    if args.dataset == "MNIST":
        load_dataset = MNIST_Dataset()
        args.num_classes = 6
        in_channel = 1
    elif args.dataset == "CIFAR10":
        load_dataset = CIFAR10_Dataset()
        args.num_classes = 6
        in_channel = 3

    elif args.dataset == "SVHN":
        load_dataset = SVHN_Dataset()
        args.num_classes = 6
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
    config.dataset = args.dataset
    config.save = config.dataset
    if args.dataset == "TinyImageNet":
        config.img_size = 64
    config.save = os.path.join(config.save, datetime.now().strftime("%d_%m_%Y%H_%M_%S"))
    # if not os.path.exists(config.save):
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
    # seed = config.seed
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)

    config.beta_scheduler = DeterministicWarmup(config.nepochs, config.beta)


    seed_sampler = int(args.seed_sampler)
    # train_dataset, val_dataset, pick_dataset, test_dataset = load_dataset.sampler_search(seed_sampler, args)
    train_dataset, val_dataset, test_dataset = load_dataset.sampler_train(seed_sampler, args)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    # pick_loader = DataLoader(pick_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    config.niters_per_epoch = len(train_dataset) // config.batch_size

    # Model #######################################

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
            stem_head_width=config.stem_head_width[idx], in_channel=config.in_channel, z_dim=config.z_dim, temperature=config.temperature, img_size=config.img_size,
            skip_connect=config.skip_connect, latent_dim32=config.latent_dim32)

        last = [2]
        lasts.append(last)
        model.build_structure(last)
        logging.info("net: " + str(model))
        for b in last:
            if len(config.width_mult_list) > 1:
                plot_op(getattr(model, "ops%d"%b), getattr(model, "path%d"%b), width=getattr(model, "widths%d"%b), head_width=config.stem_head_width[idx][1], F_base=config.Fch).savefig(os.path.join(config.save, "ops_%d_%d.png"%(arch_idx,b)), bbox_inches="tight")
            else:
                plot_op(getattr(model, "ops%d"%b), getattr(model, "path%d"%b), F_base=config.Fch).savefig(os.path.join(config.save, "ops_%d_%d.png"%(arch_idx,b)), bbox_inches="tight")
        plot_path_width(model.lasts, model.paths, model.widths).savefig(os.path.join(config.save, "path_width%d.png"%arch_idx))
        plot_path_width([2, 1, 0], [model.path2, model.path1, model.path0], [model.widths2, model.widths1, model.widths0]).savefig(os.path.join(config.save, "path_width_all%d.png"%arch_idx))
        logging.info("ops:" + str(model.ops))
        logging.info("path:" + str(model.paths))
        logging.info("last:" + str(model.lasts))

        device = torch.device("cuda:0")

        model = nn.DataParallel(model)
        model = model.to(device)

        # total_param = sum(p.numel() for p in model.parameters())
        # print("Total parameter number:")
        # print(total_param)
        #
        # temp_input = torch.randn(1, 3, 32, 32).to(device)
        # # temp_input = temp_input.cuda()
        # # print(model.module.is_cuda)
        # macs, params = profile(model, inputs=(temp_input,))
        # print("The FLOP is:")
        # print(macs)
        # print(params)
        #
        # return None


        init_weight(model.module, nn.init.kaiming_normal_, torch.nn.BatchNorm2d, config.bn_eps, config.bn_momentum, mode='fan_in', nonlinearity='relu')

        if arch_idx == 0 and len(config.arch_idx) > 1:
            partial = torch.load(os.path.join(config.teacher_path, "weights%d.pt"%arch_idx))
            state = model.state_dict()
            pretrained_dict = {k: v for k, v in partial.items() if k in state}
            state.update(pretrained_dict)
            model.load_state_dict(state)
        elif config.is_eval or config.auroc:
            model.load_state_dict(torch.load(os.path.join(config.eval_path, "weights0.pt")))

        # Optimizer ###################################
        base_lr = config.lr
        if arch_idx == 1 or len(config.arch_idx) == 1:
            sgd_optimizer = torch.optim.SGD(model.module.parameters(), lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
            adam_optimizer = torch.optim.Adam(model.module.parameters(), lr=base_lr, weight_decay=config.weight_decay)
        # print("--------------------------------------------------------------------")
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name)
        # print("--------------------------------------------------------------------")





    print(config.is_eval)
    if config.auroc:
        config.save = config.eval_path
        # test_train(model,train_loader,sgd_optimizer)
        print("start calculating AUROC")
        model.eval()
        perf = auroc_cal(config, model, train_loader, val_loader, test_loader)


    elif config.is_eval:
        config.save = config.eval_path
        # test_train(model,train_loader,sgd_optimizer)
        print("start evaluation")
        model.eval()
        best_f1_score = 0
        best_threshold = 0.5
        config.re_threshold = None
        for threshold in np.arange(0.5, 0.55, 0.05):
            config.threshold = threshold
            f1_score = test(model, train_loader, val_loader, test_loader, 1, logger, False)
            print("The f1 score of threshold {} is {}".format(threshold, f1_score))
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                best_threshold = threshold
            print("The best f1 score is {} at threshold {}".format(best_f1_score, best_threshold))
    elif config.is_train:
        best_f1 = 0
        tbar = tqdm(range(config.nepochs), ncols=80)
        optimizer = sgd_optimizer
        val_per_epoch = 5
        os.mkdir(os.path.join(config.save,"best_fea"))
        for epoch in tbar:
            config.beta = next(config.beta_scheduler)
            print("The value of beta in current epoch is {}".format(config.beta))

            logging.info(config.load_path)
            logging.info(config.save)
            logging.info("lr: " + str(optimizer.param_groups[0]['lr']))
            # training
            tbar.set_description("[Epoch %d/%d][train...]" % (epoch + 1, config.nepochs))

            train(train_loader, model, optimizer, logger, epoch)
            torch.cuda.empty_cache()

            adjust_learning_rate(base_lr, 0.992, optimizer, epoch+1, config.nepochs)

            # validation
            if epoch == 40:
                val_per_epoch = 1
            if not config.is_test and (epoch % val_per_epoch == 0 or epoch == 0):
                tbar.set_description("[Epoch %d/%d][validation...]" % (epoch + 1, config.nepochs))
                with torch.no_grad():
                    f1_score = test(model, train_loader, val_loader, test_loader, epoch, logger, True)
                    print("F1 score of current epoch is {}".format(f1_score))
                    logger.add_scalar('Acc_val/total_f1', f1_score, epoch)

                    if f1_score > best_f1 and config.test:
                        best_f1 = f1_score
                        best_val_epoch = epoch
                        shutil.move(os.path.join(config.save, "train_fea.txt"), os.path.join(config.save, "best_fea", "train_fea.txt"))
                        shutil.move(os.path.join(config.save, "train_tar.txt"),
                                    os.path.join(config.save, "best_fea", "train_tar.txt"))
                        shutil.move(os.path.join(config.save, "train_pre.txt"),
                                    os.path.join(config.save, "best_fea", "train_pre.txt"))
                        shutil.move(os.path.join(config.save, "train_rec.txt"),
                                    os.path.join(config.save, "best_fea", "train_rec.txt"))
                        # save model
                        states = {}
                        states['epoch'] = epoch
                        states['model'] = model.state_dict()
                        # states['val_loss'] = total_val_loss
                        states['f1_score'] = best_f1
                        torch.save(states, os.path.join(config.save, 'best_model.pkl'))
                        print('!!!Best Val Epoch: {}, Best Val F1:{:.4f}'.format(best_val_epoch, best_f1))
                        save(model, os.path.join(config.save, "best.pt"))

            save(model, os.path.join(config.save, "weights_{}.pt".format(epoch)))

    # for idx, arch_idx in enumerate(config.arch_idx):
    #     save(model, os.path.join(config.save, "weights%d.pt"%arch_idx))


def train(train_loader, model, optimizer, logger, epoch):
    model.train()

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format, ncols=80)
    dataloader = iter(train_loader)

    correct_train = 0
    epoch_ce_loss = 0
    epoch_re_loss = 0
    epoch_kl_loss = 0
    epoch_contras_loss = 0
    count = 0

    open('%s/train_fea.txt' % config.save, 'w').close()
    open('%s/train_tar.txt' % config.save, 'w').close()
    open('%s/train_pre.txt' % config.save, 'w').close()
    open('%s/train_rec.txt' % config.save, 'w').close()
    # if epoch >= 40:
    #     config.lamda = config.lamda * 0.9
    img_index = 1
    for step in pbar:

        optimizer.zero_grad()

        minibatch = dataloader.next()
        imgs = minibatch[0]
        target = minibatch[1]
        # print(target)
        imgs = imgs.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        imgs, target = Variable(imgs), Variable(target)

        description = ""
        # print("The class of the vector is:")
        # print(torch.Tensor.cpu(target[0]).detach().numpy())
        latent, latent_mu, latent_var, predict, predict_test, reconstructed, outputs = model(imgs)

        contras, yh = model.module.contrastive_loss(imgs, latent_mu, outputs, target, reconstructed, img_index)
        contras_loss = config.wcontras * contras
        epoch_contras_loss += contras_loss


        rec = reconstruction_function(reconstructed, imgs)
        re_loss = config.wre * rec
        epoch_re_loss += re_loss

        ce = nllloss(predict, target)
        ce_loss = config.lamda * ce
        epoch_ce_loss += ce_loss

        z_latent_mu, y_latent_mu = torch.split(latent_mu, [config.z_dim, config.latent_dim32], dim=1)
        z_latent_var, y_latent_var = torch.split(latent_var, [config.z_dim, config.latent_dim32], dim=1)
        pm_z, pv_z = torch.zeros(z_latent_mu.shape).cuda(), torch.ones(z_latent_var.shape).cuda()
        pm, pv = torch.zeros(y_latent_mu.shape).cuda(), torch.ones(y_latent_var.shape).cuda()
        kl_latent = kl_normal(y_latent_mu, y_latent_var, pm, pv, yh)
        kl_z = kl_normal(z_latent_mu, z_latent_var, pm_z, pv_z, 0)
        kl_loss = config.beta * (kl_latent + config.beta_z * kl_z)
        epoch_kl_loss += kl_loss

        loss = ce_loss + kl_loss + re_loss + contras_loss
        rec_loss = (reconstructed - imgs).pow(2).sum((3, 2, 1))


        loss.backward()
        optimizer.step()

        if img_index < 10:
            re = torch.Tensor.cpu(reconstructed[0]).detach().numpy()
            ori = torch.Tensor.cpu(imgs[0]).detach().numpy()
            temp = re
            temp = temp.transpose(1, 2, 0)
            temp = temp * (0.2023, 0.1994, 0.2010) + (0.4914, 0.4822, 0.4465)
            # temp = temp * 0.3081 + 0.1307
            temp = temp * 255
            temp = temp.astype(np.uint8)
            img = Image.fromarray(temp)
            img.save(os.path.join("cf_img", "{}_re.jpeg".format(img_index)))

            ori = ori
            ori = ori.transpose(1, 2, 0)
            ori = ori * (0.2023, 0.1994, 0.2010) + (0.4914, 0.4822, 0.4465)
            # ori = ori * 0.3081 + 0.1307
            ori = ori * 255
            ori = ori.astype(np.uint8)
            ori = Image.fromarray(ori)
            ori.save(os.path.join("cf_img", "{}_ori.jpeg".format(img_index)))
            img_index += 1

        pbar.set_description("[Step %d/%d]"%(step + 1, len(train_loader)) + description)
        logger.add_scalar('train/ce_loss', ce_loss, epoch * len(pbar) + step)
        logger.add_scalar('train/re_loss', re_loss, epoch * len(pbar) + step)
        logger.add_scalar('train/kl_loss', kl_loss / config.beta, epoch * len(pbar) + step)
        logger.add_scalar('train/contras_loss', contras_loss, epoch * len(pbar) + step)

        outlabel = predict.data.max(1)[1]  # get the index of the max log-probability
        correct_train += outlabel.eq(target.view_as(outlabel)).sum().item()

        cor_fea = y_latent_mu[(outlabel == target)]
        cor_tar = target[(outlabel == target)]
        # cor_fea = y_latent_mu
        # cor_tar = target
        pred_label = torch.Tensor.cpu(outlabel).detach().numpy()
        cor_fea = torch.Tensor.cpu(cor_fea).detach().numpy()
        cor_tar = torch.Tensor.cpu(cor_tar).detach().numpy()
        rec_loss = torch.Tensor.cpu(rec_loss).detach().numpy()
        with open('%s/train_pre.txt' % config.save, 'ab') as f:
            np.savetxt(f, pred_label, fmt='%f', delimiter=' ', newline='\r')
            f.write(b'\n')
        with open('%s/train_fea.txt' % config.save, 'ab') as f:
            np.savetxt(f, cor_fea, fmt='%f', delimiter=' ', newline='\r')
            f.write(b'\n')
        with open('%s/train_tar.txt' % config.save, 'ab') as t:
            np.savetxt(t, cor_tar, fmt='%d', delimiter=' ', newline='\r')
            t.write(b'\n')
        with open('%s/train_rec.txt' % config.save, 'ab') as m:
            np.savetxt(m, rec_loss, fmt='%f', delimiter=' ', newline='\r')
            m.write(b'\n')
        count += imgs.shape[0]
    train_acc = float(100 * correct_train) / count
    logger.add_scalar('loss_epoch/train_ce', epoch_ce_loss / count, epoch)
    logger.add_scalar('loss_epoch/train_re', epoch_re_loss / count, epoch)
    logger.add_scalar('loss_epoch/train_kl', epoch_kl_loss / config.beta / count, epoch)
    logger.add_scalar('loss_epoch/train_contras', epoch_contras_loss / count, epoch)
    logger.add_scalar('train_acc/train_acc', train_acc, epoch)
    print('Train_Acc: {}/{} ({:.2f}%)'.format(correct_train, count, train_acc))
    torch.cuda.empty_cache()

    return train_acc

def test_train(model, train_loader, optimizer):
    open('%s/train_fea.txt' % config.save, 'w').close()
    open('%s/train_tar.txt' % config.save, 'w').close()
    open('%s/train_pre.txt' % config.save, 'w').close()
    open('%s/train_rec.txt' % config.save, 'w').close()

    model.eval()
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format, ncols=80)
    dataloader = iter(train_loader)
    for step in pbar:

        optimizer.zero_grad()

        minibatch = dataloader.next()
        imgs = minibatch[0]
        target = minibatch[1]
        # print(target)
        imgs = imgs.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        imgs, target = Variable(imgs), Variable(target)

        latent, latent_mu, latent_var, predict, predict_test, reconstructed, outputs = model(imgs)
        z_latent_mu, y_latent_mu = torch.split(latent_mu, [config.z_dim, config.latent_dim32], dim=1)

        rec_loss = (reconstructed - imgs).pow(2).sum((3, 2, 1))

        outlabel = predict.data.max(1)[1]  # get the index of the max log-probability

        cor_fea = y_latent_mu[(outlabel == target)]
        cor_tar = target[(outlabel == target)]
        pred_label = torch.Tensor.cpu(outlabel).detach().numpy()
        cor_fea = torch.Tensor.cpu(cor_fea).detach().numpy()
        cor_tar = torch.Tensor.cpu(cor_tar).detach().numpy()
        rec_loss = torch.Tensor.cpu(rec_loss).detach().numpy()
        with open('%s/train_pre.txt' % config.save, 'ab') as f:
            np.savetxt(f, pred_label, fmt='%f', delimiter=' ', newline='\r')
            f.write(b'\n')
        with open('%s/train_fea.txt' % config.save, 'ab') as f:
            np.savetxt(f, cor_fea, fmt='%f', delimiter=' ', newline='\r')
            f.write(b'\n')
        with open('%s/train_tar.txt' % config.save, 'ab') as t:
            np.savetxt(t, cor_tar, fmt='%d', delimiter=' ', newline='\r')
            t.write(b'\n')
        with open('%s/train_rec.txt' % config.save, 'ab') as m:
            np.savetxt(m, rec_loss, fmt='%f', delimiter=' ', newline='\r')
            m.write(b'\n')

def test(model, train_loader, val_loader, test_loader, epoch=0, logger=None, val=False,  device=torch.device('cuda')):
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
    pred_list = []
    target_list = []

    for data_val, target_val in val_loader:
        data_val, target_val = data_val.cuda(), target_val.cuda()
        with torch.no_grad():
            data_val, target_val = Variable(data_val), Variable(target_val)

        latent, mu_val, latent_var, predict, output_val, de_val, outputs = model(data_val)

        if val:
            total_num += len(target_val)
            rec = reconstruction_function(de_val, data_val)
            re_loss = config.wre * rec
            total_re_loss += re_loss

            ce = nllloss(predict, target_val)
            ce_loss = config.lamda * ce
            total_ce_loss += ce_loss

            contras, yh = model.module.contrastive_loss(data_val, mu_val, outputs, target_val, de_val)
            contras_loss = config.wcontras * contras
            total_contras_loss += contras_loss

            z_latent_mu, y_latent_mu = torch.split(mu_val, [config.z_dim, config.latent_dim32], dim=1)
            z_latent_var, y_latent_var = torch.split(latent_var, [config.z_dim, config.latent_dim32], dim=1)
            pm_z, pv_z = torch.zeros(z_latent_mu.shape).cuda(), torch.ones(z_latent_var.shape).cuda()
            pm, pv = torch.zeros(y_latent_mu.shape).cuda(), torch.ones(y_latent_var.shape).cuda()
            kl_latent = kl_normal(y_latent_mu, y_latent_var, pm, pv, yh)
            kl_z = kl_normal(z_latent_mu, z_latent_var, pm_z, pv_z, 0)
            kl_loss = config.beta * (kl_latent + config.beta_z * kl_z)
            total_kl_loss += kl_loss

            vallabel = predict.data.max(1)[1]  # get the index of the max log-probability
            correct_val += vallabel.eq(target_val.view_as(vallabel)).sum().item()

            target_list = target_list + target_val.squeeze().tolist()
            pred_list = pred_list + vallabel.squeeze().tolist()

        pre_val = output_val.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        rec_val = (de_val - data_val).pow(2).sum((3, 2, 1))
        _, mu_val = torch.split(mu_val, [config.z_dim, config.latent_dim32], dim=1)
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

    if val:

        closed_val_f1 = f1_score(target_list, pred_list, average='macro')
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
            logger.add_scalar("Acc_val/acc", val_acc, epoch)
            logger.add_scalar("Acc_val/closed_f1_score", closed_val_f1, epoch)

    for data_omn, target_omn in test_loader:
        tar_omn = torch.from_numpy(config.num_classes * np.ones(target_omn.shape[0]))
        data_omn = data_omn.cuda()
        with torch.no_grad():
            data_omn = Variable(data_omn)
        _, mu_omn, _, _, output_omn, de_omn, _ = model(data_omn)
        output_omn = torch.exp(output_omn)
        # prob_omn = output_omn.max(1)[0]  # get the value of the max probability
        pre_omn = output_omn.max(1, keepdim=True)[1]  # get the index of the max log-probability
        rec_omn = (de_omn - data_omn).pow(2).sum((3, 2, 1))
        _, mu_omn = torch.split(mu_omn, [config.z_dim, config.latent_dim32], dim=1)
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
    if val and perf[-1] != 0:
        f1_matrix = np.loadtxt(config.save + '/performance.txt')
        precision = f1_matrix[-1][3]
        recall = f1_matrix[-1][4]
        f1 = (precision * recall) / (precision + recall)
        logger.add_scalar('Acc_val/open_f1_score', f1, epoch)
    return perf[-1]

if __name__ == '__main__':
    main() 
