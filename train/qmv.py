import torch
from torch.autograd import Variable
import os
import scipy.stats as st
os.environ['R_HOME'] = '/home/neuron/anaconda3/envs/nas_osr/lib/R'
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
import pprint, pickle
import argparse
import scipy.stats as stats
numpy2ri.activate()
mvt = importr('mvtnorm')



def revise(args, rec_loss=None):

    if not type(rec_loss).__module__ == np.__name__:
        train_rec = np.loadtxt('%s/train_rec.txt' %args.save)
    else:
        train_rec = rec_loss

    rec_mean = np.mean(train_rec)
    rec_std = np.std(train_rec)
    rec_thres = rec_mean + 2 * rec_std #95%

    test_rec = np.loadtxt('%s/test_rec.txt' %args.save)
    test_pre = np.loadtxt('%s/test_pre.txt' %args.save)
    test_pre[(test_rec > rec_thres)] = args.num_classes
    open('%s/test_pre_after.txt' %args.save , 'w').close()  # clear
    np.savetxt('%s/test_pre_after.txt' %args.save , test_pre, delimiter=' ', fmt='%d')


def revise_cf(args, lvae, feature_y_mean, val_loader, test_loader, rec_loss=None):
    print('Start counterfactual revise ...')
    if not type(rec_loss).__module__ == np.__name__:
        train_rec = np.loadtxt('%s/train_rec.txt' %args.save)
    else:
        train_rec = rec_loss

    rec_mean = np.mean(train_rec)
    rec_std = np.std(train_rec)
    if args.re_threshold == None:
        args.re_threshold = 2
    rec_thres = rec_mean + args.re_threshold * rec_std
    # rec_thres = rec_mean + 2 * rec_std #95%

    test_rec_cf = lvae.module.rec_loss_cf(feature_y_mean, val_loader, test_loader, args)
    test_rec_cf = test_rec_cf.cpu().numpy()
    test_pre = np.loadtxt('%s/test_pre.txt' %args.save)
    test_pre[(test_rec_cf > rec_thres)] = args.num_classes
    open('%s/test_pre_after.txt' %args.save , 'w').close()  # clear
    np.savetxt('%s/test_pre_after.txt' %args.save , test_pre, delimiter=' ', fmt='%d')
    return test_rec_cf > rec_thres

class GAU(object):
    def __init__(self, args, fea=None, tar=None):
        if not type(fea).__module__ == np.__name__:
            self.trainfea = np.loadtxt('%s/train_fea.txt' %args.save )
            self.traintar = np.loadtxt('%s/train_tar.txt' %args.save )
        else:
            self.trainfea = fea
            self.traintar = tar

        self.labelset = set(self.traintar)
        self.labelnum = len(self.labelset)
        self.num, self.dim = np.shape(self.trainfea)
        self.gau = self.train()

    def train(self):
        trainfea = self.trainfea
        traintar = self.traintar
        labelnum = self.labelnum
        trainsize = self.trainfea.shape[0]
        for i in range(labelnum):
            locals()['matrix' + str(i)] = np.empty(shape=[0,self.dim])

        gau = []
        muandsigma = []
        for j in range(trainsize):
            for i in range(labelnum):
                if traintar[j] == i:
                    locals()['matrix' + str(i)] = np.append((locals()['matrix' + str(i)]), np.array([np.array(trainfea[j])]),
                                                            axis=0)

        for i in range(labelnum):
            locals()['mu' + str(i)] = np.mean(np.array(locals()['matrix' + str(i)]),axis=0)
            locals()['sigma' + str(i)] = np.cov(np.array((locals()['matrix' + str(i)] - locals()['mu' + str(i)])).T)
            locals()['gau' + str(i)] = [locals()['mu' + str(i)],locals()['sigma' + str(i)]]
            print(i)
            print(locals()['mu' + str(i)])
            print(np.diag(locals()['sigma' + str(i)])**0.5)
            gau.append(locals()['gau' + str(i)])

        return gau

    def test(self, testsetlist, args):


        threshold = args.threshold

        testfea = np.loadtxt(testsetlist[0])
        testtar = np.loadtxt(testsetlist[1])
        testpre = np.loadtxt(testsetlist[2])

        labelnum = self.labelnum
        gau = self.gau
        dim = self.dim
        performance = np.zeros([labelnum + 1, 5])
        testsize = testfea.shape[0]
        result = []
        if threshold != 0:
            print('threshold is', args.threshold)
            print('Reconstruction threshold is', args.re_threshold)


        def multivariateGaussian(vector, mu, sigma):
            vector = np.array(vector)
            d = (np.mat(vector - mu)) * np.mat(np.linalg.pinv(sigma)) * (np.mat(vector - mu).T)
            p = np.exp(d * (-0.5)) / (((2 * np.pi) ** int(dim/2)) * (np.linalg.det(sigma)) ** (0.5))
            p = float(p)
            return p

        def multivariateGaussianNsigma(sigma,threshold):
            q = np.array(mvt.qmvnorm(threshold, sigma = sigma, tail = "both")[0])
            n = q[0]
            m = (np.diag(sigma) ** 0.5) * n
            d = (np.mat(m) * np.mat(np.linalg.pinv(sigma)) * (np.mat(m).T))
            p = np.exp(d * (-0.5)) / (((2 * np.pi) ** int(dim/2)) * (np.linalg.det(sigma)) ** (0.5))
            return p

        pNsigma = np.zeros(labelnum)
        p = np.zeros(labelnum)
        mu = []
        sigma = []

        for j in range(labelnum):
            mu.append(gau[j][0])
            sigma.append(gau[j][1])
            pNsigma[j] = multivariateGaussianNsigma(sigma[j],threshold)


        for i in range(testsize):
            for j in range(labelnum):
                p[j] = multivariateGaussian(testfea[i],mu[j],sigma[j])

            delta = p-pNsigma
            # print(delta)
            if len(delta[delta > 0]) == 0:
                #Unseen
                prediction = labelnum
            else:
                #Seen
                prediction = testpre[i]

            result.append(prediction)

        #result
        result = np.array(result)
        np.savetxt('%s/Result.txt' %args.save, result)

        if not args.auroc:
            for i in range(labelnum+1):
                locals()['resultIndex' + str(i)] = np.argwhere(result == i)
                locals()['targetIndex' + str(i)] = np.argwhere(testtar == i)

            for i in range(labelnum+1):
                locals()['tp' + str(i)] = np.sum((testtar[(locals()['resultIndex' + str(i)])]) == i)
                locals()['fp' + str(i)] = np.sum((testtar[(locals()['resultIndex' + str(i)])]) != i)
                locals()['fn' + str(i)] = np.sum((result[locals()['targetIndex' + str(i)]]) != i)
                # print(locals()['tp' + str(i)],locals()['fp' + str(i)],locals()['fn' + str(i)])

                performance[i, 0] = locals()['tp' + str(i)]
                performance[i, 1] = locals()['fp' + str(i)]
                performance[i, 2] = locals()['fn' + str(i)]

            for i in range(labelnum+1):
                locals()['precision' + str(i)] = locals()['tp' + str(i)]/(locals()['tp' + str(i)]+locals()['fp' + str(i)] + 1)
                locals()['recall' + str(i)] = locals()['tp' + str(i)]/(locals()['tp' + str(i)]+locals()['fn' + str(i)] + 1)
                locals()['f-measure' + str(i)] = 2* locals()['precision' + str(i)]*locals()['recall' + str(i)]/(locals()['precision' + str(i)] + locals()['recall' + str(i)])

                performance[i, 3] = locals()['precision' + str(i)]
                performance[i, 4] = locals()['recall' + str(i)]

            performancesum = performance.sum(axis = 0)
            mafmeasure = 2*performancesum[3]*performancesum[4]/(performancesum[3]+performancesum[4])
            maperformance = np.append((performancesum)[3:],mafmeasure)/(labelnum+1)

            print(performance)
            np.savetxt('%s/performance.txt' %args.save , performance)
            return maperformance

        else:
            # tp = np.sum(np.logical_and(result==(labelnum), testtar==(labelnum)))
            # fp = np.sum(np.logical_and(result==(labelnum), testtar!=(labelnum)))
            # fn = np.sum(np.logical_and(result!=(labelnum), testtar==(labelnum)))
            # tn = np.sum(np.logical_and(result!=(labelnum), testtar!=(labelnum)))
            #
            # return [tp/(tp+fn), fp/(tn+fp)]
            return result

def get_mean_y(train_feature, train_target, args):
    # label_num = int(torch.max(train_target)) + 1
    label_num = args.num_classes
    # print(label_num)
    feature_mean_y = []
    for label_i in range(label_num):
        feature_i = train_feature[(train_target == label_i)]
        feature_i = feature_i.mean(0)
        feature_mean_y.append(feature_i)

    return feature_mean_y

def ocr_test(args, lvae, train_loader, val_loader, test_loader):
    if not args.use_model:
        revise(args)
        gau = GAU(args)

    else:
        lvae.eval()
        train_fea_all = []
        train_tar_all = []
        train_rec_loss_all = []


        # get train feature
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(train_loader):
                if args.cuda:
                    data = data.cuda()
                    target = target.cuda()
                data, target = Variable(data), Variable(target)

                _, mu, _, _, output, x_re, _ = lvae(data)

                z_mu, y_mu = torch.split(mu, [args.z_dim, args.latent_dim32], dim=1)
                train_rec_loss = (x_re - data).pow(2).sum((3, 2, 1))
                outlabel = output.data.max(1)[1]  # get the index of the max log-probability
                train_fea = y_mu[(outlabel == target)]
                train_tar = target[(outlabel == target)]

                train_fea_all.append(train_fea)
                train_tar_all.append(train_tar)
                train_rec_loss_all.append(train_rec_loss)

        train_fea = torch.cat(train_fea_all, 0)
        train_tar = torch.cat(train_tar_all, 0)
        train_rec_loss = torch.cat(train_rec_loss_all, 0)

        ## cf
        if args.cf:
            with torch.no_grad():
                if args.yh:
                    target_en = torch.eye(args.num_classes)
                    feature_y_mean = lvae.module.get_yh(target_en.cuda())
                else:
                    feature_y_mean = get_mean_y(train_fea, train_tar, args)
                    feature_y_mean = torch.cat(feature_y_mean, dim=0).view(args.num_classes, args.latent_dim32)

                if args.cf_threshold:
                    rec_loss_cf_train = lvae.module.rec_loss_cf_train(feature_y_mean, train_loader, args)
                    train_rec_loss = rec_loss_cf_train.cpu().numpy()
                else:
                    train_rec_loss = train_rec_loss.cpu().numpy()

                revise_cf(args, lvae, feature_y_mean, val_loader, test_loader, rec_loss=train_rec_loss)

        else:
            train_rec_loss = train_rec_loss.cpu().numpy()
            revise(args, rec_loss=train_rec_loss)

        if args.use_model_gau:
            train_fea = train_fea.cpu().numpy()
            train_tar = train_tar.cpu().numpy()
            gau = GAU(args, train_fea, train_tar)
        else:
            gau = GAU(args)

    # if gau.labelnum != args.num_classes:
    #     print("Some classes do not have any correct features. Skipping this one")
    #     return [0, 0, 0]
    traintar = gau.traintar.tolist()
    label_count = [traintar.count(x) for x in set(traintar)]
    if 1 in label_count or  gau.labelnum != args.num_classes:
        print("Some classes do not have enough correct features. Skipping this one")
        return [0, 0, 0]

    test_sample = ['%s/test_fea.txt' % args.save, '%s/test_tar.txt' % args.save,
           '%s/test_pre_after.txt' % args.save]


    perf_test = gau.test(test_sample, args)

    print("Precision: %.4f  Recall: %.4f  F1 Score: %.4f" %(perf_test[0], perf_test[1], perf_test[2]))
    if args.cf:
        np.savetxt('%s/ma_cf.txt' % args.save, perf_test)
    else:
        np.savetxt('%s/ma.txt' % args.save, perf_test)

    ### write F1 score in one txt in father dir
    save_path_father = '%s/ma_all.txt' %args.save
    if not os.path.exists(save_path_father):
        # assert args.run_idx == 0
        with open(save_path_father, "w") as f:
            f.write(str(perf_test[2]))
    else:
        with open(save_path_father, "a") as f:
            f.write('\n')
            f.write(str(perf_test[2]))

    return perf_test

def auroc_cal(args, lvae, train_loader, val_loader, test_loader):
    lvae.eval()
    train_fea_all = []
    train_tar_all = []
    train_rec_loss_all = []

    # get train feature
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data = data.cuda()
                target = target.cuda()
            data, target = Variable(data), Variable(target)

            _, mu, _, _, output, x_re, _ = lvae(data)

            z_mu, y_mu = torch.split(mu, [args.z_dim, args.latent_dim32], dim=1)
            train_rec_loss = (x_re - data).pow(2).sum((3, 2, 1))
            outlabel = output.data.max(1)[1]  # get the index of the max log-probability
            train_fea = y_mu[(outlabel == target)]
            train_tar = target[(outlabel == target)]

            train_fea_all.append(train_fea)
            train_tar_all.append(train_tar)
            train_rec_loss_all.append(train_rec_loss)

    train_fea = torch.cat(train_fea_all, 0)
    train_tar = torch.cat(train_tar_all, 0)
    train_rec_loss = torch.cat(train_rec_loss_all, 0)
    # train_fea = np.loadtxt('%s/train_fea.txt' % args.save)
    # train_pred = np.loadtxt('%s/train_pred.txt' % args.save)
    # train_tar = np.loadtxt('%s/train_tar.txt' % args.save)
    # train_rec_loss = np.loadtxt('%s/train_rec.txt' % args.save)

    tp_fp_list = []
    # Rreconstruction threshold
    with torch.no_grad():
        if args.yh:
            target_en = torch.eye(args.num_classes)
            feature_y_mean = lvae.module.get_yh(target_en.cuda())
        else:
            feature_y_mean = get_mean_y(train_fea, train_tar, args)
            feature_y_mean = torch.cat(feature_y_mean, dim=0).view(args.num_classes, args.latent_dim32)

        if args.cf_threshold:
            rec_loss_cf_train = lvae.module.rec_loss_cf_train(feature_y_mean, train_loader, args)
            train_rec_loss = rec_loss_cf_train.cpu().numpy()
        else:
            train_rec_loss = train_rec_loss.cpu().numpy()

    re_results = {}
    for re_threshold in np.arange(2, 0, -0.5):
        print("Calculating re threshold ", str(re_threshold))
        args.re_threshold = re_threshold
        results = revise_cf(args, lvae, feature_y_mean, val_loader, test_loader, rec_loss=train_rec_loss)
        re_results[re_threshold] = results

    if args.use_model_gau:
        train_fea = train_fea.cpu().numpy()
        train_tar = train_tar.cpu().numpy()
        gau = GAU(args, train_fea, train_tar)
    else:
        gau = GAU(args)


    #Distribution probability threshold
    dist_results = {}
    for threshold in np.arange(0.5, 0.95, 0.05):
        args.threshold = threshold

        test_sample = ['%s/test_fea.txt' % args.save, '%s/test_tar.txt' % args.save,
               '%s/test_pre_after.txt' % args.save]
        perf_test = gau.test(test_sample, args)
        dist_results[threshold] = perf_test

    # calculate tpr, fpr
    testtar = np.loadtxt('%s/test_tar.txt' % args.save)
    labelnum = 6
    for key_re, value_re in re_results.items():
        for key_dist, value_dist in dist_results.items():
            result = (value_re | value_dist)

            tp = np.sum(np.logical_and(result == (labelnum), testtar == (labelnum)))
            fp = np.sum(np.logical_and(result==(labelnum), testtar!=(labelnum)))
            fn = np.sum(np.logical_and(result!=(labelnum), testtar==(labelnum)))
            tn = np.sum(np.logical_and(result!=(labelnum), testtar!=(labelnum)))

            tp_fp_list.append([tp/(tp+fn), fp/(tn+fp)])

    np.savetxt('%s/tpr_fpr.txt' % args.save, np.array(tp_fp_list))

    return tp_fp_list
