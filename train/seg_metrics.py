# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: Hang Zhang
# ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
##
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import torch


class Cls_Metrics(object):
    def __init__(self, n_classes=10):
        # self.n_classes = n_classes
        self.correct8 = 0
        self.correct16 = 0
        self.correct32 = 0
        self.correct_final = 0
        self.total = 0


    def update(self, pred8, pred16, pred32, pred_final, target):
        c8, c16, c32, c_final, total = batch_cls(pred8,pred16,pred32,pred_final,target)
        self.correct8 += c8
        self.correct16 += c16
        self.correct32 += c32
        self.correct_final += c_final
        self.total += total

    def get_scores(self):

        return self.correct8/self.total, self.correct16/self.total, self.correct32/self.total, self.correct_final/self.total

    def reset(self):
        self.correct8 = 0
        self.correct16 = 0
        self.correct32 = 0
        self.correct_final = 0
        self.total = 0


def batch_cls(pred8, pred16, pred32, pred_final, target):
    c8 = 0
    c16 = 0
    c32 = 0
    c_final = 0
    pred8 = pred8.data.max(1)[1]
    c8 += pred8.eq(target.view_as(pred8)).sum().item()
    pred16 = pred16.data.max(1)[1]
    c16 += pred16.eq(target.view_as(pred16)).sum().item()
    pred32 = pred32.data.max(1)[1]
    c32 += pred32.eq(target.view_as(pred32)).sum().item()
    pred_final = pred_final.data.max(1)[1]
    c_final += pred_final.eq(target.view_as(pred_final)).sum().item()
    return c8, c16, c32, c_final, len(target)

# def batch_pix_accuracy(predict, target):
#     """Batch Pixel Accuracy
#     Args:
#         predict: input 4D tensor
#         target: label 3D tensor
#     """
#     _, predict = torch.max(predict, 1)
#     predict = predict.cpu().numpy() + 1
#     target = target.cpu().numpy() + 1
#     pixel_labeled = np.sum(target > 0)
#     pixel_correct = np.sum((predict == target)*(target > 0))
#     assert pixel_correct <= pixel_labeled, \
#         "Correct area should be smaller than Labeled"
#     return pixel_correct, pixel_labeled
#
#
# def batch_intersection_union(predict, target, nclass):
#     """Batch Intersection of Union
#     Args:
#         predict: input 4D tensor
#         target: label 3D tensor
#         nclass: number of categories (int)
#     """
#     _, predict = torch.max(predict, 1)
#     mini = 1
#     maxi = nclass
#     nbins = nclass
#     predict = predict.cpu().numpy() + 1
#     target = target.cpu().numpy() + 1
#
#     k = (target >= 1) & (target <= nclass)
#     # predict = predict * (target > 0).astype(predict.dtype)
#     predict = predict * k.astype(predict.dtype)
#     intersection = predict * (predict == target)
#     # areas of intersection and union
#     area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
#     area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
#     area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
#     area_union = area_pred + area_lab - area_inter
#     assert (area_inter <= area_union).all(), \
#         "Intersection area should be smaller than Union area"
#     return area_inter, area_union
#
#
# # ref https://github.com/CSAILVision/sceneparsing/blob/master/evaluationCode/utils_eval.py
# def pixel_accuracy(im_pred, im_lab):
#     im_pred = np.asarray(im_pred)
#     im_lab = np.asarray(im_lab)
#
#     # Remove classes from unlabeled pixels in gt image.
#     # We should not penalize detections in unlabeled portions of the image.
#     pixel_labeled = np.sum(im_lab > 0)
#     pixel_correct = np.sum((im_pred == im_lab) * (im_lab > 0))
#     #pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
#     return pixel_correct, pixel_labeled
#
#
# def intersection_and_union(im_pred, im_lab, num_class):
#     im_pred = np.asarray(im_pred)
#     im_lab = np.asarray(im_lab)
#     # Remove classes from unlabeled pixels in gt image.
#     im_pred = im_pred * (im_lab > 0)
#     # Compute area intersection:
#     intersection = im_pred * (im_pred == im_lab)
#     area_inter, _ = np.histogram(intersection, bins=num_class-1,
#                                  range=(1, num_class - 1))
#     # Compute area union:
#     area_pred, _ = np.histogram(im_pred, bins=num_class-1,
#                                 range=(1, num_class - 1))
#     area_lab, _ = np.histogram(im_lab, bins=num_class-1,
#                                range=(1, num_class - 1))
#     area_union = area_pred + area_lab - area_inter
#     return area_inter, area_union
