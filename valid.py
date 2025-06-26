# -*- codeing = utf-8 -*-
# @Time : 2023/10/26 22:58
# @Author : 李昌杏
# @File : test.py
# @Software : PyCharm
import scipy
import torch
import torch.utils.data
from torch import nn
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

def map_sake( gt_labels_gallery, predicted_features_query, gt_labels_query, scores, k=None):
    mean_mAP = []
    for fi in range(predicted_features_query.shape[0]):
        mapi = eval_AP_inner(gt_labels_query[fi], scores[fi], gt_labels_gallery, top=k)
        mean_mAP.append(mapi)
    return mean_mAP

def prec_sake( gt_labels_gallery, predicted_features_query, gt_labels_query, scores, k=None):
    # compute precision for two modalities
    mean_prec = []
    for fi in range(predicted_features_query.shape[0]):
        prec = eval_precision(gt_labels_query[fi], scores[fi], gt_labels_gallery, top=k)
        mean_prec.append(prec)
    # print("precision for all samples: ", np.nanmean(mean_prec))
    return np.nanmean(mean_prec)

def eval_AP_inner(inst_id, scores, gt_labels, top=None):
    pos_flag = gt_labels == inst_id
    tot = scores.shape[0]  # total retrieved samples
    tot_pos = np.sum(pos_flag)  # total true position

    sort_idx = np.argsort(-scores)
    tp = pos_flag[sort_idx]  # sorted true positive
    fp = np.logical_not(tp)  # sorted false positive

    if top is not None:
        top = min(top, tot)
        tp = tp[:top]  # select top-k true position
        fp = fp[:top]
        tot_pos = min(top, tot_pos)

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    try:
        rec = tp / tot_pos
        prec = tp / (tp + fp)
    except:
        print(inst_id, tot_pos)
        return np.nan

    ap = VOCap(rec, prec)
    return ap

def VOCap(rec, prec):
    mrec = np.append(0, rec)  # put 0 in the first element
    mrec = np.append(mrec, 1)  # put 1 in the last element

    mpre = np.append(0, prec)
    mpre = np.append(mpre, 0)

    for ii in range(len(mpre) - 2, -1, -1):  # sort mpre, the smaller, the latter
        mpre[ii] = max(mpre[ii], mpre[ii + 1])

    msk = [i != j for i, j in zip(mrec[1:], mrec[0:-1])]
    ap = np.sum((mrec[1:][msk] - mrec[0:-1][msk]) * mpre[1:][msk])
    return ap

def eval_precision(inst_id, scores, gt_labels, top=100):
    pos_flag = gt_labels == inst_id
    tot = scores.shape[0]

    top = min(top, tot)

    sort_idx = np.argsort(-scores)
    return np.sum(pos_flag[sort_idx][:top]) / top

def map_sake_test(test_sketch_features,test_image_features, labels_images,labels_sketches,k=None):
    scores = torch.mm(test_sketch_features,test_image_features.T)
    if k == None:
        k = scores.shape[1]
    mean_mAP = []
    mean_prec = []
    mAP_ls = [[] for _ in range(len(np.unique(labels_sketches)))]
    for idx,fi in enumerate(tqdm(range(scores.shape[0]))):
        mapi = eval_AP_inner(labels_sketches[fi], scores[fi],  labels_images, top=k)
        mean_mAP.append(mapi)
        mAP_ls[labels_sketches[fi]].append(mapi)
        prec = eval_precision(labels_sketches[fi], scores[fi],  labels_images, top=100)
        mean_prec.append(prec)
    # print("precision for all samples: ", np.nanmean(mean_prec))
    # for idx,item in enumerate(mAP_ls):
    #     print(idx,sum(item)/len(item))
    return np.mean(mean_mAP), np.nanmean(mean_prec)

def test(img_loader, skt_loader, img_model, skt_model, k=None):
    img_model.eval()
    skt_model.eval()
    test_images = []
    test_sketches = []
    labels_images = []
    labels_sketches = []
    with torch.no_grad():
        for idx, (photo, label) in enumerate(tqdm(img_loader)):
            photo, label = photo.cuda(), label
            photo_reprs = img_model(photo).detach().cpu()
            for i in label:
                labels_images.append(i)

            test_images.append(photo_reprs)

        test_image_features = torch.cat(test_images).detach()
        labels_images = torch.tensor(labels_images)

        for idx, (sketch, label) in enumerate(tqdm(skt_loader)):
            sketch, label = sketch.cuda(), label
            sketch_reprs = skt_model(sketch).detach().cpu()
            for i in label:
                labels_sketches.append(i)
            test_sketches.append(sketch_reprs)

        test_sketch_features = torch.cat(test_sketches).detach()
        labels_sketches = torch.tensor(labels_sketches)

        labels_images = labels_images.numpy()
        labels_sketches = labels_sketches.numpy()

        print(test_image_features.shape, test_sketch_features.shape)
        test_image_features = F.normalize(test_image_features, dim=1, p=2)
        test_sketch_features = F.normalize(test_sketch_features, dim=1, p=2)
        i_map, prec = map_sake_test(test_sketch_features, test_image_features, labels_images, labels_sketches,k=k)
        print('MAP:', i_map, "Prec", prec)
    return i_map, prec

