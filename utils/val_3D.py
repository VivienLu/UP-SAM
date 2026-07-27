import math
from glob import glob

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from tqdm import tqdm
import logging
import os
from SAM_Med3D.lora_sam_med3d import finetune_model_predict3D_lab, finetune_model_predict3D_unlab, finetune_sam_test_predict3D
from monai.losses import DiceCELoss
from dataloaders.LeftAtrium import LAHeart
import time


sam_dice_ce_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1, AMC=False):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]

                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1 = net(test_patch)
                    # ensemble
                    if AMC:
                        y2 = 0.25 * (y1[0]+ y1[1]+ y1[2]+ y1[3])
                        y = torch.softmax(y2, dim=1)
                    else:
                        y = torch.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    return label_map


def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred,gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred,gt)
        return np.array([dice, jc, hd95, asd])
    else:
        return np.zeros(4)


def test_all_case(net, data_dir, test_list="full_test.list", num_classes=3, 
    patch_size=(48, 160, 160), stride_xy=32, stride_z=24, AMC=False, test_save_path=None):

    with open(data_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    if 'LA' in data_dir:
        image_list = [data_dir + "/{}/mri_norm2.h5".format(
            item.replace('\n', '').split(",")[0]) for item in image_list]
    else:
        image_list = [data_dir + "/{}.h5".format(
            item.replace('\n', '').split(",")[0]) for item in image_list]
    total_metric = np.zeros((num_classes-1, 4))

    print("Validation begin")
    for image_path in tqdm(image_list):
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        prediction = test_single_case(
            net, image, stride_xy, stride_z, patch_size, num_classes=num_classes, AMC=AMC)
        for i in range(1, num_classes):
            total_metric[i-1, :] += cal_metric(label == i, prediction == i)

    print("Validation end")
    return total_metric / len(image_list)


def test_all_case_print_single(net, data_dir, test_list="full_test.list", num_classes=3, 
    patch_size=(48, 160, 160), stride_xy=32, stride_z=24, AMC=False, test_save_path=None, save_result=False):

    with open(data_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    if 'LA' in data_dir:
        image_list = [data_dir + "/{}/mri_norm2.h5".format(
            item.replace('\n', '').split(",")[0]) for item in image_list]
    else:
        image_list = [data_dir + "/{}.h5".format(
            item.replace('\n', '').split(",")[0]) for item in image_list]
    total_metric = np.zeros((num_classes-1, 4))

    idx = 0
    total_metric_all = np.zeros((num_classes-1, len(image_list), 4))
    total_num = len(image_list)

    print("Validation begin")
    time_start = time.time()

    for image_path in tqdm(image_list):
        h5f = h5py.File(image_path, 'r')
        if 'LA' in image_path:
            ids = image_path.split("/")[-2]
        else:
            ids = image_path.split("/")[-1].replace(".h5", "")
        image = h5f['image'][:]
        label = h5f['label'][:]
        prediction = test_single_case(
            net, image, stride_xy, stride_z, patch_size, num_classes=num_classes, AMC=AMC)

        logging.info('\n[{}/{}] {}%:\t{}'.format(idx+1,total_num,round((idx+1)/total_num*100),ids))

        for i in range(1, num_classes):
            score = cal_metric(label == i, prediction == i)
            total_metric[i-1, :] += score
            total_metric_all[i-1, idx, :] = score
            logging.info('Class: {}: dice {:.2f}% | jaccard {:.2f}% | hd95 {:.2f} | asd {:.2f}'
                            .format(i, score[0]*100,score[1]*100,score[2],score[3]))
        if save_result:
            np.save(os.path.join(test_save_path, ids+'_UPSAM_results.npy'), {'img':image, 'lab': label, 'pred':prediction})
            logging.info('Save at: {}'.format(os.path.join(test_save_path, ids +'_UPSAM_results.npy')))
        idx += 1
    time_end = time.time()

    for i in range(num_classes-1):
        logging.info('\nTotal Performance: \nClass: {}: dice {:.2f}% | jaccard {:.2f}% | hd95 {:.2f} | asd {:.2f}'
                        .format(i+1, total_metric_all[i,:,0].mean()*100,
                                total_metric_all[i,:,1].mean()*100, total_metric_all[i,:,2].mean(),
                                total_metric_all[i,:,3].mean(),))
        logging.info('{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'
                        .format(total_metric_all[i,:,0].mean()*100,
                                total_metric_all[i,:,1].mean()*100, total_metric_all[i,:,2].mean(),
                                total_metric_all[i,:,3].mean(),))
        logging.info('{:.2f}({:.2f})\t{:.2f}({:.2f})\t{:.2f}({:.2f})\t{:.2f}({:.2f})'
                        .format(total_metric_all[i,:,0].mean()*100, total_metric_all[i,:,0].std()*100,
                                total_metric_all[i,:,1].mean()*100, total_metric_all[i,:,1].std()*100,
                                total_metric_all[i,:,2].mean(), total_metric_all[i,:,2].std(),
                                total_metric_all[i,:,3].mean(), total_metric_all[i,:,3].std()))
        logging.info('Inference time: {:.2f} s'.format((time_end-time_start)/len(image_list)))

    print("Validation end")
    return total_metric / len(image_list)


def test_all_case_LA_sam(net, data_dir, test_list="full_test.list", num_classes=3, 
    patch_size=(48, 160, 160), stride_xy=32, stride_z=24, AMC=False, test_save_path=None):

    with open(data_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    if 'LA' in data_dir:
        image_list = [data_dir + "/{}/mri_norm2.h5".format(
            item.replace('\n', '').split(",")[0]) for item in image_list]
    else:
        image_list = [data_dir + "/{}.h5".format(
            item.replace('\n', '').split(",")[0]) for item in image_list]
    total_metric = np.zeros((num_classes-1, 4))

    test_set = LAHeart(data_dir, split='test', patch_size=patch_size)

    print("Validation begin")
    dice = 0
    for data in tqdm(test_set):
        image = data['image'].unsqueeze(0).cuda()
        label = data['label'].unsqueeze(0).cuda()
        prediction, train_dice_sam = finetune_sam_test_predict3D(
                                image, label, net,
                                num_clicks=4, prev_masks=None)
        label = label.cpu().detach().numpy().squeeze()
        dice += train_dice_sam
        for i in range(1, num_classes):
            total_metric[i-1, :] += cal_metric(label == i, prediction == i)

    print("Validation end")
    return total_metric / len(image_list), dice

# def test_all_case_print_single(net, data_dir, test_list="full_test.list", 
#     num_classes=4, patch_size=(48, 160, 160), stride_xy=32, stride_z=24, AMC=False, 
#     test_save_path = None):
#     with open(list_dir + '/{}'.format(test_list), 'r') as f:
#         image_list = f.readlines()
#     if 'LA' in list_dir:
#         image_list = [data_dir + "/{}/mri_norm2.h5".format(
#             item.replace('\n', '').split(",")[0]) for item in image_list]
#     else:
#         image_list = [data_dir + "/{}.h5".format(
#             item.replace('\n', '').split(",")[0]) for item in image_list]
#     total_metric = np.zeros((num_classes-1, 4))
#     total_metric_all = np.zeros((num_classes-1, len(image_list), 4))

#     idx = 0
#     total_num = len(image_list)
#     print("Validation begin")
#     for image_path in tqdm(image_list):
#         h5f = h5py.File(image_path, 'r')
#         image = h5f['image'][:]
#         label = h5f['label'][:]
#         prediction = test_single_case(
#             net, image, stride_xy, stride_z, patch_size, num_classes=num_classes, AMC=AMC)

#         logging.info('\n[{}/{}] {}%:\t{}'.format(idx+1,total_num,round((idx+1)/total_num*100),image_path))

#         for i in range(1, num_classes):
#             score = cal_metric(label == i, prediction == i)
#             total_metric[i-1, :] += score
#             total_metric_all[i-1, idx, :] = score
#             logging.info('Class: {}: dice {:.2f}% | jaccard {:.2f}% | hd95 {:.2f} | asd {:.2f}'
#                             .format(i, score[0]*100,score[1]*100,score[2],score[3]))

#         if test_save_path:
#             if 'LA' in list_dir:
#                 image_id = image_path.split('/')[-2]
#             else:
#                 image_id = image_path.split('/')[-1][:-3]
#             np.save(os.path.join(test_save_path, image_id+'.npy'), 
#                     {'img': image, 'lab':label, 'pred': prediction})

#         idx += 1
#     total_metric = total_metric / len(image_list)

#     avg = np.array([total_metric[0, 0].mean()*100, total_metric[0, 1].mean()*100, total_metric[0, 2].mean(), total_metric[0, 3].mean()])

#     std = np.array([total_metric_all[0, :, 0].std()*100, total_metric_all[0, :, 1].std()*100, total_metric_all[0, :, 2].std(), total_metric_all[0, :, 3].std()])


#     logging.info('Avg:\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(
#             avg[0], avg[1], avg[2], avg[3]))

#     logging.info('Std:\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(
#             std[0], std[1], std[2], std[3]))

#     # logging.info('avg: {}\navg shape: {}\nstd: {}\nstd shape: {}'.format(avg, avg.shape, std, std.shape))
#     logging.info("Validation end")
#     return total_metric, total_metric_all, avg, std
