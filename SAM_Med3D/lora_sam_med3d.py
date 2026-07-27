import os
join = os.path.join
import numpy as np
from glob import glob
import torch
from SAM_Med3D.segment_anything.build_sam3D import sam_model_registry3D
from SAM_Med3D.segment_anything.utils.transforms3D import ResizeLongestSide3D
from SAM_Med3D.segment_anything import sam_model_registry
from tqdm import tqdm
import argparse
import SimpleITK as sitk
import torch.nn.functional as F
from torch.utils.data import DataLoader
import SimpleITK as sitk
import torchio as tio
import numpy as np
from collections import OrderedDict, defaultdict
import json
import pickle
from SAM_Med3D.utils.click_method import (get_next_click3D_torch_ritm, 
                                            get_next_click3D_torch_2, 
                                            get_next_click3D_torch_unc,
                                            get_next_click3D_torch_unc_unlab)

      
click_methods = {
    'default': get_next_click3D_torch_ritm,
    'ritm': get_next_click3D_torch_ritm,
    'random': get_next_click3D_torch_2,
    'uncertainty': get_next_click3D_torch_unc,
    'uncertainty_unlb': get_next_click3D_torch_unc_unlab,
}


def compute_dice(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.
    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    dice = 0
    for i in range(mask_gt.shape[0]):
        volume_sum = mask_gt[i].sum() + mask_pred[i].sum()
        if volume_sum == 0:
            return np.NaN
        volume_intersect = (mask_gt[i] & mask_pred[i]).sum()
        dice += 2*volume_intersect / volume_sum
    return dice/mask_gt.shape[0]

def finetune_model_predict3D_lab(img_batch, lab_batch, 
                                 sam_model_tune, num_classes, loss_func,  
                                 num_clicks=4, 
                                 device='cuda', crop_size=128, 
                                 click_method='random', click_method_aft='uncertainty',
                                 prev_masks=None):
    '''
    img_batch: (B, 1, W, H, D)
    lab_batch: (B, W, H, D)
    '''
    norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)

    labeled_batch_binary_mask = F.one_hot(lab_batch, num_classes=num_classes)[..., 1:]
    labeled_batch_binary_mask = labeled_batch_binary_mask.permute(0, 4, 1, 2, 3) # （B, C-1, W, H, D) 
    seg_prob = torch.zeros_like(labeled_batch_binary_mask).to(device) # [2, 1, 128, 128, 128]
    dice_score = 0.0
    loss_sam = 0
    for class_num  in range(num_classes-1): # C-1
        img3D = norm_transform(img_batch.squeeze(dim=1)) # (B, 1, W, H, D)
        img3D = img3D.unsqueeze(dim=1)
        gt3D = labeled_batch_binary_mask[:, class_num, :, :, :].unsqueeze(1) # (B, 1, W, H, D)
        if prev_masks is None:
            prev_masks = torch.zeros_like(gt3D).to(device)
        low_res_masks = F.interpolate(prev_masks.float(), size=(crop_size//4,
                                      crop_size//4,crop_size//4))

        with torch.no_grad():
            image_embedding = sam_model_tune.image_encoder(img3D.to(device)) # (1, 384, 16, 16, 16)

        for num_click in range(num_clicks):
            with torch.no_grad():
                if(num_click>1):
                    click_method = click_method_aft
                batch_points, batch_labels = click_methods[click_method](prev_masks.to(device), gt3D.to(device))

                points_co = torch.cat(batch_points, dim=0).to(device)  
                points_la = torch.cat(batch_labels, dim=0).to(device)  

                sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
                    points=[points_co, points_la],
                    boxes=None,
                    masks=low_res_masks.to(device),
                )
                low_res_masks, _ = sam_model_tune.mask_decoder(
                    image_embeddings=image_embedding.to(device), # (B, 384, 64, 64, 64)
                    image_pe=sam_model_tune.prompt_encoder.get_dense_pe(), # (1, 384, 64, 64, 64)
                    sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 384)
                    dense_prompt_embeddings=dense_embeddings, # (B, 384, 64, 64, 64)
                    multimask_output=False,
                    )
                prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], 
                                mode='trilinear', align_corners=False) # (B, 1, W, H, D)
                medsam_seg_prob = torch.sigmoid(prev_masks) # (B, 1, W, H, D)
                medsam_seg = (medsam_seg_prob.cpu().numpy().squeeze() > 0.5).astype(np.uint8)
        dice_score += round(compute_dice(gt3D.detach().cpu().numpy().astype(np.uint8), medsam_seg), 4)
        seg_prob[:, class_num, :, :, :] = prev_masks.squeeze() # （B, C-1, W, H, D)
        loss_sam += loss_func(prev_masks, gt3D)

    seg_pred_bg = 1 - seg_prob.max(dim=1, keepdim=True)[0] # (B, 1, W, H, D)
    seg_pred_all = torch.cat((seg_pred_bg, seg_prob), dim=1)  # （B, C, W, H, D)

    return seg_pred_all.float(), dice_score/(num_classes-1), loss_sam/(num_classes-1)

def finetune_sam_train_predict3D(img3D, gt3D, sam_model_tune,
                            device='cuda', crop_size=128,
                            click_method='random', click_method_aft='uncertainty',
                            num_clicks=10, prev_masks=None):
    norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
    img3D = norm_transform(img3D.squeeze(dim=1)) # (N, C, W, H, D)
    img3D = img3D.unsqueeze(dim=1)
    gt3D = gt3D.unsqueeze(dim=1)
    
    dice_list = []
    if prev_masks is None:
        prev_masks = torch.zeros_like(gt3D).to(device)
    low_res_masks = F.interpolate(prev_masks.float(), size=(crop_size//4, crop_size//4, crop_size//4))

    with torch.no_grad():
        image_embedding = sam_model_tune.image_encoder(img3D.to(device)) # (1, 384, 16, 16, 16)
    for num_click in range(num_clicks):
        # with torch.no_grad():
        if(num_click>1):
            click_method = click_method_aft
        batch_points, batch_labels = click_methods[click_method](prev_masks.to(device), gt3D.to(device))

        points_co = torch.cat(batch_points, dim=0).to(device)  
        points_la = torch.cat(batch_labels, dim=0).to(device)  

        sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
            points=[points_co, points_la],
            boxes=None,
            masks=low_res_masks.to(device),
        )
        low_res_masks, _ = sam_model_tune.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 384, 64, 64, 64)
            image_pe=sam_model_tune.prompt_encoder.get_dense_pe(), # (1, 384, 64, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 384)
            dense_prompt_embeddings=dense_embeddings, # (B, 384, 64, 64, 64)
            multimask_output=False,
            )
        prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)

        medsam_seg_prob = torch.sigmoid(prev_masks)  # (B, 1, 128, 128, 128)
        medsam_seg = (medsam_seg_prob > 0.5).float()
        # medsam_seg = medsam_seg.contiguous()

    return prev_masks, medsam_seg


def finetune_sam_test_predict3D(img3D, gt3D, sam_model_tune,
                            device='cuda', crop_size=128,
                            click_method='random', click_method_aft='uncertainty',
                            num_clicks=10, prev_masks=None):
    norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
    img3D = norm_transform(img3D.squeeze(dim=1)) # (N, C, W, H, D)
    img3D = img3D.unsqueeze(dim=1)
    gt3D = gt3D.unsqueeze(dim=1)
    
    dice_list = []
    if prev_masks is None:
        prev_masks = torch.zeros_like(gt3D).to(device)
    low_res_masks = F.interpolate(prev_masks.float(), size=(crop_size//4, crop_size//4, crop_size//4))

    with torch.no_grad():
        image_embedding = sam_model_tune.image_encoder(img3D.to(device)) # (1, 384, 16, 16, 16)
    for num_click in range(num_clicks):
        with torch.no_grad():
            if(num_click>1):
                click_method = click_method_aft
            batch_points, batch_labels = click_methods[click_method](prev_masks.to(device), gt3D.to(device))

            points_co = torch.cat(batch_points, dim=0).to(device)  
            points_la = torch.cat(batch_labels, dim=0).to(device)  

            sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
                points=[points_co, points_la],
                boxes=None,
                masks=low_res_masks.to(device),
            )
            low_res_masks, _ = sam_model_tune.mask_decoder(
                image_embeddings=image_embedding.to(device), # (B, 384, 64, 64, 64)
                image_pe=sam_model_tune.prompt_encoder.get_dense_pe(), # (1, 384, 64, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 384)
                dense_prompt_embeddings=dense_embeddings, # (B, 384, 64, 64, 64)
                multimask_output=False,
                )
            prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)

            medsam_seg_prob = torch.sigmoid(prev_masks)  # (B, 1, 64, 64, 64)
            # convert prob to mask
            medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
            medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
            # pred_list.append(medsam_seg)
            # iou_list.append(round(compute_iou(medsam_seg, gt3D[0][0].detach().cpu().numpy()), 4))
            dice_list.append(round(compute_dice(gt3D[0][0].detach().cpu().numpy().astype(np.uint8), medsam_seg), 4))

    return medsam_seg, dice_list[-1]

def finetune_model_predict3D_unlab(img_batch, vnet_pred_batch, 
                                 sam_model_tune, num_classes, 
                                 num_clicks=4, 
                                 device='cuda', crop_size=128, 
                                 click_method='random', click_method_aft='uncertainty_unlb',
                                 prev_masks=None):
    '''
    img_batch: (B, 1, W, H, D)
    vnet_pred_batch: (B, C, W, H, D)
    '''

    norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)

    lab_batch = torch.argmax(vnet_pred_batch, dim=1) # (B, 1, W, H, D)
    labeled_batch_binary_mask = F.one_hot(lab_batch, num_classes=num_classes)[..., 1:]  # （B, C-1, W, H, D)
    labeled_batch_binary_mask = labeled_batch_binary_mask.permute(0, 4, 1, 2, 3) # （B, C-1, W, H, D)
    seg_prob = torch.zeros_like(labeled_batch_binary_mask).to(device)
    for class_num  in range(num_classes-1): # C-1
        img3D = norm_transform(img_batch.squeeze(dim=1)) # (B, 1, W, H, D)
        img3D = img3D.unsqueeze(dim=1)
        gt3D = labeled_batch_binary_mask[:, class_num, :, :, :].unsqueeze(1) # (B, 1, W, H, D)
        pred3D = torch.sigmoid(vnet_pred_batch[:, class_num+1, :, :, :]).unsqueeze(1)

        if prev_masks is None:
            prev_masks = torch.zeros_like(gt3D).to(device)
        low_res_masks = F.interpolate(prev_masks.float(), size=(crop_size//4,
                                      crop_size//4,crop_size//4))

        with torch.no_grad():
            image_embedding = sam_model_tune.image_encoder(img3D.to(device)) # (1, 384, 16, 16, 16)

        for num_click in range(num_clicks):
            with torch.no_grad():
                if(num_click>1):
                    click_method = click_method_aft
                if click_method_aft == 'uncertainty_unlb':
                    batch_points, batch_labels = \
                                click_methods[click_method](prev_masks.to(device), pred3D.to(device))
                else:
                    batch_points, batch_labels = \
                                click_methods[click_method](prev_masks.to(device), gt3D.to(device))

                points_co = torch.cat(batch_points, dim=0).to(device)  
                points_la = torch.cat(batch_labels, dim=0).to(device)  

                sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
                    points=[points_co, points_la],
                    boxes=None,
                    masks=low_res_masks.to(device),
                )
                low_res_masks, _ = sam_model_tune.mask_decoder(
                    image_embeddings=image_embedding.to(device), # (B, 384, 64, 64, 64)
                    image_pe=sam_model_tune.prompt_encoder.get_dense_pe(), # (1, 384, 64, 64, 64)
                    sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 384)
                    dense_prompt_embeddings=dense_embeddings, # (B, 384, 64, 64, 64)
                    multimask_output=False,
                    )
                prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], 
                                mode='trilinear', align_corners=False) # (B, 1, W, H, D)
                medsam_seg_prob = torch.sigmoid(prev_masks).squeeze() # (B, 1, W, H, D)
        seg_prob[:, class_num, :, :, :] = medsam_seg_prob # （B, C-1, W, H, D)

    seg_pred_bg = 1 - seg_prob.max(dim=1, keepdim=True)[0] # (B, 1, W, H, D)
    seg_pred_all = torch.cat((seg_pred_bg, seg_prob), dim=1)  # （B, C, W, H, D)
    return seg_pred_all.float()
