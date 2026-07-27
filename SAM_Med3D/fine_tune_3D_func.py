import os
join = os.path.join
import numpy as np
from glob import glob
import torch
import sys
sys.path.append('.')
import random

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


from dataloaders.brats19 import (BraTS2019, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from dataloaders.LeftAtrium import LAHeart
from dataloaders.pancreas import Pancreas
from dataloaders.AortaDissection import AortaDissection
import loralib as lora
from medpy import metric
import logging

click_methods = {
    'default': get_next_click3D_torch_ritm,
    'ritm': get_next_click3D_torch_ritm,
    'random': get_next_click3D_torch_2,
    'uncertainty': get_next_click3D_torch_unc,
    'uncertainty_unlb': get_next_click3D_torch_unc_unlab,
}

norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)

def finetune_model_predict3D(img3D, gt3D, sam_model_tune, args, device='cuda', 
                                click_method='random', click_method_aft='random', num_clicks=10, prev_masks=None):

    img3D = norm_transform(img3D.squeeze(dim=1)) # (N, C, W, H, D)
    img3D = img3D.unsqueeze(dim=1)

    if prev_masks is None:
        prev_masks = torch.zeros_like(gt3D).to(device)
    low_res_masks = F.interpolate(prev_masks.float(), size=(args.crop_size//4,args.crop_size//4,args.crop_size//4))

    with torch.no_grad():
        image_embedding = sam_model_tune.image_encoder(img3D.to(device)) # (1, 384, 16, 16, 16)
    for num_click in range(num_clicks):
        with torch.no_grad():
            if(num_click>0):
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
        medsam_seg = medsam_seg_prob.cpu().clone().detach().numpy().squeeze(1)
        medsam_mask = (medsam_seg > 0.5).astype(np.uint8)
    one_hot_medsam_seg_prob = torch.cat([1-medsam_seg_prob, medsam_seg_prob], dim=1)
    return prev_masks, medsam_mask


def compute_iou(pred_mask, gt_semantic_seg):
    in_mask = np.logical_and(gt_semantic_seg, pred_mask)
    out_mask = np.logical_or(gt_semantic_seg, pred_mask)
    iou = np.sum(in_mask) / np.sum(out_mask)
    return iou

def compute_dice(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.
    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum

def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred,gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred,gt)
        return np.array([dice, jc, hd95, asd])
    else:
        return np.zeros(4)

def load_dataset(args):
    if 'LA' in args.data_dir:
        return LAHeart(args.data_dir, split='test', patch_size=args.patch_size, SAM=True)
    if 'Pancreas' in args.data_dir:
        return Pancreas(args.data_dir, split='test', patch_size=args.patch_size, SAM=True)
    if 'BraTS' in args.data_dir:
        return BraTS2019(args.data_dir, split='test', patch_size=args.patch_size, SAM=True)
    if 'TBAD' in args.data_dir:
        return AortaDissection(args.data_dir, split='test', patch_size=args.patch_size, SAM=True)


def test_all_case(args, sam_model_tune, print_single=False):

    test_dataset = load_dataset(args)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        sampler=None,
        batch_size=1, 
        shuffle=False
    )

    sam_trans = ResizeLongestSide3D(sam_model_tune.image_encoder.img_size)

    total_num = len(test_dataloader)
    score_array = np.zeros((1, 4))
    score_array_all = np.zeros((total_num, 4))

    idx = 0
    for batch_data in tqdm(test_dataloader):
    # for i_batch, batch_data in enumerate(test_dataloader):
        image3D, gt3D, img_name = batch_data
        sz = image3D.size()
        if(sz[2]<args.crop_size or sz[3]<args.crop_size or sz[4]<args.crop_size):
            print("[ERROR] wrong size", sz, "for", img_name)
            
        seg_pred, seg_mask = finetune_model_predict3D(
            image3D, gt3D, sam_model_tune, args, device='cuda', 
            click_method=args.point_method, num_clicks=args.num_clicks, 
            prev_masks=None)
        score = cal_metric(gt3D.cpu().clone().detach().numpy().squeeze(), seg_mask)
        score_array[0, :] += score
        score_array_all[idx, :] = score
        idx += 1

        if print_single:
            logging.info('[{}/{}]\t{}:\ndice: {:.2f}%\tjaccard: {:.2f}%\thd95: {:.2f}\tassd: {:.2f}'.format(
                idx, total_num, img_name, score[0]*100, score[1]*100, score[2], score[3],
            ))
    return score_array/total_num, score_array_all


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/amax/data/luwenjing/P1_UPCoL/Datasets/LA_dataset', help='Name of Experiment')
    parser.add_argument('--patch_size', type=list,  default=[128,128,128],
                        help='patch size of network input')
    parser.add_argument('--seed', type=int,  default=1337, help='random seed')

    parser.add_argument('--labeled_num', type=int, default=10,
                        help='labeled data')

    # sam 
    parser.add_argument('-pm', '--point_method', type=str, default='random')
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('-nc', '--num_clicks', type=int, default=5)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--checkpoint_path', type=str, default='/amax/data/luwenjing/P3_SemiMedSAM/codes/SAM-Med3D/ckpt/sam_med3d_turbo.pth')

    args = parser.parse_args()

    # args.checkpoint_path = '/amax/data/luwenjing/P3_SemiMedSAM/codes/results/SAM-Med3D/LA-lora-sam/LA_lab-16-adw-outupscal-hypermlp-lr1e-05-fixbuild3/best_sam_model.pth'
    args.checkpoint_path = '/amax/data/luwenjing/P3_SemiMedSAM/codes/Semi-MedSAM/results/LA_lab-4/vnet_amc_sam_al_unlab/cons-0.5_adw-adw_lr-1e-05-wd-0.1/best_sam_model.pth'
    snapshot_path = os.path.dirname(args.checkpoint_path)
    # snapshot_path = '/amax/data/luwenjing/P3_SemiMedSAM/codes/results/SAM-Med3D/LA'
    logging.basicConfig(filename=snapshot_path+"/pm-{}_nc-{}_log.txt".format(args.point_method, args.num_clicks), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    sam_model_tune = sam_model_registry3D['vit_b_ori'](checkpoint=None).cuda()
    # lora.mark_only_lora_as_trainable(sam_model_tune, bias='lora_only')
    if args.checkpoint_path is not None:
        model_dict = torch.load(args.checkpoint_path, map_location=device)
        # state_dict = model_dict['model_state_dict']
        # sam_model_tune.load_state_dict(state_dict, strict=False)
        # sam_model_tune.load_state_dict(state_dict)
        sam_model_tune.load_state_dict(model_dict)

    for n, p in sam_model_tune.named_parameters():
        print(n, p.requires_grad)
    #     p.requires_grad = False
    # for n, p in sam_model_tune.named_parameters():
    #     if "output_upscaling" in n:
    #         p.requires_grad = True
    #     if "output_hypernetworks_mlps" in n:
    #         p.requires_grad = True

    # avg_metric, avg_metric_all = test_all_case(args, sam_model_tune, print_single=True)
    # logging.info(
    #     'dice_score : {:.2f}% jaccard : {:.2f}%  hd95 : {:.2f} assd : {:.2f}'.format(
    #         avg_metric_all[:, 0].mean()*100, avg_metric_all[:, 1].mean()*100,
    #         avg_metric_all[:, 2].mean(), avg_metric_all[:, 3].mean()))
    # logging.info(
    #     '{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(
    #         avg_metric_all[:, 0].mean()*100, avg_metric_all[:, 1].mean()*100,
    #         avg_metric_all[:, 2].mean(), avg_metric_all[:, 3].mean()))
    # logging.info(
    #     '{:.2f}({:.2f})\t{:.2f}({:.2f})\t{:.2f}({:.2f})\t{:.2f}({:.2f})'.format(
    #         avg_metric_all[:, 0].mean()*100, avg_metric_all[:, 0].std()*100, 
    #         avg_metric_all[:, 1].mean()*100, avg_metric_all[:, 1].std()*100, 
    #         avg_metric_all[:, 2].mean(), avg_metric_all[:, 2].std(), 
    #         avg_metric_all[:, 3].mean(), avg_metric_all[:, 3].std(), ))