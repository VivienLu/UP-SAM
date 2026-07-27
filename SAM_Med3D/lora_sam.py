import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

sys.path.append('.')
from dataloaders.brats19 import (BraTS2019, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from dataloaders.LeftAtrium import LAHeart
from dataloaders.pancreas import Pancreas
from dataloaders.AortaDissection import AortaDissection

# from networks.vnet_amc import VNet_AMC
from utils import losses, ramps
# from utils.val_3D import test_all_case

from SAM_Med3D.segment_anything.build_sam3D import sam_model_registry3D
# from SAM_Med3D.lora_sam_med3d import finetune_model_predict3D_lab, finetune_model_predict3D_unlab
from SAM_Med3D.fine_tune_3D_func import test_all_case, finetune_model_predict3D

import loralib as lora
from medpy import metric
from monai.losses import DiceCELoss

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str,
                    default='/amax/data/luwenjing/P1_UPCoL/Datasets/LA_dataset', help='Name of Experiment')
parser.add_argument('--save_path', type=str,
                    default='/amax/data/luwenjing/P3_SemiMedSAM/codes/results/SAM-Med3D/LA-lora-sam', help='Name of Experiment')
parser.add_argument('--num_classes', type=int,
                    default=2, help='number of classes')

parser.add_argument('--exp', type=str,
                    default='LA', help='experiment_name')
parser.add_argument('--max_iterations', type=int,
                    default=3000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=0,
                    help='whether use deterministic training')

parser.add_argument('--base_lr', type=float,  default=1e-5,
                    help='segmentation network learning rate')
parser.add_argument('--momentum', type=float,  default=0.9,
                    help='segmentation network learning rate')
parser.add_argument('--weight_decay', type=float,  default=1e-7,
                    help='segmentation network learning rate')
parser.add_argument('--gamma', type=float, default=0.1)

parser.add_argument('--patch_size', type=list,  default=[128,128,128],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')

# label and unlabel
parser.add_argument('--labeled_num', type=int, default=4,
                    help='labeled data')
# sam 
parser.add_argument('-pm', '--point_method', type=str, default='random')
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('-nc', '--num_clicks', type=int, default=5)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--checkpoint_path', type=str, default='/amax/data/luwenjing/P3_SemiMedSAM/codes/SAM-Med3D/ckpt/sam_med3d_turbo.pth')

args = parser.parse_args()
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda")

def load_dataset(args):
    if 'LA' in args.data_dir:
        return LAHeart(args.data_dir, split='train', patch_size=args.patch_size, num=args.labeled_num, SAM=True)
    if 'Pancreas' in args.data_dir:
        return Pancreas(args.data_dir, split='train', patch_size=args.patch_size, num=args.labeled_num, SAM=True)
    if 'BraTS' in args.data_dir:
        return BraTS2019(args.data_dir, split='train', patch_size=args.patch_size, num=args.labeled_num, SAM=True)
    if 'TBAD' in args.data_dir:
        return AortaDissection(args.data_dir, split='train', patch_size=args.patch_size, num=args.labeled_num, SAM=True)

def train(args, snapshot_path):
    base_lr = args.base_lr
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = args.num_classes

    # model = VNet_AMC(n_channels=1, n_classes=args.num_classes, n_branches=4).cuda()
    sam_model_tune = sam_model_registry3D['vit_b_ori'](checkpoint=None).cuda()
    # lora.mark_only_lora_as_trainable(sam_model_tune, bias='lora_only')
    if args.checkpoint_path is not None:
        model_dict = torch.load(args.checkpoint_path, map_location=device)
        state_dict = model_dict['model_state_dict']
        # sam_model_tune.load_state_dict(state_dict, strict=False)
        sam_model_tune.load_state_dict(state_dict)

    for n, p in sam_model_tune.named_parameters():
        p.requires_grad = False

    for n, p in sam_model_tune.named_parameters():
        if "output_upscaling" in n:
            p.requires_grad = True
        if "output_hypernetworks_mlps" in n:
            p.requires_grad = True
        if "iou_prediction_head" in n:
            p.requires_grad = True
        if "iou_token" in n:
            p.requires_grad = True
        if "mask_tokens" in n:
            p.requires_grad = True

    db_train  = load_dataset(args)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # trainloader = DataLoader(db_train, batch_sampler=None,
    #                          num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    trainloader = DataLoader(
        dataset=db_train, sampler=None, batch_size=args.batch_size, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)

    sam_model_tune.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(args.num_classes)
    seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1

    # optimizer = optim.SGD(sam_model_tune.parameters(), lr=base_lr,
    #                       momentum=args.momentum, weight_decay=args.weight_decay)

    # optimizer = torch.optim.AdamW(sam_model_tune.parameters(), lr=args.base_lr, betas=(0.9,0.999), weight_decay=args.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[round(0.6*max_epoch), round(0.9*max_epoch)], args.gamma)
    # optimizer = optim.Adam(sam_model_tune.parameters(), lr=args.base_lr)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, sam_model_tune.parameters()), lr=args.base_lr, betas=(0.9, 0.999), weight_decay=0.1)

    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            # volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = sampled_batch
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            # SAM branch
            seg_pred, seg_mask = finetune_model_predict3D(
                volume_batch, label_batch, sam_model_tune, args, device=device, 
                click_method=args.point_method, num_clicks=args.num_clicks, 
                prev_masks=None)
            # label_batch = label_batch.squeeze()
            # loss_ce_sam = ce_loss(seg_pred, label_batch)
            # loss_dice_sam = dice_loss(torch.softmax(seg_pred, dim=1), label_batch)
            # loss_sam = 0.5*(loss_ce_sam + loss_dice_sam) 
            loss_sam = seg_loss(seg_pred, label_batch)

            # torch.set_grad_enabled(True)
            # loss_sam.requires_grad = True
            optimizer.zero_grad()
            loss_sam.backward()
            optimizer.step()
            # lr_scheduler.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            # writer.add_scalar('info/lr', lr_scheduler.get_last_lr(), iter_num)
            writer.add_scalar('info/lr', lr_, iter_num)
            # writer.add_scalar('info/loss_ce_sam', loss_ce_sam, iter_num)
            # writer.add_scalar('info/loss_dice_sam', loss_dice_sam, iter_num)
            writer.add_scalar('info/loss_sam', loss_sam, iter_num)

            # logging.info(
            #     'iteration {} : loss_ce: {:.4f}, loss_dice: {:.4f}, '
            #     'loss_sam : {:.4f}'.format(
            #     iter_num, loss_ce_sam.item(), loss_dice_sam.item(),
            #     loss_sam.item()))


            logging.info(
                'iteration {} : loss_sam : {:.4f}'.format(
                iter_num, loss_sam.item()))

            if iter_num > 0 and iter_num % 40 == 0:
                sam_model_tune.eval()

                avg_metric, _ =  test_all_case(args, sam_model_tune)
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_sam_ckpt_path = os.path.join(snapshot_path, 'best_sam_model.pth')
                    torch.save(sam_model_tune.state_dict(), save_sam_ckpt_path)

                for i in range(args.num_classes-1):
                    writer.add_scalar('val/class{}_val_dice_score'.format(i+1),
                                    avg_metric[i, 0], iter_num)
                    writer.add_scalar('val/class{}_val_jaccard'.format(i+1),
                                    avg_metric[i, 1], iter_num)
                    writer.add_scalar('val/class{}_val_hd95'.format(i+1),
                                    avg_metric[i, 2], iter_num)
                    writer.add_scalar('val/class{}_val_assd'.format(i+1),
                                    avg_metric[i, 3], iter_num)
                    logging.info(
                        'Class {}, iteration {} : dice_score : {:.2f}% jaccard : {:.2f}%  hd95 : {:.2f} assd : {:.2f}'.format(
                            i+1, iter_num, avg_metric[0, 0].mean()*100, avg_metric[0, 1].mean()*100,
                            avg_metric[0, 2].mean(), avg_metric[0, 3].mean()))

                sam_model_tune.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # -hypermlp-iouhead-token
    snapshot_path = os.path.join(args.save_path, "{}_lab-{}-adw-fix-outupscal-hypermlp-iouhead-token-tokens-lr{}-fixbuild3".format(
        args.exp, args.labeled_num, args.base_lr))
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    # logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
    #                     format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info(str(args))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s')

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(snapshot_path+'/'+'snapshot.log', encoding='utf8')
    fh.setFormatter(formatter) 
    logger.addHandler(fh)
    logging.info(str(args))

    train(args, snapshot_path)
