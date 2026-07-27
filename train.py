import argparse
import logging
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append('./')
from dataloaders.brats19 import TwoStreamBatchSampler

from networks.vnet_amc import VNet_AMC
from utils import losses
from utils.val_3D import test_all_case
from utils.utils import *

from SAM_Med3D.segment_anything.build_sam3D import sam_model_registry3D
from SAM_Med3D.fine_tune_3D_func import finetune_model_predict3D

from utils.StochSegLoss import StochasticSegmentationNetworkLossMCIntegral
from utils.stochastic_deepmedic import StochasticDeepMedic

from monai.losses import DiceCELoss


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='Data directory')
parser.add_argument('--save_path', type=str, help='Path to save results')
parser.add_argument('--num_classes', type=int,
                    default=2, help='number of classes')

parser.add_argument('--method', type=str,
                    default='UPSAM', help='method_name')
parser.add_argument('--stage', type=str,
                    default='pretrain', choices=['pretrain', 'semi_train'], help='stage_name: pretrain or semi_train')

parser.add_argument('--dataset', type=str,
                    default='LA', help='dataset_name: LA or Pancreas')
parser.add_argument('--max_iterations', type=int,
                    default=7500, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=1e-5,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[128,128,128],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=4,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--weight_decay', type=float,  default=0.1, help='ema_decay')
parser.add_argument('--al_weight', type=float,  default=0.1, help='ema_decay')
parser.add_argument('--sam_weight', type=float,  default=1.0, help='ema_decay')
parser.add_argument('--consistency', type=float,
                    default=1.0, help='consistency')

# sam 
parser.add_argument('-pm', '--point_method', type=str, default='random')
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('-nc', '--num_clicks', type=int, default=5)
parser.add_argument('--gpu', type=str, default='0')

# checkpoints
parser.add_argument('--checkpoint_path_sam', type=str)
parser.add_argument('--checkpoint_path_vnet_amc', type=str)


args = parser.parse_args()
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda")


def pretrain(args, snapshot_path):
    base_lr = args.base_lr
    if args.labeled_num < args.batch_size:
        args.batch_size = args.labeled_num
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = VNet_AMC(n_channels=1, n_classes=args.num_classes, n_branches=4).cuda()

    db_train, test_list  = load_dataset(args, pretrain=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(args.num_classes)
    focal_loss = losses.FocalLoss()
    iou_loss = losses.SoftIoULoss(nclass=args.num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            # labeled data
            # VNet branch
            outputs = model(volume_batch)
            loss_ce = ce_loss(outputs[0], label_batch)
            loss_dice = dice_loss(torch.softmax(outputs[1], dim=1), label_batch)
            loss_focal = focal_loss(outputs[2], label_batch)
            loss_iou = iou_loss(outputs[3], label_batch)
            loss_seg = 0.25 * (loss_ce + loss_dice + loss_focal + loss_iou)
            loss = loss_seg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lab_masks = torch.softmax(0.25*(outputs[0] + outputs[1][args.labeled_bs] +
                                    outputs[2] + outputs[3]), dim = 1)
            train_dice_vnet = cal_dice(lab_masks, label_batch)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/loss_focal', loss_focal, iter_num)
            writer.add_scalar('info/loss_iou', loss_iou, iter_num)
            writer.add_scalar('info/loss_seg', loss_seg, iter_num)
            writer.add_scalar('val/train_dice_vnet', train_dice_vnet, iter_num)
            logging.info(
                'iteration {} : loss : {:.4f}, loss_ce: {:.4f}, loss_dice: {:.4f}, '
                'loss_focal : {:.4f}, loss_iou : {:.4f}, loss_seg : {:.4f}, train_dice_vnet : {:.4f},'.format(
                iter_num, loss.item(), loss_ce.item(), loss_dice.item(),
                loss_focal.item(), loss_iou.item(), loss_seg.item(), train_dice_vnet))

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()

                avg_metric = test_all_case(
                    model, args.data_dir, test_list=test_list, num_classes=args.num_classes, patch_size=args.patch_size,
                    stride_xy=16, stride_z=4, AMC=True)
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                for i in range(args.num_classes-1):
                    writer.add_scalar('val/class{}_val_dice_score'.format(i+1),
                                    avg_metric[i, 0], iter_num)
                    writer.add_scalar('val/class{}_val_hd95'.format(i+1),
                                    avg_metric[i, 2], iter_num)
                    logging.info(
                        'Class {}, iteration {} : dice_score : {:.2f}% jaccard : {:.2f}%  hd95 : {:.2f} assd : {:.2f}'.format(
                            i+1, iter_num, avg_metric[0, 0].mean()*100, avg_metric[0, 1].mean()*100,
                            avg_metric[0, 2].mean(), avg_metric[0, 3].mean()))

                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Pre-Training Finished!"

def semi_train(args, snapshot_path):
    base_lr = args.base_lr
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = VNet_AMC(n_channels=1, n_classes=args.num_classes, n_branches=4).cuda()
    model_dict = torch.load(args.checkpoint_path_vnet_amc, map_location=device)
    model.load_state_dict(model_dict)

    sam_model_tune = sam_model_registry3D['vit_b_ori'](checkpoint=None).cuda()
    if args.checkpoint_path_sam is not None:
        model_dict = torch.load(args.checkpoint_path_sam, map_location=device)
        state_dict = model_dict['model_state_dict']
        sam_model_tune.load_state_dict(state_dict)
    sam_model_tune = finetune_parameters(sam_model_tune)

    db_train, test_list  = load_dataset(args)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_idxs = list(range(0, args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, len(db_train)))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    sam_model_tune.train()

    optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    optimizer_sam = optim.AdamW(filter(lambda p: p.requires_grad, sam_model_tune.parameters()), lr=args.base_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    # loss func
    ## vnet loss
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(nclass=args.num_classes)
    focal_loss = losses.FocalLoss()
    iou_loss = losses.SoftIoULoss(nclass=args.num_classes)
    ## sam loss
    seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    AL_module = nn.DataParallel(StochasticDeepMedic(num_classes=args.num_classes))
    AL_module = AL_module.cuda()
    SSLoss = StochasticSegmentationNetworkLossMCIntegral(20) 

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            # labeled data
            # VNet branch
            outputs = model(volume_batch)
            loss_ce = ce_loss(outputs[0][:args.labeled_bs], label_batch[:args.labeled_bs])
            loss_dice = dice_loss(torch.softmax(outputs[1], dim=1)[:args.labeled_bs], 
                                    label_batch[:args.labeled_bs])
            loss_focal = focal_loss(outputs[2][:args.labeled_bs], label_batch[:args.labeled_bs])
            loss_iou = iou_loss(outputs[3][:args.labeled_bs], label_batch[:args.labeled_bs])
            loss_seg = 0.25 * (loss_ce + loss_dice + loss_focal + loss_iou)

            # SAM branch
            seg_pred_lab, _ = finetune_model_predict3D(
                volume_batch[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1), sam_model_tune, args, device=device, 
                click_method=args.point_method, num_clicks=args.num_clicks, 
                prev_masks=None) # torch.Size([B, 1, 128, 128, 128]) (B, 128, 128, 128)
            loss_sam = seg_loss(seg_pred_lab, label_batch[:args.labeled_bs].unsqueeze(1))

            out_pred = 0.25*(outputs[0] + outputs[1] + outputs[2] + outputs[3])
            out_masks = torch.argmax(torch.softmax(out_pred, dim = 1), dim=1)
            train_dice_vnet = cal_dice(out_masks[:args.labeled_bs], label_batch[:args.labeled_bs])

            # unlab data
            seg_pred_unlab, seg_mask_unlab = finetune_model_predict3D(
                volume_batch[args.labeled_bs:], out_masks.unsqueeze(1)[args.labeled_bs:], sam_model_tune, args, device=device, 
                click_method=args.point_method, num_clicks=args.num_clicks, 
                prev_masks=None) # torch.Size([B, 1, 128, 128, 128]) (B, 128, 128, 128)
            
            seg_mask_unlab = torch.from_numpy(seg_mask_unlab).cuda()

            # torch.Size([B, 16, 128, 128, 128]) torch.Size([B, 128, 128, 128]) 
            logits, state = AL_module(model.al_input[args.labeled_bs:], seg_mask_unlab.long())
            state.update({'target': seg_mask_unlab.long()})
            loss_con = SSLoss(logits, **state)

            consistency_weight = get_current_consistency_weight(epoch_num, max_epoch, args.consistency)
            loss = 0.5 * (loss_sam + loss_seg) + consistency_weight * loss_con
            

            optimizer.zero_grad()
            optimizer_sam.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_sam.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/loss_focal', loss_focal, iter_num)
            writer.add_scalar('info/loss_iou', loss_iou, iter_num)
            writer.add_scalar('info/loss_seg', loss_seg, iter_num)
            writer.add_scalar('info/loss_sam', loss_sam, iter_num)
            writer.add_scalar('info/loss_con', loss_con, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('val/train_dice_vnet', train_dice_vnet, iter_num)
            logging.info(
                'iteration {} : loss : {:.4f}, loss_ce: {:.4f}, loss_dice: {:.4f}, '
                'loss_focal : {:.4f}, loss_iou : {:.4f}, loss_seg : {:.4f}, '
                'loss_sam : {:.4f}, loss_con : {:.4f}, train_dice_vnet : {:.4f},'.format(
                iter_num, loss.item(), loss_ce.item(), loss_dice.item(),
                loss_focal.item(), loss_iou.item(), loss_seg.item(), 
                loss_sam.item(), loss_con.item(), train_dice_vnet))

            if iter_num > 0 and iter_num % 40 == 0:
                model.eval()
                sam_model_tune.eval()

                avg_metric = test_all_case(
                    model, args.data_dir, test_list=test_list, num_classes=args.num_classes, patch_size=args.patch_size,
                    stride_xy=16, stride_z=4, AMC=True)
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path, 'best_model.pth')
                    save_sam_ckpt_path = os.path.join(snapshot_path, 'best_sam_model.pth')

                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                    torch.save(sam_model_tune.state_dict(), save_sam_ckpt_path)

                for i in range(args.num_classes-1):
                    writer.add_scalar('val/class{}_val_dice_score'.format(i+1),
                                    avg_metric[i, 0], iter_num)
                    writer.add_scalar('val/class{}_val_hd95'.format(i+1),
                                    avg_metric[i, 2], iter_num)
                    logging.info(
                        'Class {}, iteration {} : dice_score : {:.2f}% jaccard : {:.2f}%  hd95 : {:.2f} assd : {:.2f}'.format(
                            i+1, iter_num, avg_metric[0, 0].mean()*100, avg_metric[0, 1].mean()*100,
                            avg_metric[0, 2].mean(), avg_metric[0, 3].mean()))

                model.train()
                sam_model_tune.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Semi-Training Finished!"


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

    snapshot_path = os.path.join(args.save_path, "{}_lab-{}/{}/{}".format(
        args.dataset, args.labeled_num, args.method, args.stage))
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/codes'):
        shutil.rmtree(snapshot_path + '/codes')
    shutil.copytree('.', snapshot_path + '/codes',
                    shutil.ignore_patterns(['.git', '__pycache__']))
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s')

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(snapshot_path+'/'+'snapshot.log', encoding='utf8')
    fh.setFormatter(formatter) 
    logger.addHandler(fh)
    logging.info('Code launch on device: {}'.format(device))
    logging.info(str(args))
    logging.info('Save at : {}'.format(snapshot_path))

    if args.stage == 'pretrain':
        pretrain(args, snapshot_path)
    else:
        semi_train(args, snapshot_path)
