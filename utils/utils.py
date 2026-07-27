import torch
from utils import ramps

from dataloaders.LeftAtrium import LAHeart
from dataloaders.pancreas import Pancreas


def get_current_consistency_weight(epoch, max_epoch, consistency):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * ramps.sigmoid_rampup(epoch, max_epoch)


def load_dataset(args, pretrain=False):
    test_list = 'test.txt'
    if pretrain:
        labnum = args.labeled_num
    else:
        labnum = None
    if 'LA' in args.data_dir:
        return LAHeart(args.data_dir, split='train', patch_size=args.patch_size, num=labnum), test_list
    if 'Pancreas' or 'Pancreas-CT' in args.data_dir:
        return Pancreas(args.data_dir, split='train', patch_size=args.patch_size, num=labnum), test_list


def cal_dice(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    return (2 * intersect + smooth) / (z_sum + y_sum + smooth)


def finetune_parameters(sam_model_tune):
    for n, p in sam_model_tune.named_parameters():
        p.requires_grad = False
    for n, p in sam_model_tune.named_parameters():
        if "output_upscaling" in n:
            p.requires_grad = True
        if "output_hypernetworks_mlps" in n:
            p.requires_grad = True
    return sam_model_tune