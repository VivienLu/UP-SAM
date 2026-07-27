import argparse
import os
import shutil
from glob import glob
import numpy

import torch
import torch.nn as nn

import sys
import logging

from networks.vnet_amc import VNet_AMC
from networks.vnet import VNet
from utils.val_3D import test_all_case_print_single
import  numpy as np
from pathlib import Path

def Inference(FLAGS, model_path):
    snapshot_path = os.path.dirname(model_path)
    num_classes = 2
    test_save_path = os.path.join(snapshot_path, 'test_prediction')
    os.makedirs(test_save_path, exist_ok=True)

    logging.basicConfig(filename=test_save_path + "/record_log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info('Split path: {}\nModel path: {}'.format(FLAGS.data_dir, model_path))
    net = VNet_AMC(n_channels=1, n_classes=num_classes, n_branches = 4).cuda()

    net.load_state_dict(torch.load(model_path))
    print("init weight from {}".format(model_path))
    net.eval()
    avg_metric =  test_all_case_print_single(
                        net, FLAGS.data_dir, test_list="test.txt", num_classes=2, 
                        patch_size=FLAGS.patch_size,
                        stride_xy=16, stride_z=4, AMC=True, test_save_path=test_save_path,
                        save_result=True)

    return avg_metric

def load_net_opt(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    logging.info('Loaded from {}'.format(path))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str,
                        help="Path to the dataset.")
    parser.add_argument("--list_dir", type=str,
                        help="Paths to cross-validated datasets, list of test sets and all training sets (including all labeled and unlabeled samples)")
    parser.add_argument('--model_path', type=str, help='model_path')

    parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
    parser.add_argument('--patch_size', type=list,  default=[128,128,128], help='patch size of network input')
    FLAGS = parser.parse_args()

    # Use CUDA
    
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda")

    total_metric = Inference(FLAGS, FLAGS.model_path)
