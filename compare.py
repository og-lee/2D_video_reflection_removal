import math
import os
import signal
import time
import numpy as np 
import cv2
import math 
import sys
import apex
import torch
from apex import amp
import pytorch_ssim
# from inference_handlers.inference import infer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from network import vgg

from config import get_cfg
from inference_handlers.infer_utils.util import get_inference_engine
from loss.loss_utils import compute_loss, compute_loss1, twodecoder_loss, only_trans_loss
# Constants
from utils.Argparser import parse_argsV2
from utils.AverageMeter import AverageMeter, AverageMeterDict
from utils.Saver import save_checkpointV2, load_weightsV2
from utils.util import get_lr_schedulers, show_image_summary, get_model, cleanup_env, \
  reduce_tensor, is_main_process, synchronize, get_datasets, get_optimiser, init_torch_distributed, _find_free_port, \
  format_pred


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def main(): 
    gt_dir = os.path.join('/root/workplace/perceptual-reflection-removal/test_images/dataset4_2')
    proposed_dir = os.path.join('/root/workplace/middle_project/results/transreffixed_res_nobn_l1loss_percep_newloader_lowerperloss_deeper_4_han_morechannel_13840/1')    
    prr_dir = os.path.join('/root/workplace/perceptual-reflection-removal/test_results/dataset4_2')
    gt_trans_dir = os.path.join('/root/workplace/middle_project/results/transreffixed_res_nobn_l1loss_percep_newloader_lowerperloss_deeper_4_han_morechannel_13840/1/gt')    

    prr_list =os.listdir(prr_dir)
    proposed_list = os.listdir(proposed_dir)
    gt_list = os.listdir(gt_dir)
    gt_trans_list = os.listdir(gt_trans_dir)

    psnr_gttrans_gt = []
    psnr_gttrans_propsoed = []
    psnr_gttrans_prr = []

    psnr_proposed = []
    psnr_prr = []
    ssim_proposed = []
    ssim_prr = []

    prr_list = [i for i in prr_list if 'asynth' in i]
    proposed_list = [i for i in proposed_list if ('trans' in i and 'half' not in i)]
    gt_list = [i for i in gt_list if 'synth' in i]
    gt_trans_list = [i for i in gt_trans_list if ('trans' in i and 'half' not in i)]

    print(len(gt_list))
    print(len(prr_list))
    for i in range(len(prr_list)): 
        gt = cv2.imread(os.path.join(gt_dir,gt_list[i]))
        proposed = cv2.imread(os.path.join(proposed_dir,proposed_list[i]))
        prr = cv2.imread(os.path.join(prr_dir,prr_list[i]))
        gt_trans = cv2.imread(os.path.join(gt_trans_dir,gt_trans_list[i]))

        psnr_proposed.append(calculate_psnr(gt, proposed))
        psnr_prr.append(calculate_psnr(gt, prr))
        psnr_gttrans_gt.append(calculate_psnr(gt_trans,gt))
        psnr_gttrans_propsoed.append(calculate_psnr(gt_trans,proposed))
        psnr_gttrans_prr.append(calculate_psnr(gt_trans,prr))

        ssim_proposed.append(calculate_ssim(gt_trans, proposed))
        ssim_prr.append(calculate_ssim(gt_trans, prr))

    print('gt synth and proposed output')
    print(np.mean(psnr_proposed))
    print(np.mean(ssim_proposed))
    print()

    print('gt trans and gt synth')
    print(np.mean(psnr_gttrans_gt))
    print('')
    print('gt trans and proposed output')
    print(np.mean(psnr_gttrans_propsoed))
    print('')
    print('gt trans and prr output')
    print(np.mean(psnr_gttrans_prr))


    print('prr')
    print(np.mean(psnr_prr))
    print('ssim gt rans and prr output')
    print(np.mean(ssim_prr))

if __name__ == '__main__': 
    main()