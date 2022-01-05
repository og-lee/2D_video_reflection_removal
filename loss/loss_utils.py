import numpy as np
import torch
from torch.nn import functional as F

from utils.Constants import PRED_LOGITS
from utils.util import iou_fixed_torch


def bootstrapped_ce_loss(raw_ce, n_valid_pixels_per_im=None, fraction=0.25):
  n_valid_pixels_per_im = raw_ce.shape[-1]*raw_ce.shape[-2] if n_valid_pixels_per_im is None else n_valid_pixels_per_im
  ks = torch.max(torch.tensor(n_valid_pixels_per_im * fraction).cuda().int(), torch.tensor(1).cuda().int())
  if len(raw_ce.shape) > 3:
    bootstrapped_loss = raw_ce.reshape(raw_ce.shape[0], raw_ce.shape[1], -1).topk(ks, dim=-1)[0].mean(dim=-1).mean()
  else:
    bootstrapped_loss = raw_ce.reshape(raw_ce.shape[0], -1).topk(ks, dim=-1)[0].mean(dim=-1).mean()
  return bootstrapped_loss


def compute_loss(input_dict, pred_dict, target_dict, cfg):

  """
  :param cfg: configuration file
  :param target_dict: dict of targets({"flow":<>, "mask": <>})
  :param pred_dict: dictionary of predictions
  """

  result = {'total_loss': torch.tensor(0).float().cuda()}
  if 'ce' in cfg.TRAINING.LOSSES.NAME:
    assert pred_dict[PRED_LOGITS] is not None
    assert target_dict['mask'] is not None
    raw_pred = pred_dict[PRED_LOGITS]
    target = target_dict['mask']
    ignore_mask = target_dict['ignore_mask'] if 'ignore_mask' in target_dict else None

    criterion = torch.nn.CrossEntropyLoss(reduce=False) if cfg.TRAINING.LOSSES.MULTI_CLASS else \
      torch.nn.BCEWithLogitsLoss(reduce=False)

    if len(raw_pred.shape) > 4:
      raw_pred = F.interpolate(raw_pred, target.shape[2:], mode="trilinear")
    else:
      raw_pred = F.interpolate(raw_pred, target.shape[2:], mode="bilinear")


    if cfg.TRAINING.LOSSES.MULTI_CLASS:
      target = target.long().squeeze(1)
    else:
      target = (target!=0).float().squeeze(1)
      pred = raw_pred[:, -1]

    loss_image = criterion(pred, target)
    if cfg.TRAINING.LOSSES.USE_IGNORE_MASK and ignore_mask is not None:
      valid_mask = (ignore_mask == 0).int()
      loss_image *= valid_mask.squeeze(1)

    if cfg.TRAINING.LOSSES.BOOTSTRAP:
      loss = bootstrapped_ce_loss(loss_image)
    else:
      loss = loss_image.mean()

    iou = calc_iou(F.softmax(raw_pred, dim=1), target)
    iou = iou_fixed_torch(F.softmax(raw_pred, dim=1), target.float().cuda())

    result['loss_mask'] = loss
    result['total_loss'] += loss
    result['iou'] = iou

  return result


def compute_loss1(input_dict, pred, pred_half, cfg): 
  result = {'total_loss': torch.tensor(0).float().cuda()}
  # input dict has synth, trans, ref images 
  # pred is C D H W 
  trans = input_dict['input_trans']
  ref = input_dict['input_ref']
  trans_half = input_dict['input_trans_half']
  ref_half = input_dict['input_ref_half']

  diff_trans = F.l1_loss(trans[:,:,:,:,:] , pred[:,0:3,:,:,:])
  diff_ref = F.l1_loss(ref[:,:,:,:,:] , pred[:,3:6,:,:,:]) 
  diff_trans_half = F.l1_loss(trans_half[:,:,:,:,:] , pred_half[:,0:3,:,:,:])
  diff_ref_half = F.l1_loss(ref_half[:,:,:,:,:] , pred_half[:,3:6,:,:,:])

  result['trans_loss'] = diff_trans
  result['ref_loss'] = diff_ref
  result['trans_half_loss'] = diff_trans_half
  result['ref_half_loss'] = diff_ref_half

  result['total_loss'] += diff_trans 
  result['total_loss'] += diff_ref
  result['total_loss'] += diff_trans_half
  result['total_loss'] += diff_ref_half

  return result

def only_trans_loss(input_dict, pred, cfg):
  result = {'total_loss': torch.tensor(0).float().cuda()}
  # input dict has synth, trans, ref images 
  # pred is C D H W 
  trans = input_dict['input_trans']
  ref = input_dict['input_ref']

  # pred = torch.squeeze(pred[0])
  # trans = torch.squeeze(trans)
  # ref = torch.squeeze(ref)

  # pred_frames = torch.transpose(pred,1,0)
  # trans_frames = torch.transpose(trans,1,0)
  # ref_frames = torch.transpose(ref,1,0)

  # torch.set_printoptions(edgeitems = 10)

  diff_trans = F.l1_loss(trans , pred) 
  # diff_ref = torch.norm(ref_frames - pred_frames)

  
  # loss = diff_trans + diff_ref
  # result['trans_loss'] = diff_trans
  result['total_loss'] += diff_trans 
  # result['total_loss'] += diff_ref 

  return result



def twodecoder_loss(input_dict, pred, cfg): 
  result = {'total_loss': torch.tensor(0).float().cuda()}
  # input dict has synth, trans, ref images 
  # pred is C D H W 
  trans = input_dict['input_trans']
  ref = input_dict['input_ref']
  synth = input_dict['input_synth']

  pred_trans = torch.squeeze(pred[0])
  pred_ref = torch.squeeze(pred[1])
  trans = torch.squeeze(trans)
  ref = torch.squeeze(ref)


  pred_frames_trans = torch.transpose(pred_trans,1,0)
  pred_frames_ref = torch.transpose(pred_ref,1,0)
  trans_frames = torch.transpose(trans,1,0)
  ref_frames = torch.transpose(ref,1,0)

  # print(pred_frames[0:4,0:3,:,:].shape)
  # print(trans_frames.shape)
  # print(ref_frames.shape)
  
  # diff_trans = torch.norm(trans_frames[0:4,:,:,:] - pred_frames[0:4,0:3,:,:]) / 4
  # diff_ref = torch.norm(ref_frames[0:4,:,:,:] - pred_frames[0:4,3:6,:,:]) / 4

  diff_trans = torch.norm(trans_frames - pred_frames_trans) / 4
  diff_ref = torch.norm(ref_frames - pred_frames_ref) / 4

  # diff_ref = torch.div(torch.sum(torch.abs(ref_frames - pred_frames_ref)),4)
  # diff_trans = torch.div(torch.sum(torch.abs(trans_frames - pred_frames_trans)),4)

  # loss = diff_trans + diff_ref
  result['trans_loss'] = diff_trans
  result['ref_loss'] = diff_ref
  result['total_loss'] += diff_trans 
  result['total_loss'] += diff_ref

  return result


  


def calc_iou(pred, gt, info=None):
  pred = torch.argmax(pred, dim=1)
  pred = pred.cpu().data.numpy()
  gt = gt.cpu().data.numpy()
  # if len(gt.shape) > 3:
  #   gt = np.sum(gt, axis=1)

  ious = []
  num_frames = pred.shape[0]
  end = num_frames
  for t in range(0, end):
    ious_per_im = []
    objs_gt = np.unique(gt[t])[np.unique(gt[t]) != 0]
    objs_p = np.unique(pred[t])[np.unique(pred[t]) != 0]
    # false positives in prediction
    fp = np.setdiff1d(objs_p, objs_gt)
    merged_objs = np.append(objs_gt, fp)
    for o in merged_objs:
      p = (pred[t] == o).astype(np.uint8)
      g = (gt[t] == o).astype(np.uint8)
      i = np.logical_and(p > 0, g > 0).sum()
      u = np.logical_or(p > 0, g > 0).sum()
      if u == 0:
          iou = 1.0
      else:
          iou = i / u
      ious_per_im += [iou]
    if len(ious_per_im) > 0:
      ious.append(np.mean(ious_per_im))
    else:
      ious.append(1.0)

  if len(ious) > 0:
    miou = np.mean(ious)
  else:
    miou = 1.0
  return torch.tensor(miou)
