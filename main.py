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

NUM_EPOCHS = 400
TRAIN_KITTI = False
MASK_CHANGE_THRESHOLD = 1000

BBOX_CROP = True
BEST_IOU = 0
torch.backends.cudnn.benchmark = True


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))


class Trainer:
  def __init__(self, args, port):
    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    self.cfg = cfg
    self.port = port
    assert os.path.exists('saved_models'), "Create a path to save the trained models: <default: ./saved_models> "
    self.model_dir = os.path.join('saved_models', cfg.NAME)
    self.writer = SummaryWriter(log_dir=os.path.join(self.model_dir, "summary"))
    self.iteration = 0
    print("Arguments used: {}".format(args), flush=True)

    self.trainset, self.testset = get_datasets(cfg)
    self.model = get_model(cfg)
    print("Using model: {}".format(self.model.__class__), flush=True)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
      self.model, self.optimiser = self.init_distributed(cfg)
    # TODO: do not use distributed package in this case
    elif torch.cuda.is_available():
      self.model, self.optimiser = self.init_distributed(cfg)
    else:
      raise RuntimeError("CUDA not available.")

    # self.model, self.optimiser, self.start_epoch, start_iter = \
    #   load_weightsV2(self.model, self.optimiser, args.wts, self.model_dir)
    self.lr_schedulers = get_lr_schedulers(self.optimiser, cfg, self.start_epoch)
    self.batch_size = self.cfg.TRAINING.BATCH_SIZE

    args.world_size = 1
    print(args)
    self.args = args
    self.epoch = 0
    self.best_loss_train = math.inf
    self.losses = AverageMeterDict()
    self.ious = AverageMeterDict()

    num_samples = None if cfg.DATALOADER.NUM_SAMPLES == -1 else cfg.DATALOADER.NUM_SAMPLES
    if torch.cuda.device_count() > 1:
      # shuffle parameter does not seem to shuffle the data for distributed sampler
      self.train_sampler = torch.utils.data.distributed.DistributedSampler(
        torch.utils.data.RandomSampler(self.trainset, replacement=True, num_samples=num_samples),
        shuffle=True)
    else:
      self.train_sampler = torch.utils.data.RandomSampler(self.trainset, replacement=True, num_samples=num_samples) \
        if num_samples is not None else None
    shuffle = True if self.train_sampler is None else False
    self.trainloader = DataLoader(self.trainset, batch_size=self.batch_size, num_workers=cfg.DATALOADER.NUM_WORKERS,
                                  shuffle=shuffle, sampler=self.train_sampler)
    self.vgg = vgg.Vgg19(requires_grad=False).to(torch.device('cuda'))
    # print(summary(self.model, tuple((3, cfg.INPUT.TW, 256, 256)), batch_size=1))
    # print(summary(self.model, tuple((3, cfg.INPUT.TW, 480, 854)), batch_size=1))
    # params = []
    # for key, value in dict(self.model.named_parameters()).items():
    #   if value.requires_grad:
    #     params += [{'params': [value], 'lr': args.lr, 'weight_decay': 4e-5}]

  def init_distributed(self, cfg):
    torch.cuda.set_device(args.local_rank)
    init_torch_distributed(self.port)
    model = apex.parallel.convert_syncbn_model(self.model)
    model.cuda()
    optimiser = get_optimiser(model, cfg)
    model, optimiser, self.start_epoch, self.iteration = \
      load_weightsV2(model, optimiser, args.wts, self.model_dir)
    # model, optimizer, start_epoch, best_iou_train, best_iou_eval, best_loss_train, best_loss_eval, amp_weights = \
    #   load_weights(model, self.optimiser, args, self.model_dir, scheduler=None, amp=amp)  # params
    # lr_schedulers = get_lr_schedulers(optimizer, args, start_epoch)
    opt_levels = {'fp32': 'O0', 'fp16': 'O2', 'mixed': 'O1'}
    if cfg.TRAINING.PRECISION in opt_levels:
      opt_level = opt_levels[cfg.TRAINING.PRECISION]
    else:
      opt_level = opt_levels['fp32']
      print('WARN: Precision string is not understood. Falling back to fp32')
    model, optimiser = amp.initialize(model, optimiser, opt_level=opt_level)
    # amp.load_state_dict(amp_weights)
    if torch.cuda.device_count() > 1:
      model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
    self.world_size = torch.distributed.get_world_size()
    print("Intitialised distributed with world size {} and rank {}".format(self.world_size, args.local_rank))
    return model, optimiser

  def train(self):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # switch to train mode
    self.model.train()
    self.ious.reset()
    self.losses.reset()

    end = time.time()
    for i, input_dict in enumerate(self.trainloader):
      # input = input_dict["images"]
      # inputs are synth images and should make transmission and reflection images 
      input_synth = input_dict["images_synth"]
      input_ref = input_dict["images_ref"]
      input_trans = input_dict["images_trans"]
      
      # print(input_dict['info'])
      # target_dict = dict([(k, t.float().cuda()) for k, t in input_dict['target'].items()])
      if 'masks_guidance' in input_dict:
        masks_guidance = input_dict["masks_guidance"]
        masks_guidance = masks_guidance.float().cuda()
      else:
        masks_guidance = None
      info = input_dict["info"]
      data_time.update(time.time() - end)
      input_var = input_synth.float().cuda()
      # compute output
      pred = self.model(input_var, masks_guidance)
      if isinstance(pred, list): 
            pred = pred[0]
      # pred = format_pred(pred)

      input_ref_var = input_ref.float().cuda() 
      input_trans_var = input_trans.float().cuda()
      in_dict = {"input_synth": input_var, "input_trans": input_trans_var, "input_ref":input_ref_var,"guidance": masks_guidance}

      # loss_dict = compute_loss(in_dict, pred, target_dict, self.cfg)
      loss_dict = compute_loss1(in_dict, pred, self.cfg)
      # loss_dict = only_trans_loss(in_dict, pred, self.cfg)

      intrans = torch.reshape(torch.transpose(input_trans_var,1,2),(-1,input_trans_var.shape[1],input_trans_var.shape[3],input_trans_var.shape[4])) 
      pretrans = torch.reshape(torch.transpose(pred,1,2),(-1,pred.shape[1],pred.shape[3],pred.shape[4]))
      pretrans = pretrans[:,0:3,:,:]
      perloss = self.criterionVgg(intrans, pretrans)
      loss_dict['perloss'] = perloss
      loss_dict['total_loss'] += perloss
      total_loss = loss_dict['total_loss']

      # compute gradient and do SGD step
      self.optimiser.zero_grad()
      with amp.scale_loss(total_loss, self.optimiser) as scaled_loss:
        scaled_loss.backward()
      self.optimiser.step()
      self.iteration += 1

      # Average loss and accuracy across processes for logging
      if torch.cuda.device_count() > 1:
        reduced_loss = dict(
          [(key, reduce_tensor(val, self.world_size).data.item()) for key, val in loss_dict.items()])
      else:
        reduced_loss = dict([(key, val.data.item()) for key, val in loss_dict.items()])

      self.losses.update(reduced_loss)

      for k, v in self.losses.val.items():
        self.writer.add_scalar("loss_{}".format(k), v, self.iteration)
      if args.show_image_summary:
        # show_image_summary(self.iteration, self.writer, in_dict, target_dict,
                          #  pred)
        show_image_summary(self.iteration, self.writer, in_dict,
                           pred)


      torch.cuda.synchronize()
      batch_time.update((time.time() - end) / args.print_freq)
      end = time.time()

      loss_str = ' '.join(["{}:{:4f}({:4f})".format(k, self.losses.val[k], self.losses.avg[k])
                           for k, v in self.losses.val.items()])

      if args.local_rank == 0:
        if self.iteration % 10 == 0: 
          print('[Iter: {0}]Epoch: [{1}][{2}/{3}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'LOSSES - {loss})\t'.format(
            self.iteration, self.epoch, i * self.world_size * self.batch_size,
                                        len(self.trainloader) * self.batch_size * self.world_size,
                                        self.world_size * self.batch_size / batch_time.val,
                                        self.world_size * self.batch_size / batch_time.avg,
            batch_time=batch_time, data_time = data_time, loss=loss_str), flush=True)

        # if self.iteration % 10000 == 0:
        if self.iteration % 1000 == 0:
          if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
          save_name = '{}/{}.pth'.format(self.model_dir, self.iteration)
          save_checkpointV2(self.epoch, self.iteration, self.model, self.optimiser, save_name)

    if args.local_rank == 0:
      print('Finished Train Epoch {} Loss {losses.avg}'.
            format(self.epoch, losses=self.losses), flush=True)

    return self.losses.avg

  def eval(self):
    batch_time = AverageMeter()
    losses = AverageMeterDict()
    count = 0
    # switch to evaluate mode
    self.model.eval()

    end = time.time()
    print("Starting validation for epoch {}".format(self.epoch), flush=True)
    trans_psnr_total = []
    trans_ssim_total = []
    for seq in self.testset.get_video_ids():
      self.testset.set_video_id(seq)
      if torch.cuda.device_count() > 1:
        test_sampler = torch.utils.data.distributed.DistributedSampler(self.testset, shuffle=False)
      else:
        test_sampler = None
      # test_sampler.set_epoch(epoch)
      testloader = DataLoader(self.testset, batch_size=1, num_workers=1, shuffle=False, sampler=test_sampler,
                              pin_memory=True)
      losses_video = AverageMeterDict()
      psnr_trans_arr = []
      ssim_trans_arr = []
      for i, input_dict in enumerate(testloader):
        with torch.no_grad():
          input_synth = input_dict["images_synth"]
          input_ref = input_dict["images_ref"]
          input_trans = input_dict["images_trans"]


          if 'masks_guidance' in input_dict:
            masks_guidance = input_dict["masks_guidance"]
            masks_guidance = masks_guidance.float().cuda()
          else:
            masks_guidance = None
          info = input_dict["info"]

          input_var = input_synth.float().cuda()
          # compute output
          pred = self.model(input_var, masks_guidance)
          # pred = format_pred(pred)

          input_ref_var = input_ref.float().cuda() 
          input_trans_var = input_trans.float().cuda()
          in_dict = {"input_synth": input_var, "input_trans": input_trans_var, "input_ref":input_ref_var,"guidance": masks_guidance}

          #  batch channel frames height width 
          # prediction and gt is 
          if isinstance(pred, list): 
            original_pred = pred[0]
            pred = pred[0]
          else :
            original_pred = pred
            
          pred = torch.transpose(pred,1,2)
          pred = torch.reshape(pred, (-1,pred.shape[2],pred.shape[3],pred.shape[4]))
          pred_trans = pred[:,0:3,:,:]

          only_trans = True
          if only_trans:
            pred_ref = pred_trans
          else: 
            pred_ref = pred[:,3:6,:,:]

          input_trans = torch.transpose(input_trans,1,2)
          input_ref = torch.transpose(input_ref,1,2)
          input_trans = torch.reshape(input_trans, (-1,input_trans.shape[2],input_trans.shape[3],input_trans.shape[4]))
          input_ref = torch.reshape(input_ref, (-1,input_ref.shape[2],input_ref.shape[3],input_ref.shape[4]))

          input_trans = input_trans.float().cuda()
          input_ref = input_ref.float().cuda()

          # psnr 
          mse_trans = torch.mean((255*(pred_trans - input_trans))**2, dim=(1,2,3))
          mse_ref = torch.mean((255*(pred_ref - input_ref))**2, dim=(1,2,3))

          psnr_trans = 20 * torch.log10(255.0 / torch.sqrt(mse_trans))
          psnr_ref = 20 * torch.log10(255.0 / torch.sqrt(mse_ref))
          psnr_trans = torch.mean(psnr_trans) 
          psnr_ref = torch.mean(psnr_ref) 

          # ssim 
          trans_ssim = pytorch_ssim.ssim(input_trans,pred_trans)
          ref_ssim = pytorch_ssim.ssim(input_ref,pred_ref)

          psnr_trans_arr.append(psnr_trans)
          ssim_trans_arr.append(trans_ssim)

          torch.cuda.synchronize()
          batch_time.update((time.time() - end) / args.print_freq)
          end = time.time()

          if args.local_rank == 0:
            loss_str = ' '.join(["{}:{:4f}({:4f})".format(k, losses_video.val[k], losses_video.avg[k])
                                 for k, v in losses_video.val.items()])
            # print('{0}: [{1}/{2}]\t'
            #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #       'trans_psnr - {t})\t' 'trans_ssim - {ts}\t' .format(
            #   info[0]['video'], i * args.world_size, len(testloader) * args.world_size,
            #   batch_time=batch_time, t=psnr_trans,  ts=trans_ssim  ),
            #   flush=True)
      trans_psnr_total.append(torch.mean(torch.tensor(psnr_trans_arr)))
      trans_ssim_total.append(torch.mean(torch.tensor(ssim_trans_arr)))
    print('trans psnr',trans_psnr_total)
    print('trans ssim',trans_ssim_total)
    if args.local_rank == 0:
      loss_str = ' '.join(["{}:{:4f}({:4f})".format(k, losses.val[k], losses.avg[k])
                           for k, v in losses.val.items()])
      print('Finished Test: Loss --> loss {}'.format(loss_str), flush=True)

    return losses.avg

  def start(self):
    if args.task == "train":
      # best_loss = best_loss_train
      # best_iou = best_iou_train
      # if args.freeze_bn:
      #   encoders = [module for module in self.model.modules() if isinstance(module, Encoder)]
      #   for encoder in encoders:
      #     encoder.freeze_batchnorm()
      # self.criterionVgg = VGGLoss1(torch.device('cuda'), vgg=self.vgg, normalize=False)
      self.criterionVgg = vgg.VGGLoss1(torch.device('cuda'), vgg=self.vgg, normalize=True)
      # val_loss = self.eval()
      start_epoch = self.epoch
      for epoch in range(start_epoch, self.cfg.TRAINING.NUM_EPOCHS):
        self.epoch = epoch
        if self.train_sampler is not None:
          self.train_sampler.set_epoch(epoch)
        loss_mean = self.train()
        for lr_scheduler in self.lr_schedulers:
          lr_scheduler.step(epoch)

        if args.local_rank == 0:
          print("Total Loss {}".format(loss_mean))
          if loss_mean['total_loss'] < self.best_loss_train:
            if not os.path.exists(self.model_dir):
              os.makedirs(self.model_dir)
            self.best_loss_train = loss_mean['total_loss'] if loss_mean['total_loss'] < self.best_loss_train else self.best_loss_train
            save_name = '{}/{}.pth'.format(self.model_dir, "model_best_train")
            save_checkpointV2(epoch, self.iteration, self.model, self.optimiser, save_name)

        save_name = '{}/{}.pth'.format(self.model_dir, self.iteration)
        save_checkpointV2(epoch, self.iteration, self.model, self.optimiser, save_name)
        val_loss = self.eval()

    elif args.task == 'eval':
      eval_all = True
      if eval_all:
        list_pths = os.listdir('./saved_models/transreffixed_res_nobn_l1loss_percept/')
        for f in list_pths: 
          if f.endswith('.pth'): 
            args.wts = args.wts.rsplit('/',1)[0] +'/'
            args.wts = args.wts + f
            self.model, self.optimiser, self.start_epoch, self.iteration = \
              load_weightsV2(self.model, self.optimiser, args.wts, self.model_dir)
            self.eval()
      else: 
        self.eval()
    elif args.task == 'infer':
      inference_engine = get_inference_engine(self.cfg)
      inference_engine.infer(self.testset, self.model)
    else:
      raise ValueError("Unknown task {}".format(args.task))

  def backup_session(self, signalNumber, _):
    if is_main_process() and self.args.task == 'train':
      save_name = '{}/{}_{}.pth'.format(self.model_dir, "checkpoint", self.iteration)
      print("Received signal {}. \nSaving model to {}".format(signalNumber, save_name))
      save_checkpointV2(self.epoch, self.iteration, self.model, self.optimiser, save_name)
    synchronize()
    cleanup_env()
    exit(1)


def register_interrupt_signals(trainer):
  signal.signal(signal.SIGHUP, trainer.backup_session)
  signal.signal(signal.SIGINT, trainer.backup_session)
  signal.signal(signal.SIGQUIT, trainer.backup_session)
  signal.signal(signal.SIGILL, trainer.backup_session)
  signal.signal(signal.SIGTRAP, trainer.backup_session)
  signal.signal(signal.SIGABRT, trainer.backup_session)
  signal.signal(signal.SIGBUS, trainer.backup_session)
  signal.signal(signal.SIGALRM, trainer.backup_session)
  signal.signal(signal.SIGTERM, trainer.backup_session)


if __name__ == '__main__':
  args = parse_argsV2()
  port = _find_free_port()
  trainer = Trainer(args, port)
  register_interrupt_signals(trainer)
  trainer.start()
  if args.local_rank == 0:
    trainer.backup_session(signal.SIGQUIT, None)
  synchronize()
  cleanup_env()
