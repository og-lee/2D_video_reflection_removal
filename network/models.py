from functools import reduce

import torch
from torch import nn
from torch.nn import functional as F

from network.NetworkUtil import get_backbone_fn, get_module
from utils import Constants

class BaseNetwork(nn.Module):
  def __init__(self, tw=5):
    super(BaseNetwork, self).__init__()
    self.tw = tw

class Encoder3d1(nn.Module):
  def __init__(self, backbone, tw, pixel_mean, pixel_std):
    super(Encoder3d1, self).__init__()
    self.conv1_p = nn.Conv3d(1, 64, kernel_size=7, stride=(1, 2, 2),
                             padding=(3, 3, 3), bias=False)

    resnet = get_backbone_fn(backbone.NAME)(sample_size=112, sample_duration = tw)
    if backbone.PRETRAINED_WTS:
      print('Loading pretrained weights for the backbone from {} {}{}...'.format(Constants.font.BOLD,
                                                                                 backbone.PRETRAINED_WTS, Constants.font.END))
      chkpt = torch.load(backbone.PRETRAINED_WTS)
      resnet.load_state_dict(chkpt)

    self.resnet = resnet
    
    self.resnet = resnet
    self.conv1 = resnet.conv1
    self.bn1 = resnet.bn1
    self.relu = resnet.relu  # 1/2, 64
    self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

    self.layer1 = resnet.layer1  # 1/4, 256
    self.layer2 = resnet.layer2  # 1/8, 512
    self.layer3 = resnet.layer3  # 1/16, 1024
    self.layer4 = resnet.layer4  # 1/32, 2048

    self.register_buffer('mean', torch.FloatTensor(pixel_mean).view(1, 3, 1, 1, 1))
    self.register_buffer('std', torch.FloatTensor(pixel_std).view(1, 3, 1, 1, 1))

    if backbone.FREEZE_BN:
      self.freeze_batchnorm()

  def freeze_batchnorm(self):
    print("Freezing batchnorm for Encoder3d")
    # freeze BNs
    for m in self.modules():
      if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
        for p in m.parameters():
          p.requires_grad = False

  def forward(self, in_f, in_p=None):
    assert in_f is not None or in_p is not None
    f = (in_f * 255.0 - self.mean) / self.std
    f /= 255.0

    if in_f is None:
      p = in_p
      if len(in_p.shape) < 4:
        p = torch.unsqueeze(in_p, dim=1)  # add channel dim

      x = self.conv1_p(p)
    elif in_p is not None:
      p = in_p
      if len(in_p.shape) < 4:
        p = torch.unsqueeze(in_p, dim=1)  # add channel dim

      x = self.conv1(f) + self.conv1_p(p)  # + self.conv1_n(n)
    else:
      x = self.conv1(f)
    x = self.bn1(x)
    c1 = self.relu(x)  # 1/2, 64
    x = self.maxpool(c1)  # 1/4, 64
    r2 = self.layer1(x)  # 1/4, 64
    r3 = self.layer2(r2)  # 1/8, 128
    r4 = self.layer3(r3)  # 1/16, 256
    r5 = self.layer4(r4)  # 1/32, 512


    return r5, r4, r3, r2 


class Encoder3d(nn.Module):
  def __init__(self, backbone, tw, pixel_mean, pixel_std):
    super(Encoder3d, self).__init__()
    self.conv1_p = nn.Conv3d(1, 64, kernel_size=7, stride=(1, 2, 2),
                             padding=(3, 3, 3), bias=False)

    resnet = get_backbone_fn(backbone.NAME)(sample_size=112, sample_duration = tw)
    if backbone.PRETRAINED_WTS:
      print('Loading pretrained weights for the backbone from {} {}{}...'.format(Constants.font.BOLD,
                                                                                 backbone.PRETRAINED_WTS, Constants.font.END))
      chkpt = torch.load(backbone.PRETRAINED_WTS)
      resnet.load_state_dict(chkpt)

    self.resnet = resnet
    
    self.resnet = resnet
    self.conv1 = resnet.conv1
    self.bn1 = resnet.bn1
    self.relu = resnet.relu  # 1/2, 64
    self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

    self.layer1 = resnet.layer1  # 1/4, 256
    self.layer2 = resnet.layer2  # 1/8, 512
    self.layer3 = resnet.layer3  # 1/16, 1024
    self.layer4 = resnet.layer4  # 1/32, 2048

    self.register_buffer('mean', torch.FloatTensor(pixel_mean).view(1, 3, 1, 1, 1))
    self.register_buffer('std', torch.FloatTensor(pixel_std).view(1, 3, 1, 1, 1))

    if backbone.FREEZE_BN:
      self.freeze_batchnorm()

  def freeze_batchnorm(self):
    print("Freezing batchnorm for Encoder3d")
    # freeze BNs
    for m in self.modules():
      if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
        for p in m.parameters():
          p.requires_grad = False

  def forward(self, in_f, in_p=None):
    assert in_f is not None or in_p is not None
    f = (in_f * 255.0 - self.mean) / self.std
    f /= 255.0

    if in_f is None:
      p = in_p
      if len(in_p.shape) < 4:
        p = torch.unsqueeze(in_p, dim=1)  # add channel dim

      x = self.conv1_p(p)
    elif in_p is not None:
      p = in_p
      if len(in_p.shape) < 4:
        p = torch.unsqueeze(in_p, dim=1)  # add channel dim

      x = self.conv1(f) + self.conv1_p(p)  # + self.conv1_n(n)
    else:
      x = self.conv1(f)
    x = self.bn1(x)
    c1 = self.relu(x)  # 1/2, 64
    x = self.maxpool(c1)  # 1/4, 64
    r2 = self.layer1(x)  # 1/4, 64
    r3 = self.layer2(r2)  # 1/8, 128
    r4 = self.layer3(r3)  # 1/16, 256
    r5 = self.layer4(r4)  # 1/32, 512


    return r5, r4, r3, r2, c1 
  

class Decoder3d(nn.Module):
  def __init__(self, n_classes, inter_block, refine_block, pred_scale_factor=(1,4,4)):
    super(Decoder3d, self).__init__()
    mdim = 256
    self.pred_scale_factor = pred_scale_factor
    self.GC = get_module(inter_block)(2048, mdim)
    self.convG1 = nn.Conv3d(mdim, mdim, kernel_size=3, padding=1)
    self.convG2 = nn.Conv3d(mdim, mdim, kernel_size=3, padding=1)
    refine_cls = get_module(refine_block)
    self.RF4 = refine_cls(1024, mdim)  # 1/16 -> 1/8
    self.RF3 = refine_cls(512, mdim)  # 1/8 -> 1/4
    self.RF2 = refine_cls(256, mdim)  # 1/4 -> 1

    self.pred5 = nn.Conv3d(mdim, n_classes, kernel_size=3, padding=1, stride=1)
    self.pred4 = nn.Conv3d(mdim, n_classes, kernel_size=3, padding=1, stride=1)
    self.pred3 = nn.Conv3d(mdim, n_classes, kernel_size=3, padding=1, stride=1)
    self.pred2 = nn.Conv3d(mdim, n_classes, kernel_size=3, padding=1, stride=1)

  def forward(self, r5, r4, r3, r2, support):
    # there is a merge step in the temporal net. This split is a hack to fool it
    # x = torch.cat((x, r5), dim=1)
    x = self.GC(r5)
    r = self.convG1(F.relu(x))
    r = self.convG2(F.relu(r))
    m5 = x + r  # out: 1/32, 64
    m4 = self.RF4(r4, m5)  # out: 1/16, 64
    m3 = self.RF3(r3, m4)  # out: 1/8, 64
    m2 = self.RF2(r2, m3)  # out: 1/4, 64

    p2 = self.pred2(F.relu(m2))
    p3 = self.pred3(F.relu(m3))
    p4 = self.pred4(F.relu(m4))
    p5 = self.pred5(F.relu(m5))

    p = F.interpolate(p2, scale_factor=self.pred_scale_factor, mode='trilinear')
    return p

class Decoder3d_refine(nn.Module):
  def __init__(self, n_classes, inter_block, refine_block, pred_scale_factor=(1,4,4)):
    super(Decoder3d_refine, self).__init__()
    mdim = 256
    self.pred_scale_factor = pred_scale_factor
    self.GC = get_module(inter_block)(2048, mdim)
    self.convG1 = nn.Conv3d(mdim, mdim, kernel_size=3, padding=1)
    self.convG2 = nn.Conv3d(mdim, mdim, kernel_size=3, padding=1)
    refine_cls = get_module(refine_block)
    self.RF4 = refine_cls(1024, mdim)  # 1/16 -> 1/8
    self.RF3 = refine_cls(512, mdim)  # 1/8 -> 1/4
    self.RF2 = refine_cls(256, mdim)  # 1/4 -> 1

    self.beforeinter = nn.Conv3d(mdim, mdim,kernel_size=3, padding=1, stride=1)
    self.pred5 = nn.Conv3d(mdim, n_classes, kernel_size=3, padding=1, stride=1)
    self.pred4 = nn.Conv3d(mdim, n_classes, kernel_size=3, padding=1, stride=1)
    self.pred3 = nn.Conv3d(mdim, n_classes, kernel_size=3, padding=1, stride=1)
    self.pred2 = nn.Conv3d(mdim, n_classes, kernel_size=3, padding=1, stride=1)

  def forward(self, r5, r4, r3, r2, support):
    # there is a merge step in the temporal net. This split is a hack to fool it
    # x = torch.cat((x, r5), dim=1)
    x = self.GC(r5)
    r = self.convG1(F.relu(x))
    r = self.convG2(F.relu(r))
    m5 = x + r  # out: 1/32, 64
    m4 = self.RF4(r4, m5)  # out: 1/16, 64
    m3 = self.RF3(r3, m4)  # out: 1/8, 64
    m2 = self.RF2(r2, m3)  # out: 1/4, 64

    
    p = F.interpolate(m2, scale_factor=self.pred_scale_factor, mode='trilinear')
    before = self.beforeinter(p)
    p2 = self.pred2(F.tanh(before))
    p3 = self.pred3(F.relu(m3))
    p4 = self.pred4(F.relu(m4))
    p5 = self.pred5(F.relu(m5))

    # p = F.interpolate(p2, scale_factor=self.pred_scale_factor, mode='trilinear')
    return p2
 


class UnetBlock(nn.Module): 
  def __init__(self, inchannel, outchannel, pools = (2,2,2) , poolk = (3,3,3), poolp = (1,1,1)):
    super(UnetBlock, self).__init__()
    self.conv1 = nn.Conv3d(inchannel, outchannel, kernel_size=3,padding = 1)
    # self.bn1 = nn.BatchNorm3d(outchannel)
    self.conv2 = nn.Conv3d(outchannel, outchannel, kernel_size=3,padding = 1)
    # self.bn2 = nn.BatchNorm3d(outchannel)
    self.maxpool = nn.MaxPool3d(kernel_size= poolk, stride= pools, padding = poolp)

  def forward(self, x): 
    x = self.conv1(x)
    # x = self.bn1(x)
    x = self.conv2(F.relu(x))
    # x = self.bn2(x)
    x1 = F.relu(x)
    x2 = self.maxpool(x1)
    return x1, x2

class UnetEncoder3d(nn.Module): 
  def __init__(self, pixel_mean, pixel_std):
    super(UnetEncoder3d, self).__init__()
    # self.conv1_p = nn.Conv3d(1, 64, kernel_size=7, stride=(1, 2, 2),
                            #  padding=(3, 3, 3), bias=False)
    # self.block1 = UnetBlock(3,64,poolk=(1,3,3), pools=(1,2,2), poolp=(0,1,1))
    self.block1 = UnetBlock(3,64)
    self.block2 = UnetBlock(64,128)
    self.block3 = UnetBlock(128,256)
    self.block4 = UnetBlock(256,512)
    
    self.register_buffer('mean', torch.FloatTensor(pixel_mean).view(1, 3, 1, 1, 1))
    self.register_buffer('std', torch.FloatTensor(pixel_std).view(1, 3, 1, 1, 1))

  def freeze_batchnorm(self):
    print("Freezing batchnorm for Encoder3d")
    # freeze BNs
    for m in self.modules():
      if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
        for p in m.parameters():
          p.requires_grad = False

  def forward(self, in_f, in_p=None):
    assert in_f is not None or in_p is not None
    f = (in_f * 255.0 - self.mean) / self.std
    f /= 255.0

    s1, s11 = self.block1(in_f)
    s2, s22 = self.block2(s11)
    s3, s33 = self.block3(s22)
    s4, s44 = self.block4(s33)

    return s4, s3, s2, s1 

class UnetResBlock(nn.Module): 
  def __init__(self, inchannel, outchannel, pools = (2,2,2) , poolk = (3,3,3), poolp = (1,1,1)):
    super(UnetResBlock, self).__init__()
    self.conv1 = nn.Conv3d(inchannel, inchannel, kernel_size=3,padding = 1)
    self.bn1 = nn.BatchNorm3d(inchannel)
    self.conv2 = nn.Conv3d(inchannel, inchannel, kernel_size=3,padding = 1)
    self.bn2 = nn.BatchNorm3d(inchannel)
    self.conv3 = nn.Conv3d(inchannel, outchannel, kernel_size=3,padding = 1,stride = 2)
    self.bn3 = nn.BatchNorm3d(outchannel)

  def forward(self, x): 
    residual = x 
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.conv2(F.relu(x))
    x = self.bn2(x)
    x = residual + x
    # for skip
    x1 = F.relu(x)

    x2 = self.conv3(x1)
    x2 = self.bn3(x2)
    x2 = F.relu(x2)

    return x1, x2

class UnetResEncoder3d(nn.Module):
  def __init__(self, pixel_mean, pixel_std):
    super(UnetResEncoder3d, self).__init__()
    # self.conv1_p = nn.Conv3d(1, 64, kernel_size=7, stride=(1, 2, 2),
                            #  padding=(3, 3, 3), bias=False)
    # self.block1 = UnetBlock(3,64,poolk=(1,3,3), pools=(1,2,2), poolp=(0,1,1))
    self.conv1 = nn.Conv3d(3, 64, kernel_size=3,padding = 1)
    self.bn1 = nn.BatchNorm3d(64)
    self.block1 = UnetResBlock(64,64)
    self.block2 = UnetResBlock(64,128)
    self.block3 = UnetResBlock(128,256)
    self.block4 = UnetResBlock(256,512)
    
    self.register_buffer('mean', torch.FloatTensor(pixel_mean).view(1, 3, 1, 1, 1))
    self.register_buffer('std', torch.FloatTensor(pixel_std).view(1, 3, 1, 1, 1))

  def forward(self, in_f, in_p=None):
    assert in_f is not None or in_p is not None
    f = (in_f * 255.0 - self.mean) / self.std
    f /= 255.0

    x = self.conv1(in_f)
    x = self.bn1(x)
    x = F.relu(x)
    s1, s11 = self.block1(x)
    s2, s22 = self.block2(s11)
    s3, s33 = self.block3(s22)
    s4, s44 = self.block4(s33)
    # print(s1.shape)
    # print(s2.shape)
    # print(s3.shape)
    # print(s4.shape)
    # print(s44.shape)

    return s44, s4, s3, s2, s1 

class UnetDecoderBlock(nn.Module): 
  def __init__(self, inchannel, outchannel, ksize = (2,2,2), ssize = (2,2,2)):
    super(UnetDecoderBlock, self).__init__()
    # self.upconv1 = nn.ConvTranspose3d(inchannel, outchannel, 2, stride=2)
    self.upconv1 = nn.ConvTranspose3d(inchannel, outchannel, ksize, ssize)
    self.bn1 = nn.BatchNorm3d(outchannel)
    # concat
    self.conv1 =  nn.Conv3d(outchannel*2,outchannel, kernel_size=3, padding = 1)
    self.bn2 = nn.BatchNorm3d(outchannel)
    self.conv2 = nn.Conv3d(outchannel,outchannel, kernel_size=3, padding = 1)
    self.bn3 = nn.BatchNorm3d(outchannel)

  def forward(self, x, skip): 
    x = self.upconv1(x)
    x = self.bn1(x)
    x = F.relu(x)
    x = torch.cat((x, skip), dim = 1)
    x = self.conv1(x)
    x = self.bn2(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = self.bn3(x)
    x = F.relu(x)
    return x

class DecoderBlockInter(nn.Module):
  def __init__(self, inchannel, outchannel, ksize = (2,2,2)):
    super(DecoderBlockInter, self).__init__()
    self.ksize = ksize 
    self.conv11 = nn.Conv3d(inchannel, outchannel, kernel_size=3, padding =1)
    self.bn1 = nn.BatchNorm3d(outchannel)
    # concat
    self.conv1 =  nn.Conv3d(outchannel*2,outchannel, kernel_size=3, padding = 1)
    self.bn2 = nn.BatchNorm3d(outchannel)
    self.conv2 = nn.Conv3d(outchannel,outchannel, kernel_size=3, padding = 1)
    self.bn3 = nn.BatchNorm3d(outchannel)

  def forward(self, x, skip): 
    # print(x.shape)
    x = self.conv11(x) 
    x = self.bn1(x)
    x = F.interpolate(F.relu(x), scale_factor=self.ksize, mode='trilinear')
    # print(x.shape)
    # print(skip.shape)
    # print()
    # x = self.upconv1(x)
    # x = self.bn1(x)
    # x = F.relu(x)
    x = torch.cat((x, skip), dim = 1)
    x = self.conv1(x)
    x = self.bn2(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = self.bn3(x)
    x = F.relu(x)
    return x

class UnetDecoder3d(nn.Module): 
  def __init__(self, n_classes):
    super(UnetDecoder3d, self).__init__()
    self.block1 = UnetDecoderBlock(512,256,ksize=(1,2,2),ssize=(1,2,2))
    self.block2 = UnetDecoderBlock(256,128)
    self.block3 = UnetDecoderBlock(128,64)
    self.block4 = UnetDecoderBlock(64,64)
    self.conv1 = nn.Conv3d(64,64, kernel_size=3, padding = 1)
    self.bn1 = nn.BatchNorm3d(64)
    self.conv2 = nn.Conv3d(64,n_classes, kernel_size=3, padding = 1)
    self.bn2 = nn.BatchNorm3d(n_classes)

  def forward(self, r55,r5, r4, r3, r2, support):
    x = self.block1(r55, r5)
    x = self.block2(x, r4)
    x = self.block3(x, r3)
    x = self.block4(x, r2)
    x = self.conv1(x)
    # x = self.bn1(x)
    x = self.conv2(F.relu(x))
    # x = self.bn2(x)
    # x = F.relu(x)
    x = torch.tanh(x)

    return x


class ResnetDecoder3d(nn.Module): 
  def __init__(self, n_classes):
    super(ResnetDecoder3d, self).__init__()
    self.block1 = DecoderBlockInter(512,256)
    self.block2 = DecoderBlockInter(256,128)
    self.block3 = DecoderBlockInter(128,64)
    self.block4 = DecoderBlockInter(64,64,ksize=(1,2,2))

    # self.upconv1 = self.upconv1 = nn.ConvTranspose3d(64, 64, (1,2,2), (1,2,2))
    self.bn1 = nn.BatchNorm3d(64)
    self.bn2 = nn.BatchNorm3d(n_classes)
    self.conv1 = nn.Conv3d(64,64, kernel_size=3, padding = 1)
    self.conv2 = nn.Conv3d(64,n_classes, kernel_size=3, padding = 1)


  def forward(self, r5, r4, r3, r2, r1, support):
    x = self.block1(r5, r4)
    x = self.block2(x, r3)
    x = self.block3(x, r2)
    x = self.block4(x, r1)
    x = F.interpolate(x, scale_factor=(1,2,2), mode='trilinear')
    # x = self.upconv1(x)
    # x = self.bn1(x)
    # x = self.conv1(F.relu(x))
    x = self.conv1(x)
    # x = self.bn1(x)
    x = self.conv2(F.relu(x))
    # x = self.bn2(x)
    # x = F.sigmoid(x)
    x = torch.tanh(x)

    return x



class SaliencyNetwork(BaseNetwork):
  def __init__(self, cfg):
    super(SaliencyNetwork, self).__init__()
    self.encoder = Encoder3d1(cfg.MODEL.BACKBONE, cfg.INPUT.TW, cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD)
    decoders = [Decoder3d(cfg.MODEL.N_CLASSES, inter_block=cfg.MODEL.DECODER.INTER_BLOCK,
                             refine_block=cfg.MODEL.DECODER.REFINE_BLOCK)]
    self.decoders = nn.ModuleList()
    for decoder in decoders:
      self.decoders.append(decoder)
    if cfg.MODEL.FREEZE_BN:
      self.encoder.freeze_batchnorm()

  def forward(self, x, ref=None):
    r5, r4, r3, r2 = self.encoder.forward(x, ref)
    flatten = lambda lst: [lst] if type(lst) is torch.Tensor else reduce(torch.add, [flatten(ele) for ele in lst])
    p = flatten([decoder.forward(r5, r4, r3, r2, None) for decoder in self.decoders])
    # e = self.decoder_embedding.forward(r5, r4, r3, r2, None)
    return p
    # p = self.decoder.forward(r5, r4, r3, r2, None)
    # return [p]

class SaliencyNetwork1(BaseNetwork):
  def __init__(self, cfg):
    super(SaliencyNetwork1, self).__init__()
    self.encoder = Encoder3d1(cfg.MODEL.BACKBONE, cfg.INPUT.TW, cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD)
    decoders = [Decoder3d_refine(cfg.MODEL.N_CLASSES, inter_block=cfg.MODEL.DECODER.INTER_BLOCK,
                             refine_block=cfg.MODEL.DECODER.REFINE_BLOCK)]
    self.decoders = nn.ModuleList()
    for decoder in decoders:
      self.decoders.append(decoder)
    if cfg.MODEL.FREEZE_BN:
      self.encoder.freeze_batchnorm()

  def forward(self, x, ref=None):
    r5, r4, r3, r2 = self.encoder.forward(x, ref)
    flatten = lambda lst: [lst] if type(lst) is torch.Tensor else reduce(torch.add, [flatten(ele) for ele in lst])
    p = flatten([decoder.forward(r5, r4, r3, r2, None) for decoder in self.decoders])
    # e = self.decoder_embedding.forward(r5, r4, r3, r2, None)
    return p
    # p = self.decoder.forward(r5, r4, r3, r2, None)
    # return [p]

class ResnetEncoderDecoder(BaseNetwork):
  def __init__(self, cfg):
    super(ResnetEncoderDecoder, self).__init__()
    self.encoder = Encoder3d(cfg.MODEL.BACKBONE, cfg.INPUT.TW, cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD)
    self.decoder = ResnetDecoder3d(cfg.MODEL.N_CLASSES)

    # if cfg.MODEL.FREEZE_BN:
    #   self.encoder.freeze_batchnorm()

  def forward(self, x, ref=None):
    r5, r4, r3, r2, r1 = self.encoder.forward(x, ref)
    p = self.decoder(r5,r4,r3,r2,r1,None)

    return p

class ReflectionNetwork(BaseNetwork):
  def __init__(self, cfg):
    super(ReflectionNetwork, self).__init__()
    self.encoder = Encoder3d(cfg.MODEL.BACKBONE, cfg.INPUT.TW, cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD)
    self.decoder_trans = Decoder3d(cfg.MODEL.N_CLASSES, inter_block=cfg.MODEL.DECODER.INTER_BLOCK,
                             refine_block=cfg.MODEL.DECODER.REFINE_BLOCK)
    self.decoder_ref =   Decoder3d(cfg.MODEL.N_CLASSES, inter_block=cfg.MODEL.DECODER.INTER_BLOCK,
                             refine_block=cfg.MODEL.DECODER.REFINE_BLOCK)
    # self.decoders = nn.ModuleList()
    # for decoder in decoders:
      # self.decoders.append(decoder)
    if cfg.MODEL.FREEZE_BN:
      self.encoder.freeze_batchnorm()

  def forward(self, x, ref=None):
    r5, r4, r3, r2 = self.encoder.forward(x, ref)
    # flatten = lambda lst: [lst] if type(lst) is torch.Tensor else reduce(torch.add, [flatten(ele) for ele in lst])
    # p = flatten([decoder.forward(r5, r4, r3, r2, None) for decoder in self.decoders])
    # e = self.decoder_embedding.forward(r5, r4, r3, r2, None)
    # return p
    trans = self.decoder_trans.forward(r5, r4, r3, r2, None)
    refl = self.decoder_ref.forward(r5, r4, r3, r2, None)
    p = []
    p.append(trans)
    p.append(refl)
    return p

class ReflectionTrans(BaseNetwork): 
  def __init__(self, cfg):
    super(ReflectionTrans, self).__init__()
    self.encoder = UnetEncoder3d(cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD)
    self.decoder_trans = UnetDecoder3d(cfg.MODEL.N_CLASSES)

    if cfg.MODEL.FREEZE_BN:
      self.encoder.freeze_batchnorm()

  def forward(self, x, ref=None):
    r5, r4, r3, r2 = self.encoder.forward(x, ref)

    trans = self.decoder_trans.forward(r5, r4, r3, r2, None)
    p = []
    p.append(trans)
    return p

class ReflectionTransRes(BaseNetwork):
  def __init__(self, cfg):
    super(ReflectionTransRes, self).__init__()
    self.encoder = UnetResEncoder3d(cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD)
    self.decoder_trans = UnetDecoder3d(cfg.MODEL.N_CLASSES)

  def forward(self, x, ref=None):
    r55, r5, r4, r3, r2 = self.encoder.forward(x, ref)

    trans = self.decoder_trans.forward(r55, r5, r4, r3, r2, None)
    p = trans 
    return p



# class MeanShift(nn.Conv2d):
#     def __init__(self, data_mean, data_std, data_range=1, norm=True):
#         """norm (bool): normalize/denormalize the stats"""
#         c = len(data_mean)
#         super(MeanShift, self).__init__(c, c, kernel_size=1)
#         std = torch.Tensor(data_std)
#         self.weight.data = torch.eye(c).view(c, c, 1, 1)
#         if norm:
#             self.weight.data.div_(std.view(c, 1, 1, 1))
#             self.bias.data = -1 * data_range * torch.Tensor(data_mean)
#             self.bias.data.div_(std)
#         else:
#             self.weight.data.mul_(std.view(c, 1, 1, 1))
#             self.bias.data = data_range * torch.Tensor(data_mean)
#         self.requires_grad = False

# class VGGLoss1(nn.Module):
#   def __init__(self, device, vgg=None, weights=None, indices=None, normalize=True):
#     super(VGGLoss1, self).__init__()
#     if vgg is None:
#       self.vgg = Vgg19().cuda()
#     else:
#       self.vgg = vgg
#     self.criterion = nn.L1Loss()
#     self.weights = weights or [1.0 / 2.6, 1.0 / 4.8]
#     self.indices = indices or [2, 7]
#     self.device = device
#     if normalize:
#       self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).to(self.device)
#     else:
#       self.normalize = None
#     print("Vgg: Weights: ", self.weights, " indices: ", self.indices, " normalize: ", self.normalize)

#   def __call__(self, x, y):
#     if self.normalize is not None:
#       x = self.normalize(x)
#       y = self.normalize(y)
#     x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
#     loss = 0
#     for i in range(len(x_vgg)):
#       loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

#     return loss