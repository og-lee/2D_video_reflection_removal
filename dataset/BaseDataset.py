import random
from abc import abstractmethod

import cv2
import numpy as np
from PIL import Image
from imageio import imread
from torch.utils.data import Dataset
from dataset_prepare import nonlinear_map, synthesize_video1

import sys 
sys.path.append('/root/workplace/middle_project/')

from utils.Resize import resize, ResizeMode


TARGETS = 'targets'
IMAGES_ = 'images'
IMAGES_REF = 'images_ref'
IMAGES_TRANS = 'images_trans'
IMAGES_SYNTH = 'images_synth'

INFO = 'info'


def list_to_dict(list):
  """
  :param list: input list
  :return: converted dictionary
  """
  result_dict = {str(i): val for i, val in enumerate(list)}
  return result_dict


class BaseDataset(Dataset):
  def __init__(self, root, mode='train', resize_mode=None, resize_shape=None):
    self.resize_mode = ResizeMode(resize_mode)
    self.resize_shape = resize_shape
    self.mode = mode
    self.root = root
    self.samples = []
    self.create_sample_list()

  # Override in case tensors have to be normalised
  def normalise(self, tensors):
    # tensors['images'] = tensors['images'].astype(np.float32) / 255.0
    tensors['images_ref'] = tensors['images_ref'].astype(np.float32) / 255.0
    tensors['images_trans'] = tensors['images_trans'].astype(np.float32) / 255.0
    tensors['images_synth'] = tensors['images_synth'].astype(np.float32) / 255.0
    return tensors

  def is_train(self):
    return self.mode == "train"

  def pad_tensors(self, tensors_resized):
    h, w = tensors_resized["images_synth"].shape[:2]
    new_h = h + 32 - h % 32 if h % 32 > 0 else h
    new_w = w + 32 - w % 32 if w % 32 > 0 else w
    lh, uh = (new_h - h) / 2, (new_h - h) / 2 + (new_h - h) % 2
    lw, uw = (new_w - w) / 2, (new_w - w) / 2 + (new_w - w) % 2
    lh, uh, lw, uw = int(lh), int(uh), int(lw), int(uw)

    padded_tensors = {}
    for key, tensor in tensors_resized.items():
      if tensor.ndim == 2:
        tensor = tensor[..., None]
      assert tensor.ndim == 3
      padded_tensors[key] = np.pad(tensor,
                                   ((lh, uh), (lw, uw), (0, 0)),
                                   mode='constant')
    return padded_tensors

  def read_sample(self, sample):
    # targets = self.read_target(sample)


    if self.mode == 'test':
      ref = map(imread, sample['images_ref'])
      trans = map(imread, sample['images_trans'])
      synth = map(imread, sample['images_synth'])

      images_ref_resized = []
      images_trans_resized = []
      images_synth_resized = []

      for r, t, s in zip(ref, trans, synth):
        data = {"image_ref": r, "image_trans": t, "image_synth": s}
      #   need the fix resize 
        data = resize(data, self.resize_mode, self.resize_shape)
        images_ref_resized += [data['image_ref']]
        images_trans_resized += [data['image_trans']]
        images_synth_resized += [data['image_synth']]

      images_ref = np.stack(images_ref_resized)
      images_trans = np.stack(images_trans_resized)
      images_synth = np.stack(images_synth_resized)

      data = {"images_ref": images_ref, "images_trans": images_trans,"images_synth": images_synth}
    
    elif self.mode == 'train': 
      images_resized = []
      images = self.read_image(sample)
      for i in images: 
        data = {'image':i}
        data = resize(data, self.resize_mode, self.resize_shape)
        images_resized += [data['image']]

      images = np.stack(images_resized)
      data = {'images' : images}

    
    for key, val in sample.items():
      if key in ['images','images_ref','images_trans','images_synth']:
        continue
      if key in data:
        data[key] += [val]
      else:
        data[key] = [val]

    return data

  def read_target(self, sample):
    return map(lambda x: np.array(Image.open(x).convert('P'), dtype=np.uint8), sample['targets'])

  def read_image(self, sample):
    return map(imread, sample['images'])

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    if self.mode == 'train': 
      sample = self.samples[idx]
      # random sample another one 
      ref_sampleidx = np.random.randint(0,len(self.samples),1)
      ref_sample = self.samples[ref_sampleidx]
      while sample['info']['video'] == ref_sample['info']['video'] : 
        ref_sampleidx = np.random.randint(0,len(self.samples),1)
        ref_sample = self.samples[ref_sampleidx]

      trans_tensors_resized = self.read_sample(sample)
      ref_tensors_resized = self.read_sample(ref_sample)
      # do synthesizing here 



    tensors_resized = self.read_sample(sample)

    padded_tensors = self.pad_tensors(tensors_resized)

    padded_tensors = self.normalise(padded_tensors)

    return {"images": [np.transpose(padded_tensors['img1'], (2, 0, 1)).astype(np.float32),
                       np.transpose(padded_tensors['img2'], (2, 0, 1)).astype(np.float32)],
            "target": {"flow": np.transpose(padded_tensors['flow'], (2, 0, 1)).astype(np.float32)}, 'info': {}}

  @abstractmethod
  def create_sample_list(self):
    pass


class VideoDataset(BaseDataset):
  def __init__(self, root, mode='train', resize_mode=None, resize_shape=None, tw=8, max_temporal_gap=8, num_classes=2):
    self.tw = tw
    self.max_temporal_gap = max_temporal_gap
    self.num_classes = num_classes

    self.videos = []
    self.num_frames = {}
    self.num_objects = {}
    self.shape = {}

    self.current_video = None
    self.start_index = None
    super(VideoDataset, self).__init__(root, mode, resize_mode, resize_shape)
    self.nonlinear_map = nonlinear_map()

  def set_video_id(self, video):
    self.current_video = video
    self.start_index = self.get_start_index(video)

  def get_video_ids(self):
    # shuffle the list for training
    return random.sample(self.videos, len(self.videos)) if self.is_train() else self.videos

  def get_start_index(self, video):
    start_frame = 0
    return start_frame

  def pad_tensors(self, tensors_resized):
    h, w = tensors_resized["images_synth"].shape[1:3]
    new_h = h + 32 - h % 32 if h % 32 > 0 else h
    new_w = w + 32 - w % 32 if w % 32 > 0 else w
    lh, uh = (new_h - h) / 2, (new_h - h) / 2 + (new_h - h) % 2
    lw, uw = (new_w - w) / 2, (new_w - w) / 2 + (new_w - w) % 2
    lh, uh, lw, uw = int(lh), int(uh), int(lw), int(uw)

    padded_tensors = tensors_resized.copy()
    # keys = ['images', 'targets']
    keys = ['images_ref','images_trans','images_synth']

    for key in keys:
      pt = []
      t = tensors_resized[key]
      if t.ndim == 3:
        t = t[..., None]
      assert t.ndim == 4
      padded_tensors[key] = np.pad(t,
                   ((0,0),(lh, uh), (lw, uw), (0, 0)),
                   mode='constant')
    if self.mode == 'test':
      padded_tensors['info'][0]['pad'] = ((lh, uh), (lw, uw))
    return padded_tensors

  def __getitem__(self, idx):
    if self.mode == 'train': 
      sample = self.samples[idx]
      # random sample another one 
      np.random.seed()
      ref_sampleidx = np.random.randint(0,len(self.samples),1)
      ref_sample = self.samples[ref_sampleidx[0]]
      while sample['info']['video'] == ref_sample['info']['video'] : 
        ref_sampleidx = np.random.randint(0,len(self.samples),1)
        ref_sample = self.samples[ref_sampleidx[0]]

      trans_tensors_resized = self.read_sample(sample)
      ref_tensors_resized = self.read_sample(ref_sample)

      alpha = np.random.uniform(0.4,0.9)
      beta = np.random.uniform(0,5)
      synth_images, ref_images = synthesize_video1(trans_tensors_resized['images'], ref_tensors_resized['images'], self.nonlinear_map, alpha, beta)

      ref_images = np.stack(ref_images)
      synth_images = np.stack(synth_images)
      
      tensors_resized = {}
      tensors_resized['images_ref'] = ref_images 
      tensors_resized['images_trans'] = trans_tensors_resized['images']
      tensors_resized['images_synth'] = synth_images
      tensors_resized['transinfo'] = trans_tensors_resized['info']
      tensors_resized['refinfo'] = ref_tensors_resized['info']

      # for i, val in enumerate(ref_images):
      #   cv2.imwrite('./tmp/{}_{}ref.jpg'.format(str(idx),str(i)),cv2.cvtColor(ref_images[i],cv2.COLOR_RGB2BGR))
      #   cv2.imwrite('./tmp/{}_{}synth.jpg'.format(str(idx),str(i)),cv2.cvtColor(synth_images[i],cv2.COLOR_RGB2BGR))
      #   cv2.imwrite('./tmp/{}_{}trans.jpg'.format(str(idx),str(i)),cv2.cvtColor(trans_tensors_resized['images'][i],cv2.COLOR_RGB2BGR))

      padded_tensors = self.pad_tensors(tensors_resized)
      padded_tensors = self.normalise(padded_tensors)

      # print(trans_tensors_resized['images'].shape)
      # print(ref_tensors_resized['images'].shape)
      # do synthesizing here 


    elif self.mode == 'test': 
      sample = self.samples[idx]
      tensors_resized = self.read_sample(sample)
      padded_tensors = self.pad_tensors(tensors_resized)
      padded_tensors = self.normalise(padded_tensors)

    if self.mode == 'train': 
      return {"images_ref": np.transpose(padded_tensors['images_ref'], (3, 0, 1, 2)).astype(np.float32),
              "images_trans": np.transpose(padded_tensors['images_trans'], (3, 0, 1, 2)).astype(np.float32),
              "images_synth": np.transpose(padded_tensors['images_synth'], (3, 0, 1, 2)).astype(np.float32),
              'transinfo': padded_tensors['transinfo'],
              'refinfo' : padded_tensors['refinfo']}
    else: 
      return {"images_ref": np.transpose(padded_tensors['images_ref'], (3, 0, 1, 2)).astype(np.float32),
              "images_trans": np.transpose(padded_tensors['images_trans'], (3, 0, 1, 2)).astype(np.float32),
              "images_synth": np.transpose(padded_tensors['images_synth'], (3, 0, 1, 2)).astype(np.float32),
              'info': padded_tensors['info']}


  @abstractmethod
  def get_support_indices(self, index, sequence):
    pass