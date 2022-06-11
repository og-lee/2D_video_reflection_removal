
import glob
import os
import random
import numpy as np
from PIL import Image
from dataset.BaseDataset import VideoDataset, INFO, IMAGES_,IMAGES_TRANS,IMAGES_REF,IMAGES_SYNTH, TARGETS,IMAGES_TRANS_HALF,IMAGES_REF_HALF,IMAGES_SYNTH_HALF
from utils.Resize import ResizeMode


class ReflectionDataset1(VideoDataset):
    def __init__(self, root, mode='train', resize_mode=None, resize_shape=None, tw=8, max_temporal_gap=8, num_classes=2,
               imset=None):
        self.imset = imset
        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        self.raw_samples = []
        super(ReflectionDataset1, self).__init__(root, mode, resize_mode, resize_shape, tw, max_temporal_gap, num_classes)

    def filter_samples(self, video):
        filtered_samples = [s for s in self.raw_samples if s[INFO]['video'] == video]
        self.samples = filtered_samples

    def set_video_id(self, video):
        self.current_video = video
        self.start_index = self.get_start_index(video)
        self.filter_samples(video)

    def get_video_ids(self):
        # shuffle the list for training
        return random.sample(self.videos, len(self.videos)) if self.is_train() else self.videos

    def get_support_indices(self, index, sequence):
        # index should be start index of the clip
        if self.is_train():
            index_range = np.arange(index, min(self.num_frames[sequence],
                                         (index + max(self.max_temporal_gap, self.tw))))
        else:
            index_range = np.arange(index,
                              min(self.num_frames[sequence], (index + self.tw)))

        support_indices = np.random.choice(index_range, min(self.tw, len(index_range)), replace=False)
        support_indices = np.sort(np.append(support_indices, np.repeat([index],
                                                                   self.tw - len(support_indices))))

        # print(support_indices)
        return support_indices

    def create_sample_list(self):
    # image_dir = os.path.join(self.root, 'JPEGImages', '480p')
    # mask_dir = os.path.join(self.root, 'Annotations_unsupervised', '480p')

        # image_dir_ref = os.path.join(self.root, 'data', self.mode,'reflect')
        # image_dir_trans = os.path.join(self.root, 'data', self.mode,'trans')
        # image_dir_synth = os.path.join(self.root, 'data', self.mode,'synthetic')

        # self.mode = 'train'

        image_dir = os.path.join(self.root,  self.mode)

        image_dir_ref = os.path.join(self.root,  self.mode,'reflect')
        image_dir_trans = os.path.join(self.root,  self.mode,'trans')
        image_dir_synth = os.path.join(self.root,  self.mode,'synthetic')

        if self.is_train():
            _imset_f = 'train.txt' 
        elif self.imset:
            _imset_f = self.imset
        else:
            _imset_f = 'val.txt'
        
        print(_imset_f)
        with open(os.path.join(self.root, "ImageSets",_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.videos += [_video]


                if self.mode == 'train': 
                    img_list = list(glob.glob(os.path.join(image_dir, _video, '*.jpg')))
                    img_list.sort()
                    num_frames = len(glob.glob(os.path.join(image_dir, _video, '*.jpg')))
                    self.num_frames[_video] = num_frames
                    img1 = np.array(Image.open(os.path.join(image_dir, _video, '0.jpg')).convert("P"))

                    for i, img in enumerate(img_list):
                        sample = {INFO: {}, IMAGES_: []}
                        support_indices = self.get_support_indices(i, _video)
                        sample[INFO]['support_indices'] = support_indices
                        images = [os.path.join(image_dir, _video, '{:d}.jpg'.format(s)) for s in np.sort(support_indices)]

                        sample[IMAGES_] = images

                        sample[INFO]['video'] = _video
                        sample[INFO]['num_frames'] = num_frames
                        sample[INFO]['shape'] = np.shape(img1)

                        self.samples+=[sample]

                elif self.mode == 'test':
                    img_list_ref = list(glob.glob(os.path.join(image_dir_ref, _video, '*.jpg')))
                    img_list_trans = list(glob.glob(os.path.join(image_dir_trans, _video, '*.jpg')))
                    img_list_synth = list(glob.glob(os.path.join(image_dir_synth, _video, '*.jpg')))
                    img_list_ref.sort()
                    img_list_trans.sort()
                    img_list_synth.sort()
                
                    num_frames = len(glob.glob(os.path.join(image_dir_ref, _video, '*.jpg')))
                    self.num_frames[_video] = num_frames
                    img1 = np.array(Image.open(os.path.join(image_dir_synth, _video, '0.jpg')).convert("P"))


                    for i, img in enumerate(img_list_synth):
                        sample = {INFO: {}, IMAGES_REF: [], IMAGES_TRANS: [], IMAGES_SYNTH: []}
                        support_indices = self.get_support_indices(i, _video)
                        sample[INFO]['support_indices'] = support_indices
                        images_ref = [os.path.join(image_dir_ref, _video, '{:d}.jpg'.format(s)) for s in np.sort(support_indices)]
                        images_trans = [os.path.join(image_dir_trans, _video, '{:d}.jpg'.format(s)) for s in np.sort(support_indices)]
                        images_synth = [os.path.join(image_dir_synth, _video, '{:d}.jpg'.format(s)) for s in np.sort(support_indices)]


                        sample[IMAGES_REF] = images_ref
                        sample[IMAGES_TRANS] = images_trans
                        sample[IMAGES_SYNTH] = images_synth

                        sample[INFO]['video'] = _video
                        sample[INFO]['num_frames'] = num_frames
                        sample[INFO]['shape'] = np.shape(img1)

                        self.samples+=[sample]

        self.raw_samples = self.samples


if __name__ == '__main__':
    # davis = ReflectionDataset(root="/globalwork/data/DAVIS-Unsupervised/DAVIS/",
    reflection = ReflectionDataset1(root="/root/workplace/middle_project/",
                  resize_shape=(480, 854), resize_mode=ResizeMode.FIXED_SIZE, mode="train", max_temporal_gap=4, tw = 4)

    # davis.set_video_id('cat-girl')
    print("Dataset size: {}".format(reflection.__len__()))

    for i, _input in enumerate(reflection):
        print(_input['info'])
        print("Image Max {}, Image Min {}".format(_input['images_ref'].max(), _input['images_ref'].min()))
        print(_input['images_synth'].shape)
        # if i == 6: 
        #     break