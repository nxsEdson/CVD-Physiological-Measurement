import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import shutil
import numpy as np
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
from PIL import Image

import torchvision.transforms.functional as transF
import random;
# from skimage import io, transform

class PixelMap_fold_STmap(Dataset):
    def __init__(self, root_dir, Training=True, transform=None, VerticalFlip = False, video_length = 300):

        self.train = Training;
        self.root_dir = root_dir;
        self.transform = transform;
        self.video_length = video_length;

        self.VerticalFlip = VerticalFlip;

    def __len__(self):

        count = 0;
        for fn in os.listdir(self.root_dir):
            count = count + 1;
        return count;

    def __getitem__(self, idx):

        dir_idx = idx + 1;
        img_name1 = str(dir_idx) + '/img_rgb.png';
        img_name2 = str(dir_idx) + '/img_yuv.png';

        img_path1 = os.path.join(self.root_dir, img_name1);
        img_path2 = os.path.join(self.root_dir, img_name2);
        feature_map1 = Image.open(img_path1).convert('RGB');
        feature_map2 = Image.open(img_path2).convert('RGB');

        if self.VerticalFlip:
            if random.random() < 0.5:
                feature_map1 = transF.vflip(feature_map1);
                feature_map2 = transF.vflip(feature_map2);

        if self.transform:
            feature_map1 = self.transform(feature_map1)
            feature_map2 = self.transform(feature_map2)

        feature_map = torch.cat((feature_map1, feature_map2), dim = 0);

        bpm_path = self.root_dir + str(dir_idx) + '/bpm.mat';
        bpm = sio.loadmat(bpm_path)['bpm'];
        bpm = bpm.astype('float32');

        fps_path = self.root_dir + str(dir_idx) + '/fps.mat';
        fps = sio.loadmat(fps_path)['fps'];
        fps = fps.astype('float32');

        bvp_path = self.root_dir + str(dir_idx) + '/bvp.mat';
        bvp = sio.loadmat(bvp_path)['bvp'];
        bvp = bvp.astype('float32');
        bvp = bvp[0];

        return (feature_map, bpm, fps, bvp, idx);