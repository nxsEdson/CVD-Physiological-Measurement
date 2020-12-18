########################################################
# This is an example of the training and test procedure
# You need to adjust the training and test dataloader based on your data
# CopyRight @ Xuesong Niu
########################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import shutil
import sys
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import scipy.io as sio
import torchvision.models as models
from torch.optim.lr_scheduler import MultiStepLR

sys.path.append('..');

from utils.database.Pixelmap import PixelMap_fold_STmap

from utils.model.model_disentangle import HR_disentangle_cross;
from utils.loss.loss_cross import Cross_loss;
from utils.loss.loss_r import Neg_Pearson;
from utils.loss.loss_SNR import SNR_loss;

batch_size_num = 2;
epoch_num = 70;
learning_rate = 0.001;
test_batch_size = 5;

toTensor = transforms.ToTensor();
resize = transforms.Resize(size = (320,320));

#######################################################
lambda_hr = 1;
lambda_img = 0.0000025;
lambda_low_rank = 10;
lambda_ecg = 0.02;
lambda_snr = 1;
lambda_cross_fhr = 0.000005;
lambda_cross_fn = 0.000005;
lambda_cross_hr = 1;

video_length = 300;
########################################################################
### This is only a simple toy example dataloader (utils/database/PixelMap.py)
### This dataloader do not include the cross-validation division and training/test division.
### You need to adjust your dataloader based on your own data.
### parameter: root_dir: location of the MSTmaps
###            VerticalFlip: random vertical flip for data augmentation
########################################################################
train_dataset = PixelMap_fold_STmap(root_dir='./MSTmaps/',
                                    Training = True, transform=transforms.Compose([resize, toTensor]), VerticalFlip = True,
                                    video_length = video_length);
train_loader = DataLoader(train_dataset, batch_size=batch_size_num,
                          shuffle=True, num_workers=4);

test_dataset = PixelMap_fold_STmap(root_dir='./MSTmaps/',
                                    Training = False, transform=transforms.Compose([resize, toTensor]), VerticalFlip = False,
                                    video_length = video_length);
test_loader = DataLoader(test_dataset, batch_size=test_batch_size,
                         shuffle=False, num_workers=4);
#########################################################################

#########################################################################
#########################################################################
net = HR_disentangle_cross();
net.cuda();
#########################################################################

lossfunc_HR = nn.L1Loss();
lossfunc_img = nn.L1Loss();
lossfunc_cross = Cross_loss(lambda_cross_fhr = lambda_cross_fhr, lambda_cross_fn = lambda_cross_fn, lambda_cross_hr = lambda_cross_hr);
lossfunc_ecg = Neg_Pearson(downsample_mode = 0);
lossfunc_SNR = SNR_loss(clip_length = video_length, loss_type = 7);

optimizer = torch.optim.Adam([{'params': net.parameters(), 'lr': 0.0005}]);

def train():
    net.train();
    train_loss = 0;

    for batch_idx, (data, bpm, fps, bvp, idx) in enumerate(train_loader):

        data = Variable(data);
        bvp = Variable(bvp);
        bpm = Variable(bpm.view(-1,1));
        fps = Variable(fps.view(-1,1));
        data, bpm = data.cuda(), bpm.cuda();
        fps = fps.cuda()
        bvp = bvp.cuda()

        print(bvp)

        feat_hr, feat_n, output, img_out, feat_hrf1, feat_nf1, hrf1, idx1, feat_hrf2, feat_nf2, hrf2, idx2, ecg, ecg1, ecg2 = net(data);

        loss_hr = lossfunc_HR(output, bpm)*lambda_hr;
        loss_img = lossfunc_img(data, img_out)*lambda_img;
        loss_ecg = lossfunc_ecg(ecg, bvp)*lambda_ecg;

        print(loss_ecg)
        loss_SNR, tmp = lossfunc_SNR(ecg, bpm, fps, pred = output, flag = None)*lambda_snr;
        loss = loss_hr + loss_ecg + loss_img + loss_SNR;

        loss_cross, loss_hr1, loss_hr2, loss_fhr1, loss_fhr2, loss_fn1, loss_fn2, loss_hr_dis1, loss_hr_dis2 = lossfunc_cross(feat_hr, feat_n, output,
                                                                                                                              feat_hrf1, feat_nf1,
                                                                                                                              hrf1, idx1,
                                                                                                                              feat_hrf2, feat_nf2,
                                                                                                                              hrf2, idx2, bpm)
        loss = loss + loss_cross;

        train_loss += loss.item();

        optimizer.zero_grad()
        loss.backward()
        optimizer.step();

        print('Train epoch: {:.0f}, it: {:.0f}, loss: {:.4f}, loss_hr: {:.4f}, loss_img: {:.4f}, loss_cross: {:.4f}, loss_snr: {:.4f}'.format(epoch, batch_idx,
                                                                                                 loss, loss_hr, loss_img, loss_cross, loss_SNR));

def test():
    net.eval()
    test_loss = 0;

    for (data, hr, fps, bvp, idx) in test_loader:

        data = Variable(data);
        hr = Variable(hr.view(-1,1));
        data, hr = data.cuda(), hr.cuda();

        feat_hr, feat_n, output, img_out, feat_hrf1, feat_nf1, hrf1, idx1, feat_hrf2, feat_nf2, hrf2, idx2, ecg, ecg1, ecg2 = net(data, epoch);
        loss = lossfunc_HR(output, hr);

        test_loss += loss.item();

begin_epoch = 1;
scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.5)
for epoch in range(begin_epoch, epoch_num + 1):
    if epoch > 20:
        train_dataset.transform = transforms.Compose([resize, toTensor]);
        train_dataset.VerticalFlip = False;

        train_loader = DataLoader(train_dataset, batch_size=batch_size_num,
                                  shuffle=True, num_workers=4);

    train();
    test();
