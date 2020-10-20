import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import shutil
import numpy as np
import scipy.io as sio

sys.path.append('..');

from utils.model.resnet import resnet18, resnet18_part;
import time

class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)

class Generator(nn.Module):
    def __init__(self, conv_dim=64, repeat_num=2, img_mode = 3, up_time = 3):
        super(Generator, self).__init__()

        curr_dim = conv_dim;

        # Bottleneck
        layers = []
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for i in range(up_time):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=3, stride=2, padding=1, output_padding = 1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        self.main = nn.Sequential(*layers)

        layers = []
        if img_mode == 3:
            layers.append(nn.Conv2d(curr_dim, 6, kernel_size=7, stride=1, padding=3, bias=False))
        elif img_mode == 1:
            layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        elif img_mode == 4:
            layers.append(nn.Conv2d(curr_dim, 9, kernel_size=7, stride=1, padding=3, bias=False))
        elif img_mode == 0:
            layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))

        layers.append(nn.Tanh())
        self.img_reg = nn.Sequential(*layers)

    def forward(self, x):
        features = self.main(x)
        x = self.img_reg(features);

        return x

class HR_estimator_multi_task_STmap(nn.Module):
    def __init__(self, video_length = 300):
        super(HR_estimator_multi_task_STmap, self).__init__()

        self.extractor = resnet18(pretrained=False, num_classes=1, num_output=34);
        self.extractor.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.extractor.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.feature_pool = nn.AdaptiveAvgPool2d((1, 10));
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=[1, 3], stride=[1, 3],
                               padding=[0, 0]),  # [1, 128, 32]
            nn.BatchNorm2d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=[1, 5], stride=[1, 5],
                               padding=[0, 0]),  # [1, 128, 32]
            nn.BatchNorm2d(32),
            nn.ELU(),
        )

        self.video_length = video_length;
        self.poolspa = nn.AdaptiveAvgPool2d((1, int(self.video_length)))
        self.ecg_conv = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        hr, feat_out, feat = self.extractor(x);

        x = self.feature_pool(feat);
        x = self.upsample1(x);
        x = self.upsample2(x);
        x = self.poolspa(x);
        x = self.ecg_conv(x)

        ecg = x.view(-1, int(self.video_length));

        return hr, ecg, feat_out;

class HR_disentangle(nn.Module):
    def __init__(self, video_length = 300, decov_num = 1):
        super(HR_disentangle, self).__init__()

        self.extractor = HR_estimator_multi_task_STmap();
        self.Noise_encoder = resnet18_part()
        self.Noise_encoder.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.decoder = Generator(conv_dim=128, repeat_num=decov_num, img_mode = 3)

        self.video_length = video_length;
        self.poolspa = nn.AdaptiveAvgPool2d((1, int(self.video_length/2)))
        self.ecg_conv = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, img):

        hr, ecg, feat_hr = self.extractor(img);
        feat_n = self.Noise_encoder(img);
        feat = feat_hr + feat_n;
        img = self.decoder(feat);

        return feat_hr, feat_n, hr, img, ecg

class HR_disentangle_cross(nn.Module):
    def __init__(self, video_length = 300):
        super(HR_disentangle_cross, self).__init__()

        self.encoder_decoder = HR_disentangle(decov_num = 1);

        self.video_length = video_length;
        self.poolspa = nn.AdaptiveAvgPool2d((1, int(self.video_length)))
        self.ecg_conv = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, img):

        batch_size = img.size(0);

        feat_hr, feat_n, hr, img_out, ecg = self.encoder_decoder(img);

        idx1 = torch.randint(batch_size, (batch_size,))
        idx2 = torch.randint(batch_size, (batch_size,))

        idx1 = idx1.long();
        idx2 = idx2.long();

        feat_hr1 = feat_hr[idx1, :, :, :];
        feat_hr2 = feat_hr[idx2, :, :, :];
        feat_n1 = feat_n[idx1, :, :, :];
        feat_n2 = feat_n[idx2, :, :, :];

        featf1 = feat_hr1 + feat_n2;
        featf2 = feat_hr2 + feat_n1;

        imgf1 = self.encoder_decoder.decoder(featf1);
        imgf2 = self.encoder_decoder.decoder(featf2);

        feat_hrf1, feat_nf2, hrf1, img_outf1, ecg1 = self.encoder_decoder(imgf1);
        feat_hrf2, feat_nf1, hrf2, img_outf2, ecg2 = self.encoder_decoder(imgf2);

        return feat_hr, feat_n, hr, img_out, feat_hrf1, feat_nf1, hrf1, idx1, feat_hrf2, feat_nf2, hrf2, idx2, ecg, ecg1, ecg2
