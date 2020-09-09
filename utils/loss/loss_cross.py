import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, Function
import os
import shutil
import numpy as np
import scipy.io as sio
from scipy.stats import norm

class Cross_loss(nn.Module):
    def __init__(self, lambda_cross_fhr = 0.000005, lambda_cross_fn = 0.000005, lambda_cross_hr = 1):
        super(Cross_loss, self).__init__()

        self.lossfunc_HR = nn.L1Loss();
        self.lossfunc_feat = nn.L1Loss();

        self.lambda_fhr = lambda_cross_fhr;
        self.lambda_fn = lambda_cross_fn;
        self.lambda_hr = lambda_cross_hr;

    def forward(self, feat_hr, feat_n, hr, feat_hrf1, feat_nf1, hrf1, idx1, feat_hrf2, feat_nf2, hrf2, idx2, gt):


        loss_hr1 = self.lossfunc_HR(hrf1, gt[idx1, :]);
        loss_hr2 = self.lossfunc_HR(hrf2, gt[idx2, :]);

        loss_fhr1 = self.lossfunc_feat(feat_hrf1, feat_hr[idx1, :, :, :]);
        loss_fhr2 = self.lossfunc_feat(feat_hrf2, feat_hr[idx2, :, :, :]);

        loss_fn1 = self.lossfunc_feat(feat_nf1, feat_n[idx1, :, :, :]);
        loss_fn2 = self.lossfunc_feat(feat_nf2, feat_n[idx2, :, :, :]);

        loss_hr_dis1 = self.lossfunc_HR(hrf1, hr[idx1, :]);
        loss_hr_dis2 = self.lossfunc_HR(hrf2, hr[idx2, :]);

        loss = self.lambda_hr * (loss_hr1 + loss_hr2) / 2 + self.lambda_fhr * (loss_fn1 + loss_fn2) / 2 + self.lambda_fn * (loss_fn1 + loss_fn2) / 2;

        return loss, loss_hr1, loss_hr2, loss_fhr1, loss_fhr2, loss_fn1, loss_fn2, loss_hr_dis1, loss_hr_dis2