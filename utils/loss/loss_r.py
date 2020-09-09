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

class Neg_Pearson(nn.Module):  # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self, downsample_mode = 0):
        super(Neg_Pearson, self).__init__()
        self.downsample_mode = downsample_mode;
        return

    def forward(self, preds, labels):  # all variable operation

        loss = 0.0
        for i in range(preds.shape[0]):
            a = preds[i,:];
            b = labels[i,:];

            if self.downsample_mode == 1:
                b = b[0::2]

            sum_x = torch.sum(a)  # x
            sum_y = torch.sum(b)  # y
            sum_xy = torch.sum(torch.mul(a, b))  # xy
            sum_x2 = torch.sum(torch.mul(a, a))  # x^2
            sum_y2 = torch.sum(torch.mul(b, b))  # y^2
            N = preds.shape[1]
            pearson = (N * sum_xy - sum_x * sum_y)/(torch.sqrt((N*sum_x2-sum_x*sum_x)*(N*sum_y2-sum_y*sum_y)))

            loss += 1 - pearson

        if not preds.shape[0] == 0:
            loss = loss / preds.shape[0]

        return loss