import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ExpansiveVisualPrompt(nn.Module):
    def __init__(self, out_size, mask, init = 'zero', normalize=None):
        super(ExpansiveVisualPrompt, self).__init__()
        assert mask.shape[0] == mask.shape[1]
        in_size = mask.shape[0]
        self.out_size = out_size
        if init == "zero":
            self.program = torch.nn.Parameter(data=torch.zeros(3, out_size, out_size)) 
        elif init == "randn":
            self.program = torch.nn.Parameter(data=torch.randn(3, out_size, out_size)) 
        else:
            raise ValueError("init method not supported")
        self.normalize = normalize

        self.l_pad = int((out_size-in_size+1)/2)
        self.r_pad = int((out_size-in_size)/2)

        mask = np.repeat(np.expand_dims(mask, 0), repeats=3, axis=0)
        mask = torch.Tensor(mask)
        self.register_buffer("mask", F.pad(mask, (self.l_pad, self.r_pad, self.l_pad, self.r_pad), value=1))

    def forward(self, x):
        x = F.pad(x, (self.l_pad, self.r_pad, self.l_pad, self.r_pad), value=0) + torch.sigmoid(self.program) * self.mask
        if self.normalize is not None:
            x = self.normalize(x)
        return x

class ExpansiveVisualPrompt_one_channel(nn.Module):
    def __init__(self, out_size, mask, init = 'zero', normalize=None):
        super(ExpansiveVisualPrompt_one_channel, self).__init__()
        assert mask.shape[0] == mask.shape[1]
        in_size = mask.shape[0]
        self.out_size = out_size
        if init == "zero":
            self.program = torch.nn.Parameter(data=torch.zeros(1, out_size, out_size)) 
        elif init == "randn":
            self.program = torch.nn.Parameter(data=torch.randn(1, out_size, out_size)) 
        else:
            raise ValueError("init method not supported")
        self.normalize = normalize

        self.l_pad = int((out_size-in_size+1)/2)
        self.r_pad = int((out_size-in_size)/2)

        mask = np.repeat(np.expand_dims(mask, 0), repeats=1, axis=0)
        mask = torch.Tensor(mask)
        self.register_buffer("mask", F.pad(mask, (self.l_pad, self.r_pad, self.l_pad, self.r_pad), value=1))

    def forward(self, x):
        x = F.pad(x, (self.l_pad, self.r_pad, self.l_pad, self.r_pad), value=0) + torch.sigmoid(self.program) * self.mask
        if self.normalize is not None:
            x = self.normalize(x)
        return x

class AdditiveVisualPrompt(nn.Module):
    def __init__(self, size, pad):
        super(AdditiveVisualPrompt, self).__init__()

        self.size = size
        self.program = torch.nn.Parameter(data=torch.zeros(3, size, size)) 

        if size > 2*pad:
            mask = torch.zeros(3, size-2*pad, size-2*pad)
            self.register_buffer("mask", F.pad(mask, [pad for _ in range(4)], value=1))
        elif size == 2*pad:
            mask = torch.ones(3, size, size)
            self.register_buffer("mask", mask)
        else:
            raise ValueError("Pad Should Not Exceed Half Of Size")

    def forward(self, x):
        x += self.program * self.mask
        return x



class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.BatchNorm1d(4096),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.BatchNorm1d(4096),
        )
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x, out_feature=False):
        x = self.features(x)
        feature = x.view(x.size(0), -1)
        x = self.fc1(feature)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
