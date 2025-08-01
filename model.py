
import torch
from torch import nn
import torch.nn.functional as F

from typing import Tuple, List

import numpy as np

class L2Norm(nn.Module):
    def __init__(self, n_channels: int, scale: float = 20.0, eps: float = 1e-10):
        super().__init__()
        self.n_channels = n_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_channels) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)
        # L2ノルム: チャンネル方向 (dim=1)
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps  # (N, 1, H, W)
        x_normalized = x / norm  # ブロードキャストで正規化

        # 学習可能なスケールを掛ける（shapeを合わせる）
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) * x_normalized
        return out

class FeatureExtractor_VGG16(nn.Module):
    '''
    SSD300_VGG16用の特徴抽出VGG16
    
    input (3, 300, 300)
    '''

    def __init__(self):
        super().__init__()

        self._conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1), 
            nn.ReLU(True), 
            nn.Conv2d(64, 64, 3, stride=1, padding=1), 
            nn.ReLU(True), 
            nn.MaxPool2d(2, 2)
        )     # 300x300 -> 150x150

        self._conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1), 
            nn.ReLU(True), 
            nn.Conv2d(128, 128, 3, stride=1, padding=1), 
            nn.ReLU(True), 
            nn.MaxPool2d(2, 2)
        )  # 150x150 -> 75x75

        self._conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1), 
            nn.ReLU(True), 
            nn.Conv2d(256, 256, 3, stride=1, padding=1), 
            nn.ReLU(True), 
            nn.Conv2d(256, 256, 3, stride=1, padding=1), 
            nn.ReLU(True), 
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )  # 75x75 -> 38x38

        self._conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1), 
            nn.ReLU(True), 
            nn.Conv2d(512, 512, 3, stride=1, padding=1), 
            nn.ReLU(True), 
            nn.Conv2d(512, 512, 3, stride=1, padding=1), 
            nn.ReLU(True)
        )       # 38x38 -> 38x38, source1

        self._source1_l2norm = L2Norm(512, scale=20)

        self._pool4 = nn.MaxPool2d(2, 2)        # 38x38 -> 19x19

        self._conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1), 
            nn.ReLU(True), 
            nn.Conv2d(512, 512, 3, stride=1, padding=1), 
            nn.ReLU(True), 
            nn.Conv2d(512, 512, 3, stride=1, padding=1), 
            nn.ReLU(True), 
            nn.MaxPool2d(3, 1)
        )   # 19x19 -> 19x19

        self._conv6 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=6, dilation=6), 
            nn.ReLU(True), 
            nn.Conv2d(1024, 1024, 1, stride=1, padding=1), 
            nn.ReLU(True)
        )   # 19x19 -> 19x19, source2

        self._conv7 = nn.Sequential(
            nn.Conv2d(1024, 256, 1, stride=1, padding=0), 
            nn.ReLU(True), 
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.ReLU(True)
        )   # 19x19 -> 10x10, source3

        self._conv8 = nn.Sequential(
            nn.Conv2d(512, 128, 1, stride=1, padding=0), 
            nn.ReLU(True), 
            nn.Conv2d(128, 256, 3, stride=2, padding=1), 
            nn.ReLU(True)
        )   # 10x10 -> 5x5, source4

        self._conv9 = nn.Sequential(
            nn.Conv2d(256, 128, 1, stride=1, padding=0), 
            nn.ReLU(True), 
            nn.Conv2d(128, 256, 3, stride=2, padding=1), 
            nn.ReLU(True)
        )       # 5x5 -> 3x3, source5

        self._conv10 = nn.Sequential(
            nn.Conv2d(256, 128, 1, stride=1, padding=0), 
            nn.ReLU(True), 
            nn.Conv2d(128, 256, 3, stride=2, padding=0), 
            nn.ReLU(True)
        )       # 3x3 -> 1x1, source6

        self._source1_net = nn.Sequential(self._conv1, self._conv2, self._conv3, self._conv4)
        self._source2_net = nn.Sequential(self._pool4, self._conv5, self._conv6)
        self._source3_net = nn.Sequential(self._conv7)
        self._source4_net = nn.Sequential(self._conv8)
        self._source5_net = nn.Sequential(self._conv9)
        self._source6_net = nn.Sequential(self._conv10)
    
    @property
    def source_channels(self):
        return [512, 1024, 512, 256, 256, 256]
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        source1: torch.Tensor = self._source1_net.forward(x)
        source2: torch.Tensor = self._source2_net.forward(source1)
        source3: torch.Tensor = self._source3_net.forward(source2)
        source4: torch.Tensor = self._source4_net.forward(source3)
        source5: torch.Tensor = self._source5_net.forward(source4)
        source6: torch.Tensor = self._source6_net.forward(source5)

        source1 = self._source1_l2norm.forward(source1)

        # print(source1.shape)
        # print(source2.shape)
        # print(source3.shape)
        # print(source4.shape)
        # print(source5.shape)
        # print(source6.shape)

        return source1, source2, source3, source4, source5, source6

class LocHead(nn.Module):
    '''
    SSD300 + VGG16のlocヘッド
    '''

    def __init__(self, num_boxes: List[int], source_channels: List[int]):
        super().__init__()

        self._num_boxes = num_boxes
        self._source_channels = source_channels

        self._loc_layers = nn.ModuleList([
            nn.Conv2d(ch, k * 4, 3, stride = 1, padding=1) for ch, k in zip(self._source_channels, self._num_boxes)
        ])

    def forward(self, sources: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        locs = list()
        for x, loc_lay in zip(sources, self._loc_layers):
            loc: torch.Tensor = loc_lay.forward(x)
            loc = loc.permute(0, 2, 3, 1).contiguous()
            loc = loc.view(loc.size(0), -1, 4)
            locs.append(loc)
        return torch.cat(locs, dim=1)

class ConfHead(nn.Module):
    def __init__(self, num_classes: int, num_boxes: List[int], source_channels: List[int]):
        super().__init__()

        self._num_classes = num_classes
        self._num_boxes = num_boxes
        self._source_channels = source_channels

        self._conf_layers = nn.ModuleList(
            nn.Conv2d(ch, k * self._num_classes, 3, stride=1, padding=1) for ch, k in zip(self._source_channels, self._num_boxes)
        )
    
    def forward(self, sources: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        confs = list()
        for x, conf_lay, num_box in zip(sources, self._conf_layers, self._num_boxes):
            conf: torch.Tensor = conf_lay.forward(x)
            conf = conf.permute(0, 2, 3, 1).contiguous()
            conf = conf.view(x.size(0), -1, self._num_classes)
            confs.append(conf)
        return torch.cat(confs, dim=1)

class SSD300_VGG16(nn.Module):
    '''
    SSD300 + VGG16による物体検出モデル

    input (3, 300, 300)
    '''

    def __init__(self, num_classes: int, num_boxes: List[int]):
        super().__init__()

        self._num_classes = num_classes
        self._num_boxes = num_boxes

        self._fe = FeatureExtractor_VGG16()
        self._lhead = LocHead(self._num_boxes, self._fe.source_channels)
        self._chead = ConfHead(self._num_classes, self._num_boxes, self._fe.source_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sources = self._fe.forward(x)

        locs = self._lhead.forward(sources)
        confs = self._chead.forward(sources)

        return locs, confs