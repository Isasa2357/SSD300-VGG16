
from typing import Tuple, List

import numpy as np
from numpy import ndarray

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import box_iou

from SSD300VGG16.model import SSD300_VGG16
from SSD300VGG16.Decoder import Decoder
from SSD300VGG16.DBox import DBox


class Detector:
    '''
    物体検知を行う
    '''

    def __init__(self, num_classes: int):

        self._num_classes = num_classes
        self._predictor_locs_confs = SSD300_VGG16(self._num_classes)
        self._dbox = DBox()
        self._decoder = Decoder()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        locs, confs = self._predictor_locs_confs(x)

        # bbox位置の推論
        bbox_nxywh = self._decoder.forward(self._dbox.nxywhs, locs)

        # bboxの信頼度の推論
        confScores = F.softmax(confs, dim=2)

        return bbox_nxywh, confScores
    
    def detect(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        物体検知を行う

        Args:
            imgs: 物体検知を行う画像．画像サイズは300x300．
        
        Ret:
            Tuple[nxywhs, classes, confs]
            nxywhs: [N, num_bbox, 4(nx, ny, nw, nh)]
            classes: [N, num_bbox]
            confs: [N, num_bbox]
        '''
        # 推論を行う
        bbox_nxywhs, confScores = self.forward(imgs)

        # bboxをNMS処理により選択する

def train_detector(detector: Detector, imgs: ndarray, gt_nxyxys: torch.Tensor):
    '''
    Detectorの学習を行う
    '''
    # 推論
    bbox_nxywhs, confScores = detector.forward(torch.from_numpy(imgs))
    x2 = bbox_nxywhs[:, :, 0] + bbox_nxywhs[:, :, 2]
    y2 = bbox_nxywhs[:, :, 1] + bbox_nxywhs[:, :, 3]
    bbox_nxyxys = torch.stack([bbox_nxywhs[0], bbox_nxywhs[1], x2, y2], dim=2)

    # bboxとGTのIOUを計算
    ious = calc_IOUs(bbox_nxyxys, gt_nxyxys)


