
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from SSD300VGG16.model import SSD300_VGG16
from SSD300VGG16.Decoder import Decoder
from SSD300VGG16.DBox import DBox


class Detector(nn.Module):
    '''
    物体検知を行う
    '''

    def __init__(self, num_classes: int):
        super().__init__()

        self._num_classes = num_classes
        self._predictor_locs_confs = SSD300_VGG16(self._num_classes)
        self._dbox = DBox()
        self._decoder = Decoder()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        locs, confs = self._predictor_locs_confs(x)

        # bbox位置の推論
        bbox_nxywh = self._decoder.forward(self._dbox.nxywhs, locs)

        # bboxの信頼度の推論
        confScore = F.softmax(confs, dim=2)

        return bbox_nxywh, confScore