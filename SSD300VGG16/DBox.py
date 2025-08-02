
import torch
from torch import nn

from typing import Tuple, List

import numpy as np
from numpy import ndarray
import cv2

from SSD300VGG16.sources_config import SourceConfig, sources_config

class DBox(nn.Module):
    '''
    デフォルトボックスクラス
    '''

    def __init__(self, sources_cfg: List[SourceConfig]=sources_config, scale_min: float=0.2, scale_max: float=0.9):
        super().__init__()
        
        self._sources_config = sources_cfg
        self._scale_min = scale_min
        self._scale_max = scale_max

        nxywh_lst = list()
        scales = list()
        for i in range(len(self._sources_config)):
            scale = scale_min + (scale_max - scale_min) * i / (len(self._sources_config) - 1)
            scales.append(scale)
        scales.append(1.0)

        for i, source in enumerate(self._sources_config):
            nxywh_lst.append(DBox.calc_normalized_xywh(source, scales[i], scales[i + 1]))
        
        self._nxywhs: torch.Tensor
        self.register_buffer("_nxywhs", torch.tensor(np.concatenate(nxywh_lst, axis=0)))
    
    @property
    def nxywhs(self) -> torch.Tensor:
        return self._nxywhs
        
    
    @staticmethod
    def calc_normalizedCenters(source_config: SourceConfig) -> ndarray:
        '''
        sourceの中心位置を計算する

        Args: 
            source_config: 中心を計算するsource

        Return:
            centers[sourceのマップのサイズ]
        '''

        dy = 1 / source_config.tail_size[0]
        dx = 1 / source_config.tail_size[1]

        first_center = (dy / 2, dx / 2)

        centers = list()
        for i in range(source_config.tail_size[0]):
            for j in range(source_config.tail_size[1]):
                center = ((first_center[0] + dy * i, first_center[1] + dx * j))
                centers.append(center)
        
        return np.array(centers)
    
    @staticmethod
    def calc_normalized_xywh(source_config: SourceConfig, scale: float, next_scale: float) -> ndarray:
        centers = DBox.calc_normalizedCenters(source_config)

        xywhs = np.empty((len(centers) * source_config.dboxNum, 4), dtype=np.float32)
        xywhs_idx = 0

        for center in centers:
            # 各アスペクト比に対してアンカーボックスをnxywhで定義
            for aspect in source_config.aspects:
                sqrt_aspect = aspect ** 0.5
                w = scale * sqrt_aspect
                h = scale / sqrt_aspect
                x = center[1] - w / 2
                y = center[0] - h / 2
                # print(xywhs[xywhs_idx])
                # print([x, y, w, h])
                # print(np.array([x, y, w, h], dtype=np.float32))
                xywhs[xywhs_idx] = np.array([x, y, w, h], dtype=np.float32)
                xywhs_idx += 1

            # 中間スケールのアンカーボックスをアスペクト比1で作成
            aspect = 1
            sqrt_aspect = aspect ** 0.5
            mid_scale = (next_scale * scale) ** 0.5
            w = mid_scale * sqrt_aspect
            h = mid_scale / sqrt_aspect
            x = center[1] - w / 2
            y = center[0] - h / 2
            xywhs[xywhs_idx] = np.array([x, y, w, h], dtype=np.float32)
            xywhs_idx += 1
        
        return xywhs