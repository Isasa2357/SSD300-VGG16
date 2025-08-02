

import torch
from torch import nn

class Decoder(nn.Module):
    '''
    DBoxとSSD300+VGG16が出力するオフセットからBBoxを生成
    '''

    def __init__(self, scale_xy=0.1, scale_wh=0.2):
        super().__init__()
        self._scale_xy: torch.Tensor
        self.register_buffer('_scale_xy', torch.tensor(scale_xy))
        self._scale_wh: torch.Tensor
        self.register_buffer('_scale_wh', torch.tensor(scale_wh))
        

    def forward(self, dbox_nxywh: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """
        dbox_nxywh: (num_boxes, 4) — [cx, cy, w, h]
        offset:     (N, num_boxes, 4) — SSDの出力オフセット
        return:     (N, num_boxes, 4) — 復元されたBBoxes（cx, cy, w, h）
        """

        # DBox（静的）をバッチに合わせて拡張
        if dbox_nxywh.dim() == 2:
            dbox_nxywh = dbox_nxywh.unsqueeze(0).expand(offset.size(0), -1, -1)  # (N, num_boxes, 4)

        # cx, cy
        cx = dbox_nxywh[:, :, 0] + offset[:, :, 0] * self._scale_xy * dbox_nxywh[:, :, 2]
        cy = dbox_nxywh[:, :, 1] + offset[:, :, 1] * self._scale_xy * dbox_nxywh[:, :, 3]

        # w, h
        w = dbox_nxywh[:, :, 2] * torch.exp(offset[:, :, 2] * self._scale_wh)
        h = dbox_nxywh[:, :, 3] * torch.exp(offset[:, :, 3] * self._scale_wh)

        # 結合
        bboxes = torch.stack([cx, cy, w, h], dim=2)  # (N, num_boxes, 4)
        return bboxes