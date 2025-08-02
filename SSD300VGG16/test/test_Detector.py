
import torch
import numpy as np

from SSD300VGG16.Detector import Detector



def test_detectorDryrun():
    detector = Detector(21)

    imgs = torch.tensor(np.random.random((4, 3, 300, 300)), dtype=torch.float)

    bbox_nxywh, confScore = detector.forward(imgs)

    print(f'bbox nxywh shape: {bbox_nxywh.shape}')
    print(f'conf score shape: {confScore.shape}')