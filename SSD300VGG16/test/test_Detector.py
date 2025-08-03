
import torch
import numpy as np

from SSD300VGG16.Detector import *

############################## bboxをgtに割り当てる処理 ##############################

def test_detectorDryrun():
    detector = Detector(21)

    imgs = torch.tensor(np.random.random((4, 3, 300, 300)), dtype=torch.float)

    bbox_nxywh, confScore = detector.forward(imgs)

    print(f'bbox nxywh shape: {bbox_nxywh.shape}')
    print(f'conf score shape: {confScore.shape}')

def test_Detector():
    detector = Detector(3)

    imgs = torch.tensor(np.random.random((4, 3, 300, 300)), dtype=torch.float)

    bbox_nxywh, confScore = detector.forward(imgs)

    print(bbox_nxywh)
    print(confScore)
