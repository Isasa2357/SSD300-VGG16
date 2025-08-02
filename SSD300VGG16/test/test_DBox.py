
import os
import numpy as np
import cv2

import torch

from SSD300VGG16.DBox import DBox
from SSD300VGG16.sources_config import *

def test_centers():

    def draw_centers(source: SourceConfig):
        img = cv2.imread('SSD300VGG16\\test\\300x300.jpg')

        centers = DBox.calc_normalizedCenters(source)

        denormalized_centers = centers * (img.shape[0], img.shape[1])
        denormalized_centers = denormalized_centers.astype(np.int32)

        for center in denormalized_centers:
            cv2.circle(img, center=center, radius=1, color=(0, 0, 0), thickness=-1)
        
        cv2.imshow('300x300', img)
        cv2.waitKey(0)

    draw_centers(source1_config)
    draw_centers(source2_config)
    draw_centers(source3_config)
    draw_centers(source4_config)
    draw_centers(source5_config)
    draw_centers(source6_config)

def test_xywhs():

    scale_max = 0.9
    scale_min = 0.2
    scales = [0.0] * 7

    for i in range(len(scales)):
        scales[i] = scale_min + (scale_max - scale_min) * i / (len(scales)-1)
    scales[-1] = 1.0

    def draw_dbox(source: SourceConfig, scale: float, next_scale):
        img = cv2.imread('SSD300VGG16\\test\\300x300.jpg')
        
        nxywhs = DBox.calc_normalized_xywh(source, scale, next_scale)
        
        padded_img = np.array([[[255, 255, 255] for _ in range(700)] for _ in range(700)], dtype=np.uint8)
        padded_img[200:500, 200:500, :] = img

        padded_nxywhs = nxywhs * [img.shape[1], img.shape[0], img.shape[1], img.shape[0]] + [200, 200, 0, 0]

        for nxywh in padded_nxywhs:
            x1y1 = (int(nxywh[0]), int(nxywh[1]))
            x2y2 = (int(nxywh[0] + nxywh[2]), int(nxywh[1] + nxywh[3]))

            cv2.rectangle(padded_img, x1y1, x2y2, color=(0, 0, 0), thickness=1)

        cv2.imshow('padded img', padded_img)
        cv2.waitKey(0)
    
    draw_dbox(source1_config, scale=scales[0], next_scale=scales[1])
    draw_dbox(source2_config, scale=scales[1], next_scale=scales[2])
    draw_dbox(source3_config, scale=scales[2], next_scale=scales[3])
    draw_dbox(source4_config, scale=scales[3], next_scale=scales[4])
    draw_dbox(source5_config, scale=scales[4], next_scale=scales[5])
    draw_dbox(source6_config, scale=scales[5], next_scale=scales[6])
    

def test_DBox():
    dbox = DBox()

    print(dbox.nxywhs.shape)

    img = cv2.imread('SSD300VGG16\\test\\300x300.jpg')
    
    padded_img = np.array([[[255, 255, 255] for _ in range(700)] for _ in range(700)], dtype=np.uint8)
    padded_img[200:500, 200:500, :] = img

    nxywhs = dbox.nxywhs * torch.tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]]) + torch.tensor([200, 200, 0, 0])
    for nxywh in nxywhs:
        x1y1 = (int(nxywh[0]), int(nxywh[1]))
        x2y2 = (int(nxywh[0] + nxywh[2]), int(nxywh[1] + nxywh[3]))

        cv2.rectangle(padded_img, x1y1, x2y2, color=(0, 0, 0), thickness=1)

    cv2.imshow('padded img', padded_img)
    cv2.waitKey(0)