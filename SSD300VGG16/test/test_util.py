
import torch

from SSD300VGG16.util import *

def test_calc_IOUs():
    nxyxys = torch.tensor(
        [[[0.2, 0.5, 0.3, 0.6], 
          [0.5, 0.5, 0.8, 0.8], 
          [0.2, 0.2, 0.5, 0.6]],
         [[0.1, 0.2, 0.2, 0.3], 
          [0.7, 0.6, 0.9, 0.9], 
          [0.3, 0.2, 1.0, 1.0]]]
    )

    gt_nxyxys = torch.tensor(
        [[0.2, 0.3, 0.4, 0.4], 
         [0.4, 0.5, 0.7, 0.7], 
         [0.0, 0.0, 1.0, 1.0], 
         [0.0, 0.0, 0.2, 0.2]]
    )

    ious = calc_IOUs(nxyxys, gt_nxyxys)
    print(ious)

def test_make_allotTable():
    nxyxys = torch.tensor(
        [[[0.2, 0.5, 0.3, 0.6], 
          [0.5, 0.5, 0.8, 0.8], 
          [0.2, 0.2, 0.5, 0.6]],
         [[0.1, 0.2, 0.2, 0.3], 
          [0.7, 0.6, 0.9, 0.9], 
          [0.3, 0.2, 1.0, 1.0]]]
    )

    gt_nxyxys = torch.tensor(
        [[[0.2, 0.3, 0.4, 0.4], 
          [0.4, 0.5, 0.7, 0.7], 
          [0.0, 0.0, 1.0, 1.0], 
          [0.0, 0.0, 0.2, 0.2]], 
         [[0.2, 0.3, 0.4, 0.4], 
          [0.4, 0.5, 0.7, 0.7], 
          [0.0, 0.0, 1.0, 1.0], 
          [0.0, 0.0, 0.2, 0.2]]]
    )

    ious = calc_IOUs(nxyxys, gt_nxyxys)
    allot_indices, allot_ious = find_maxIOU(ious)

    print(f'ious\n{ious}')

    print(allot_indices)
    print(allot_ious)