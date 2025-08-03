
from typing import List, Tuple

from copy import copy, deepcopy

import torch
from torchvision.ops import box_iou

############################## bboxをgtに割り当てる処理 ##############################

def calc_IOUs(nxyxys: torch.Tensor, gt_nxyxys: torch.Tensor) -> List[torch.Tensor]:
    '''
    nxyxysとgts_nxyxyのIOUを計算する

    Args:
        nxyxys: [num_boxes, 4]
        gt_nxyxys: [num_gt, 4]
    
    Return:
        ious: Tuple[num_boxes, num_gt]
    '''
    ious_lst = list()

    
    return ious_lst

def find_maxIOU(ious: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    '''
    iousから最大のgtのidxとiouxを計算する

    Ret:
        matched_gt_indices, matched_gt_ious
        matched_gt_indices: List[N, num_boxes]
        matched_gt_ious: List[N, num_boxes]
    '''

    matched_gt_indices = list()
    matched_gt_ious = list()
    for iou_table in ious:
        iou_max = torch.max(iou_table, dim=1)
        matched_gt_indices.append(iou_max.indices)
        matched_gt_ious.append(iou_max.values)
    
    return matched_gt_indices, matched_gt_ious

def allot_bbox_to_gt_onestep(bbox_nxyxys: torch.Tensor, gt_nxyxys: torch.Tensor) -> Tuple[List[bool], List[int], List[float]]:
    '''
    allot_bbox_to_gtの1バッチの処理
    '''

    device = bbox_nxyxys.device

    # iouを計算する
    iou_table = box_iou(bbox_nxyxys, gt_nxyxys)

    # 各bboxに対して最大のIOUとなるgtを割り当てる
    iou_max = torch.max(iou_table, dim=1)
    matched_gt_indices = iou_max.indices
    matched_gt_ious = iou_max.values

    # IOUが0.5以上のものを割り当て成功とする
    match_gt_bbox_success = (matched_gt_ious >= 0.5)

    # 各gtが割り当てられたbboxの数を計算する
    matched_bbox_count = torch.zeros(len(gt_nxyxys), dtype=torch.int, device=device)
    for i in range(len(matched_bbox_count)):
        matched_bbox_count[i] = (matched_gt_indices == i and match_gt_bbox_success).sum().item()
    
    for i in range(len(matched_bbox_count)):
        if matched_bbox_count[i] != 0:
            continue

def allot_bbox_to_gt(bbox_nxyxys: torch.Tensor, gt_nxyxys: torch.Tensor) -> Tuple[List[]]:
    '''
    bboxにgtに割り当てる．

    Ret:
        alloted: [N, List[num_boxes]]．Trueの時，対応する位置のbboxが割り当てられたことを示す
        matched_gt_indices: [N, List[num_boxes]]．割り当てられたgtの位置
        matched_gt_ious: [N, List[num_boxes]]．割り当てられたgtとのIOU
    '''


