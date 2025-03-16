import torch
import numpy as np
import torch.nn.functional as F
import cv2
from lib.utils.depth_utils import *
from lib.config import cfgs

def seg_loss(predicts, labels):
    """
    scores: a tensor [batch_size, num_classes+1, height, width]
    labels: a tensor [batch_size, num_classes+1, height, width]
    """
    class_weights = torch.tensor(cfgs.seg_classes_weight, device=predicts.device)
    cross_entropy = F.cross_entropy(predicts, labels.long(), weight=class_weights)

    return cross_entropy


def vote_loss(vertex_pred, vertex_targets, mask):
    l1_smooth = F.smooth_l1_loss(vertex_pred * mask, vertex_targets * mask)
    return l1_smooth

def depth_loss(pred_dense_depth, gt_sparse_depth):
    valid_mask = gt_sparse_depth > 0.
    ord_label, mask = create_dense_depth(gt_sparse_depth)

    entropy = -pred_dense_depth * ord_label
    loss = torch.sum(entropy, dim=1)[valid_mask]
    return loss.mean()

def losses_compute(pred_seg, gt_seg, pred_vertex, gt_vertex, pred_dense_depth, gt_sparse_depth):
    seg_weight = cfgs.losses_weight[0]
    vote_weight = cfgs.losses_weight[1]
    depth_weight = cfgs.losses_weight[2]
    mask = gt_seg > 0.
    seg = seg_loss(pred_seg, gt_seg)
    vote = vote_loss(pred_vertex, gt_vertex, mask)
    depth = depth_loss(pred_dense_depth, gt_sparse_depth)

    total_loss = (seg_weight * seg + vote_weight * vote + depth_weight * depth)

    return seg, vote, depth, total_loss