import numpy as np
import torch
import torch.nn as nn

from lib.config import cfgs

def create_sparse_depth(depth_dense):

    label = torch.sum((depth_dense > 0.5), dim=1)

    ord_num = cfgs.num_depth_bin
    beta = cfgs.max_depth

    if cfgs.SID:
        t0 = torch.exp(np.log(beta) * label.float() / ord_num)
        t1 = torch.exp(np.log(beta) * (label.float() + 1) / ord_num)
    else:
        t0 = 1.0 + (beta - 1.0) * label.float() / ord_num
        t1 = 1.0 + (beta - 1.0) * (label.float() + 1) / ord_num
    depth = (t0 + t1) / 2 - 1

    return depth

def create_dense_depth(depth_sparse):
    N, H, W = depth_sparse.shape
    ord_num = cfgs.num_depth_bin
    beta = cfgs.max_depth
    ord_c0 = torch.ones(N, ord_num, H, W).to(depth_sparse.device)
    if cfgs.SID:
        label = ord_num * torch.log(depth_sparse) / np.log(beta)
    else:
        label = ord_num * (depth_sparse - 1.0) / (beta - 1.0)

    label = label.long()
    mask = torch.linspace(0, ord_num - 1, ord_num, requires_grad=False) \
        .view(1, ord_num, 1, 1).to(depth_sparse.device)
    mask = mask.repeat(N, 1, H, W).contiguous().long()
    mask = (mask > label)
    ord_c0[mask] = 0
    ord_c1 = 1 - ord_c0
    ord_label = torch.cat((ord_c0, ord_c1), dim=1)

    return ord_label, mask

