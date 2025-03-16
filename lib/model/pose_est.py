import torch

from lib.model.ransac.ransac_voting_gpu import ransac_voting_layer_v3, estimate_voting_distribution_with_mean

def KeypointPoseEstimation(output):
    vertex = output['vertex'].permute(0, 2, 3, 1)
    b, h, w, vn_2 = vertex.shape
    vertex = vertex.view(b, h, w, vn_2 // 2, 2)
    mask = torch.argmax(output['seg'], 1) > 0
    # if True:
    #     mean = ransac_voting_layer_v3(mask, vertex, 512, inlier_thresh=0.99)
    #     kpt_2d, var = estimate_voting_distribution_with_mean(mask, vertex, mean)
    #     output.update({'mask': mask, 'kpt_2d': kpt_2d, 'var': var})
    # else:
    kpt_2d = ransac_voting_layer_v3(mask, vertex, 128, inlier_thresh=0.99, max_num=100)
    output['kpt_2d'] = kpt_2d
    output['mask'] = mask
    return output