import numpy as np

def dist(x1, y1, z1, x2, y2, z2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

def dist2d(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2)

def inf_two_exp(val):
    inf = 1
    while val > inf:
        inf <<= 1
    return inf

def get_gpu_layout(dim0, dim1, dim2):
    # This function is used to determine block and thread dimensions for CUDA kernels.
    # In Python, we don't need this explicitly since we're not writing CUDA kernels.
    pass


import torch
# import cupy as cp


def generate_hypothesis_vanishing_point(direct, coords, idxs):
    # This function generates vanishing point hypotheses.
    tn, vn, _ = direct.shape
    hn, vn, _ = idxs.shape

    hypo_pts = torch.zeros((hn, vn, 3), device=direct.device)

    for hi in range(hn):
        for vi in range(vn):
            id0 = idxs[hi, vi, 0]
            id1 = idxs[hi, vi, 1]

            dx0 = direct[id0, vi, 0]
            dy0 = direct[id0, vi, 1]
            cx0 = coords[id0, 0]
            cy0 = coords[id0, 1]

            dx1 = direct[id1, vi, 0]
            dy1 = direct[id1, vi, 1]
            cx1 = coords[id1, 0]
            cy1 = coords[id1, 1]

            lx0 = dy0
            ly0 = -dx0
            lz0 = cy0 * dx0 - cx0 * dy0

            lx1 = dy1
            ly1 = -dx1
            lz1 = cy1 * dx1 - cx1 * dy1

            x = ly0 * lz1 - lz0 * ly1
            y = lz0 * lx1 - lx0 * lz1
            z = lx0 * ly1 - ly0 * lx1

            # Adjust direction if necessary
            val_x0 = dx0 * (x - z * cx0)
            val_x1 = dx1 * (x - z * cx1)
            val_y0 = dy0 * (y - z * cy0)
            val_y1 = dy1 * (y - z * cy1)

            if val_x0 < 0 and val_x1 < 0 and val_y0 < 0 and val_y1 < 0:
                z = -z
                x = -x
                y = -y

            if val_x0 * val_x1 < 0 or val_y0 * val_y1 < 0:
                x = 0.0
                y = 0.0
                z = 0.0

            hypo_pts[hi, vi, 0] = x
            hypo_pts[hi, vi, 1] = y
            hypo_pts[hi, vi, 2] = z

    return hypo_pts


def voting_for_hypothesis_vanishing_point(direct, coords, hypo_pts, inlier_thresh):
    # This function performs voting for vanishing point hypotheses.
    tn, vn, _ = direct.shape
    hn, vn, _ = hypo_pts[:, :, :2].shape

    inliers = torch.zeros((hn, vn, tn), dtype=torch.uint8, device=direct.device)

    for hi in range(hn):
        for vi in range(vn):
            for ti in range(tn):
                cx = coords[ti, 0]
                cy = coords[ti, 1]
                hx = hypo_pts[hi, vi, 0]
                hy = hypo_pts[hi, vi, 1]
                hz = hypo_pts[hi, vi, 2]
                direct_x = direct[ti, vi, 0]
                direct_y = direct[ti, vi, 1]

                diff_x = hx - cx * hz
                diff_y = hy - cy * hz

                norm1 = torch.sqrt(direct_x ** 2 + direct_y ** 2)
                norm2 = torch.sqrt(diff_x ** 2 + diff_y ** 2)

                if norm1 < 1e-6 or norm2 < 1e-6:
                    continue

                angle_dist = (direct_x * diff_x + direct_y * diff_y) / (norm1 * norm2)

                val_x = diff_x * direct_x
                val_y = diff_y * direct_y

                if val_x < 0 or val_y < 0:
                    continue

                if abs(angle_dist) > inlier_thresh:
                    inliers[hi, vi, ti] = 1

    return inliers

def generate_hypothesis(direct, coords, idxs):
    # This function generates hypotheses based on the input tensors.
    # For simplicity, let's assume we're working with PyTorch tensors.
    tn, vn, _ = direct.shape
    hn, vn, _ = idxs.shape

    hypo_pts = torch.zeros((hn, vn, 2), device=direct.device)

    for hi in range(hn):
        for vi in range(vn):
            t0 = idxs[hi, vi, 0]
            t1 = idxs[hi, vi, 1]

            nx0 = direct[t0, vi, 1]
            ny0 = -direct[t0, vi, 0]
            cx0 = coords[t0, 0]
            cy0 = coords[t0, 1]

            nx1 = direct[t1, vi, 1]
            ny1 = -direct[t1, vi, 0]
            cx1 = coords[t1, 0]
            cy1 = coords[t1, 1]

            # Compute intersection
            if abs(nx1 * ny0 - nx0 * ny1) < 1e-6:
                continue

            y = (nx1 * (nx0 * cx0 + ny0 * cy0) - nx0 * (nx1 * cx1 + ny1 * cy1)) / (nx1 * ny0 - nx0 * ny1)
            x = (ny1 * (nx0 * cx0 + ny0 * cy0) - ny0 * (nx1 * cx1 + ny1 * cy1)) / (ny1 * nx0 - ny0 * nx1)

            hypo_pts[hi, vi, 0] = x
            hypo_pts[hi, vi, 1] = y

    return hypo_pts


def voting_for_hypothesis(direct, coords, hypo_pts, inlier_thresh):
    # This function performs voting for the hypotheses.
    tn, vn, _ = direct.shape
    hn, vn, _ = hypo_pts.shape

    inliers = torch.zeros((hn, vn, tn), dtype=torch.uint8, device=direct.device)

    for hi in range(hn):
        for vi in range(vn):
            for ti in range(tn):
                cx = coords[ti, 0]
                cy = coords[ti, 1]
                hx = hypo_pts[hi, vi, 0]
                hy = hypo_pts[hi, vi, 1]
                nx = direct[ti, vi, 0]
                ny = direct[ti, vi, 1]

                dx = hx - cx
                dy = hy - cy

                norm1 = torch.sqrt(nx ** 2 + ny ** 2)
                norm2 = torch.sqrt(dx ** 2 + dy ** 2)

                if norm1 < 1e-6 or norm2 < 1e-6:
                    continue

                angle_dist = (dx * nx + dy * ny) / (norm1 * norm2)

                if angle_dist > inlier_thresh:
                    inliers[hi, vi, ti] = 1

    return inliers


# Example usage
if __name__ == "__main__":
    # Assuming direct, coords, and idxs are PyTorch tensors on the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    direct = torch.randn(10, 5, 2, device=device)
    coords = torch.randn(10, 2, device=device)
    idxs = torch.randint(0, 10, (5, 5, 2), device=device)

    hypo_pts = generate_hypothesis(direct, coords, idxs)
    inliers = voting_for_hypothesis(direct, coords, hypo_pts, 0.5)

    print(hypo_pts)
    print(inliers)
