import torch
import torch.nn.functional as F


def compute_sid_thresholds(alpha, beta, K, xi=0.0):
    """
    Compute SID thresholds based on Section 3.2.

    Args:
        alpha (float): Minimum depth value (e.g., 0.0 for KITTI).
        beta (float): Maximum depth value (e.g., 80.0 for KITTI).
        K (int): Number of discrete intervals.
        xi (float): Shift parameter (default 0.0, paper uses xi=1.0 to set alpha*=1.0).

    Returns:
        torch.Tensor: Thresholds [t_0, t_1, ..., t_K] of shape [K+1].
    """
    alpha_star = alpha + xi
    beta_star = beta + xi
    log_alpha = torch.log(torch.tensor(alpha_star))
    log_beta_alpha = torch.log(torch.tensor(beta_star / (alpha_star + 1e-10)))
    thresholds = torch.exp(log_alpha + log_beta_alpha * torch.arange(K + 1) / K)
    return thresholds


def DepthEstimation2(logits, alpha, beta, K, xi=0.0):
    """
    Decode ordinal logits into continuous depth values (Section 3.3).

    Args:
        logits (torch.Tensor): Network output logits.
                              Shape: [batch_size, height, width, 2 * K].
        alpha (float): Minimum depth value.
        beta (float): Maximum depth value.
        K (int): Number of discrete intervals.
        xi (float): Shift parameter.

    Returns:
        torch.Tensor: Continuous depth map.
                      Shape: [batch_size, height, width].
    """
    batch_size, channels, height, width = logits.size()
    print(logits.size())
    assert channels == 2 * K, f"Expected 2K channels, got {channels}"

    # Step 1: Compute probabilities P_{(w,h)}^k = P(l_{(w,h)} > k)
    logits = logits.view(batch_size, height, width, K, 2)  # [B, H, W, K, 2]
    probs = F.softmax(logits, dim=-1)  # [B, H, W, K, 2]
    probs = probs[..., 1]  # P(l > k), shape: [B, H, W, K]

    # Step 2: Estimate discrete label l_hat_{(w,h)} = sum(eta(P >= 0.5))
    mask = (probs >= 0.5).float()  # [B, H, W, K], 1 where P >= 0.5, 0 otherwise
    l_hat = mask.sum(dim=-1)  # [B, H, W], sum over thresholds

    # Step 3: Decode to continuous depth d_hat_{(w,h)}
    thresholds = compute_sid_thresholds(alpha, beta, K, xi).to(logits.device)  # [K+1]

    # Ensure l_hat is within valid range for indexing
    l_hat = l_hat.long().clamp(max=K - 1)  # [B, H, W], clamp to avoid out-of-bounds

    # Get t[l_hat] and t[l_hat + 1]
    t_lower = thresholds[l_hat]  # [B, H, W]
    t_upper = thresholds[l_hat + 1]  # [B, H, W]

    # Compute depth as midpoint minus shift
    depth = (t_lower + t_upper) / 2.0 - xi  # [B, H, W]

    return depth


# Example usage
def test_depth_decoding():
    # Simulate network output
    batch_size, height, width, K = 2, 24, 32, 80  # Example from KITTI
    logits = torch.randn(batch_size, height, width, 2 * K)  # [B, H, W, 2K]

    # Parameters for KITTI (Section 4.1)
    alpha = 0.0  # Min depth
    beta = 80.0  # Max depth
    xi = 1.0  # Shift to make alpha* = 1.0

    # Decode to depth
    # depth_map = decode_depth(logits, alpha, beta, K, xi)
    # print(f"Depth map shape: {depth_map.shape}")  # Expected: [2, 24, 32]
    # print(f"Sample depths: {depth_map[0, 0, 0:5]}")  # First few values

def DepthEstimation(x):
    """
    :input x: shape = (N,C,H,W), C = 2*ord_num (2*K)
    :return: ord prob is the label probability of each label, N x OrdNum x H x W
    """
    N, C, H, W = x.size()  # (N, 2K, H, W)
    ord_num = C // 2

    label_0 = x[:, 0::2, :, :].clone().view(N, 1, ord_num, H, W)  # (N, 1, K, H, W)
    label_1 = x[:, 1::2, :, :].clone().view(N, 1, ord_num, H, W)  # (N, 1, K, H, W)

    label = torch.cat((label_0, label_1), dim=1)  # (N, 2, K, H, W)
    label = torch.clamp(label, min=1e-8, max=1e8)  # prevent nans

    label_ord = torch.nn.functional.softmax(label, dim=1)
    prob = label_ord[:, 1, :, :, :].clone()  # label_ord is the output softmax probability of this model
    return prob


if __name__ == "__main__":
    test_depth_decoding()