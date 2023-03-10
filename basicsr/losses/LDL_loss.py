import torch
import torch.nn.functional as F

def get_local_weights(residual, ksize):

    pad = (ksize - 1) // 2
    residual_pad = F.pad(residual, pad=[pad, pad, pad, pad], mode='reflect')

    unfolded_residual = residual_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    pixel_level_weight = torch.var(unfolded_residual, dim=(-1, -2), unbiased=True, keepdim=True).squeeze(-1).squeeze(-1)

    return pixel_level_weight

def get_refined_artifact_map(img_gt, img_output, img_ema, ksize):

    residual_ema = torch.sum(torch.abs(img_gt - img_ema), 1, keepdim=True)  # R2 = abs(HR - SR2)
    residual_SR = torch.sum(torch.abs(img_gt - img_output), 1, keepdim=True)  # R1 = abs(HR - SR1)

    # theta = (var(R)) ** (1/5)
    patch_level_weight = torch.var(residual_SR.clone(), dim=(-1, -2, -3), keepdim=True) ** (1/5)
    # M(i, j) = var(R(i - (n - 1) / 2: i + (n - 1) / 2, j - (n - 1) / 2: j + (n - 1) / 2))
    pixel_level_weight = get_local_weights(residual_SR.clone(), ksize)
    overall_weight = patch_level_weight * pixel_level_weight

    overall_weight[residual_SR < residual_ema] = 0  # 非伪影部分置零

    return overall_weight