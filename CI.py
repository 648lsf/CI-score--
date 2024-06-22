import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform

def dice_coefficient(pred, target):
    smooth = 1.0
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def combined_sensitivity(real_masks, pred_masks):
    real_union = torch.max(real_masks, dim=0)[0]
    pred_union = torch.max(pred_masks, dim=0)[0]
    
    if real_union.sum() == 0 and pred_union.sum() == 0:
        return 1.0
    
    tp = (real_union * pred_union).sum().item()
    fn = (real_union * (1 - pred_union)).sum().item()
    
    return tp / (tp + fn) if (tp + fn) > 0 else 1.0

def maximum_dice_matching(real_masks, pred_masks):
    M = real_masks.size(0)
    N = pred_masks.size(0)
    dice_scores = torch.zeros((M, N))

    for i in range(M):
        for j in range(N):
            dice_scores[i, j] = dice_coefficient(real_masks[i], pred_masks[j])

    max_dice_per_real = dice_scores.max(dim=1)[0]
    max_dice_per_real[max_dice_per_real == 0] = 1.0
    
    return max_dice_per_real.mean().item()

def variance_distribution(masks):
    pairwise_dists = pdist(masks.view(masks.size(0), -1).cpu().numpy(), 'euclidean')
    return pairwise_dists.var()

def diversity_agreement(real_masks, pred_masks):
    V_Y_min = variance_distribution(real_masks)
    V_Y_max = variance_distribution(real_masks)
    
    V_Y_hat_min = variance_distribution(pred_masks)
    V_Y_hat_max = variance_distribution(pred_masks)
    
    delta_Vmin = abs(V_Y_min - V_Y_hat_min)
    delta_Vmax = abs(V_Y_max - V_Y_hat_max)
    
    return 1 - (delta_Vmin + delta_Vmax) / 2

def harmonic_mean(x, y, z):
    return 3 / (1/x + 1/y + 1/z)

def evaluate_metrics(real_masks, pred_masks):
    combined_sens = combined_sensitivity(real_masks, pred_masks)
    max_dice_match = maximum_dice_matching(real_masks, pred_masks)
    diversity_agree = diversity_agreement(real_masks, pred_masks)
    
    CI = harmonic_mean(combined_sens, max_dice_match, diversity_agree)
    
    return {
        'Combined Sensitivity': combined_sens,
        'Maximum Dice Matching': max_dice_match,
        'Diversity Agreement': diversity_agree,
        'CI': CI
    }

# 示例用法
M = 5  # 真实掩码数量
N = 7  # 预测掩码数量
H, W = 128, 128  # 掩码尺寸

# 生成随机的真实和预测掩码（仅作为示例，实际应用中使用真实数据）
real_masks = torch.randint(0, 2, (M, 1, H, W)).float()
pred_masks = torch.randint(0, 2, (N, 1, H, W)).float()

metrics = evaluate_metrics(real_masks, pred_masks)
print(metrics)
