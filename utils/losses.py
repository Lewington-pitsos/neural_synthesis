import torch
import numpy as np
from utils import stats, shift, img

def smoothing_loss(tensor, sigma: int =0.0001, scale=1):
    """
    Calculates smoothing factor loss according to "Deep Correlations for Texture synthesis
    NOTE: the larger the incoming tensor the smaller the smoothing_loss. I
    ended up always scaling it by the number of elements in tensor.
    """
    axis_offset = tensor.dim() - 2
    shifter = shift.Shifter()
    shifts = torch.stack(shifter.cross_displacements(tensor))
    
    differences = tensor.unsqueeze(0) - shifts
    diff_sq = differences ** 2
    diff_sq_std = diff_sq * -sigma
    sum_exp = torch.sum(torch.exp(diff_sq_std), dim=0)   
    sum_log = torch.sum(torch.log(sum_exp))

    return 1 / ((sum_log * (2 * sigma)))  

def deep_correlation_loss(target: list, sample_features: list, scale=1) -> int:

    loss = 0

    for index in range(len(sample_features)):
        sample_matrix = stats.deep_correlation_matrix(sample_features[index])
        loss += 0.25 * torch.sum((target[index] - sample_matrix) ** 2 )

    return loss * 1000
    
def smooth_loss(_, sample_features, scale=1):
    loss = 0
    for layer in sample_features:
        loss += losses.smoothing_loss(layer) * layer.numel()
    
    return loss 
