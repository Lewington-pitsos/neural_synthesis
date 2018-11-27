import torch
import numpy as np
from utils import stats

def smoothing_loss(tensor):
    """
    Calculates smoothing factor loss according to "Deep Correlations for Texture synthesis"
    """
    # print(tensor.size()) # [M, Q]
    axis_offset = tensor.dim() - 2
    
    shifts = stats.all_shifts(tensor, axis_offset)
    # print(shifts.size()) # should be [8, M, Q]
    
    differences = tensor.unsqueeze(0) - shifts
    # print(differences.size())
    
    diff_sq = differences ** 2
    # print(diff_sq.size()) # should be [8, M, Q]
    
    diff_sq_std = diff_sq * (-tensor.std())
    # print(diff_sq_std.size()) # should be [8, M, Q]
    
    sum_exp = torch.sum(torch.exp(diff_sq_std), dim=0)   
    # print(sum_exp.size()) # should be [M, Q]
    
    sum_log = torch.sum(torch.log(sum_exp))
    # print(sum_log.size()) # should be []
    
    return 1 / (sum_log * (2 * tensor.std()))