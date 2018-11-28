import torch
import numpy as np
from utils import stats

def smoothing_loss(tensor, sigma=0.0001):
    """
    Calculates smoothing factor loss according to "Deep Correlations for Texture synthesis"
    """
    # print(tensor.size()) # [M, Q]
    print("tensot rum: {}".format(torch.sum(tensor)))
    axis_offset = tensor.dim() - 2
    
    shifts = stats.all_shifts(tensor, axis_offset)
    # print(shifts.size()) # should be [8, M, Q]
    
    differences = tensor.unsqueeze(0) - shifts
    # print(differences.size())
    
    diff_sq = differences ** 2
    # print(diff_sq.size()) # should be [8, M, Q]
    print(torch.sum(diff_sq))

    
    diff_sq_std = diff_sq * (-sigma)
    # print(diff_sq_std.size()) # should be [8, M, Q]
    print(torch.sum(diff_sq_std))
    
    sum_exp = torch.sum(torch.exp(diff_sq_std), dim=0)   
    # print(sum_exp.size()) # should be [M, Q]
    print(torch.sum(sum_exp))
    print(min(sum_exp))
    
    sum_log = torch.sum(torch.log(sum_exp))
    #sum_log = torch.sum(sum_exp)

    # print(sum_log.size()) # should be []

    print(sum_log)
    return (1) / ((sum_log * (2 * sigma)))
