import torch
import numpy as np
from utils import stats, shift, img

def smoothing_loss(tensor, sigma=0.0001):
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

    print(sum_log)

    return 1 / ((sum_log * (2 * sigma)))  

def deep_correlation_loss(target, sample_features):

    loss = 0

    for index in range(len(sample_features)):
        sample_matrix = stats.deep_correlation_matrix(sample_features[index])
        loss += torch.sum((target[index] - sample_matrix) ** 2 )

    return loss
    

    
