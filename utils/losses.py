import torch
import numpy as np
from utils import stats, shift

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

def deep_correlation_matrix(tensor):
    # Generate displacements
    shifter = shift.Shifter()
    
    x_size = tensor.size()[3]
    y_size = tensor.size()[2]
    x_max_displacement =  x_size // 2
    y_max_displacement = y_size // 2

    displacements = shifter.all_displacements(tensor, x_max_displacement, y_max_displacement)

    # Calculate total score
    score = 0
    for displacement in displacements:
        x_displacement, y_displacement = displacement[1] 
        weighting = ((x_size - abs(x_displacement)) * (y_size - abs(y_displacement))) ** -1
        score += weighting * (tensor * displacement[0])
    
    return score
  

def deep_correlation_loss(target, sample_features):

    loss = 0

    for index in range(len(sample_features)):
        sample_matrix = deep_correlation_matrix(sample_features[index])
        loss += torch.sum((target[index] - sample_matrix) ** 2 )

    return loss
    

    
