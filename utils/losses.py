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

def deep_correlation_matrix(tensor):
    # Generate displacements
    shifter = shift.Shifter()
    x_dim = 3
    y_dim = 2
    
    x_size = tensor.size()[x_dim]
    y_size = tensor.size()[y_dim]
    x_max_displacement =  x_size // 2
    y_max_displacement = y_size // 2


    displacements = shifter.all_displacements(tensor, x_max_displacement, y_max_displacement)

    # Calculate the displacement matrix
    matrix = tensor.new(tensor * 0.0)

    for displacement in displacements:
        x_displacement, y_displacement = displacement[1] 
        weighting = (
            (x_size - abs(x_displacement)) * (y_size - abs(y_displacement))
        ) ** -1

        displacement_score = torch.sum(
            weighting * (tensor * displacement[0]), 
            (y_dim, x_dim)
        )
        matrix[:, :, y_displacement + y_max_displacement - 1, x_displacement + x_max_displacement - 1] = displacement_score

    return matrix
  

def deep_correlation_loss(target, sample_features):

    loss = 0

    for index in range(len(sample_features)):
        sample_matrix = deep_correlation_matrix(sample_features[index])
        loss += torch.sum((target[index] - sample_matrix) ** 2 )

    return loss
    

    
