import torch
import numpy as np
from utils import shift

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d) # resise F_XL into \hat F_XL
    
    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

def deep_correlation_matrix(tensor):
    # Generate an iterator over all requires shifts
    shifter = shift.Shifter()
    x_dim = 3
    y_dim = 2
    
    x_size = tensor.size()[x_dim]
    y_size = tensor.size()[y_dim]
    x_max_displacement =  x_size // 2
    y_max_displacement = y_size // 2

    nexus = shift.ShiftItr(tensor, x_max_displacement, y_max_displacement)

    # Calculate the displacement matrix
    matrix = tensor.new(tensor * 0.0)

    for displacement in nexus:
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

def extract_features(hooks, callback=None):
    """
    We assume an input has just been passed through the model. We return all
    the features collected by the passed in hooks, possibly processed through
    a callback function.
    """
    
    if callback == None:
        return [hook.features for hook in hooks]
    
    return [callback(hook.features) for hook in hooks]