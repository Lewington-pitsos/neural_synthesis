import torch
import numpy as np

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

def extract_features(hooks, callback=None):
    """
    We assume an input has just been passed through the model. We return all
    the features collected by the passed in hooks, possibly processed through
    a callback function.
    """
    
    if callback == None:
        return [hook.features for hook in hooks]
    
    return [callback(hook.features) for hook in hooks]


def thinned_axis(ndarray, distance, axis):
    for _ in range(distance):
        ndarray = np.delete(ndarray, 0, axis=axis)
        ndarray = np.delete(ndarray, -1, axis=axis)

    return ndarray

def thinned(tensor, axis_offset: int=0, x: int=1, y: int=1):
    """
    Accepts a tensor. Returns a clone of that tensor with edge
    values raplced with 0's. 
    x and y dictate how many edge values along each axis to replace. 
    axis_offset is used for tensors with > 2 dimensions.
    """
    x_axis = 1 + axis_offset
    y_axis = axis_offset
    
    if torch.cuda.is_available(): # must convert to cpu if currently on gpu
        tensor = tensor.cpu()

    ndarray = tensor.detach().numpy()
    padding_tuples = [(0, 0) for i in range(axis_offset)] + [(0, 0), (0, 0)]

    if x > 0:
        ndarray = thinned_axis(ndarray, x, x_axis)
        padding_tuples[-1] = (x, x)

    if y > 0:
        ndarray = thinned_axis(ndarray, y, y_axis)
        padding_tuples[-2] = (y, y)
    
    return torch.from_numpy(np.pad(ndarray, padding_tuples, "constant")).to(device)

def shifted(tensor, distance: int, axis: int):
    """
    Accepts a tensor. Returns a clonde of that tensor shifted
    by the given distance along the given axis.
    """
    
    if torch.cuda.is_available(): # must convert to cpu if currently on gpu
        tensor = tensor.cpu()

    ndarray = tensor.detach().cpu().numpy()
    return torch.from_numpy(np.roll(ndarray, distance, axis)).to(device)


def all_shifts(tensor, x, y, axis_offset: int=0, diagonal: bool=True) -> list:
    """
    Creates and returns a new tensor by shifting tensor in 4 or 8 
    directions. 
    x and y dictate how far to shift along those axies. 
    axis_offset is used for tensors with > 2 dimensions.
    """

    offest_tuples = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    if diagonal:
        offest_tuples += [(-1, -1), (1, -1), (-1, 1), (1, 1)]
    
    for index in range(len(offest_tuples)):
        offest_tuples[index] = (offest_tuples[index][0] * x, offest_tuples[index][1] * y)

    shifts = []
    for offset in offest_tuples: 
        thinned_tensor = thinned(tensor, axis_offset=axis_offset, x=offset[1], y=offset[0])
        shifts.append(shifted(thinned_tensor, offset, (0 + axis_offset, 1 + axis_offset)))

    return torch.stack(shifts)

def all_half_shifts(tensor, diagonal: bool=True) -> list:
    """
    Returns all possible combinations of shifts of passed in tensor
    within the passed in x and y ranges. 
    """

    shifts = []

    axis_offset = tensor.dim() - 2

    x_axis_size = tensor.size()[axis_offset + 1]
    y_axis_size = tensor.size()[axis_offset]

    for i in range(x_axis_size // 2):
        for j in range(y_axis_size // 2):
            shifts.append(all_shifts(tensor, x=i, y=j, axis_offset=axis_offset, diagonal=diagonal))
    
    return shifts