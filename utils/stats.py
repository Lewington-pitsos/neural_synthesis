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

def thinned(tensor, axis_offset: int=0):
    """
    Accepts a tensor. Returns a clone of that tensor with edge
    values raplced with 0's. Axis offset is used for tensors with
    > 2 dimensions.
    """
    x_axis = 0 + axis_offset
    y_axis = 1 + axis_offset
    
    if torch.cuda.is_available(): # must convert to cpu if currently on gpu
        tensor = tensor.cpu()

    ndarray = tensor.detach().numpy()
    ndarray = np.delete(ndarray, 0, axis=x_axis)
    ndarray = np.delete(ndarray, 0, axis=y_axis)
    ndarray = np.delete(ndarray, -1, axis=x_axis)
    ndarray = np.delete(ndarray, -1, axis=y_axis)
    
    padding_tuples = [(0, 0) for i in range(axis_offset)] + [(1, 1), (1, 1)]
    
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


def all_shifts(tensor, x: int=1, y: int=1, axis_offset: int=0, diagonal: bool=True):
    """
    Replaces the edges of the last two dimensions in tensor with 0's.
    Creates and returns a new tensor by shifting tensor in all 8 
    directions. 
    """
    thinned_tensor = thinned(tensor, axis_offset)

    offest_tuples = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    if diagonal:
        offest_tuples += [(-1, -1), (1, -1), (-1, 1), (1, 1)]
    
    for index in range(len(offest_tuples)):
        offest_tuples[index] = (offest_tuples[index][0] * x, offest_tuples[index][1] * y)

    shifts = []
    for offset in offest_tuples: 
        shifts.append(shifted(thinned_tensor, offset, (0 + axis_offset, 1 + axis_offset)))

    
    return torch.stack(shifts)
