import itertools
import torch
import numpy as np

class Shifter():
    def __init__(self, dimensions_expected: int=4, axes: tuple=(3, 2)):
        self.__dimensions_expected = dimensions_expected
        self.__axes = axes
        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __validate(self):
        """
        Raises errors unless the passed in tensor and distance tuple are
        correctly formatted.
        """
        if self.__tensor.dim() != self.__dimensions_expected:
            raise ValueError("tensor has the wrong number of dimensions, expecting:  {}".format(self.__dimensions_expected))

        if self.__tensor.size(3) <= abs(self.__distance[0]) or self.__tensor.size(2) <= abs(self.__distance[1]):
            raise ValueError("Distance is invalid: value too great")
        

    def __thinned(self, tensor):
        """
        Accepts a tensor with 4 dimensions. 
        Returns a clone of that tensor with edge values replaced with 0's. 
        x and y dictate how many edge values along each axis to replace. 
        axis_offset is used for tensors with > 2 dimensions.
        """
        tensor_clone = tensor.clone()
        x, y = self.__distance

        if y >= 0:
            tensor_clone[:, :, :y] = 0
        else:
            tensor_clone[:, :, y:] = 0
        if x >= 0:
            tensor_clone[:, :, :, :x] = 0
        else:
            tensor_clone[:, :, :, x:] = 0

        return tensor_clone

    def __shifted(self):
        """
        Accepts a tensor. Returns a clone of that tensor shifted
        by the given distance along the given axis.
        """
        ndarray = self.__tensor.detach().cpu().numpy()

        return torch.from_numpy(np.roll(ndarray, self.__distance, self.__axes)).to(self.__device)
    
    def cross_displacements(self, tensor):
        """
        Returns all displacements of distance 1 along either the x 
        or the y axes, but never both at once.
        """
        displacements = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        return [self.displaced(tensor, displacement) for displacement in displacements]


    def displaced(self, tensor, distance: tuple):
        """
        Returns a clone of the current tensor shifted along the x and y
        axies according to distance (x, y). Values shifted off the edge do
        not wrap. All incoming values are zeros.
        """

        self.__distance = distance
        self.__tensor = tensor

        return self.__thinned(self.__shifted())

def generate(iterable):
    for element in iterable:
        yield element

class ShiftItr():
    """
    Takes a tensor and maximum shifts for each axes. Iterates over every
    possible shift of that tensor within those bounds.
    NOTE: we iterate to conserve memory.
    """
    def __init__(self, tensor, x_max: int, y_max: int):
        self.__shifter = Shifter()
        self.__tensor = tensor

        x_displacements = list(range(-x_max, x_max + 1))
        y_displacements = list(range(-y_max, y_max + 1))

        self.__displacements = generate(itertools.product(x_displacements, y_displacements))

    def __iter__(self):
        return self
    
    def __next__(self):
        displacement = next(self.__displacements, None)

        if displacement != None:
            return (self.__shifter.displaced(self.__tensor, displacement), displacement)
        else:
            raise StopIteration()