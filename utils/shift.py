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
        Returns a clone of that tensor with edge values raplced with 0's. 
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
        Accepts a tensor. Returns a clonde of that tensor shifted
        by the given distance along the given axis.
        """

        ndarray = self.__tensor.detach().cpu().numpy()

        return torch.from_numpy(np.roll(ndarray, self.__distance, self.__axes)).to(self.__device)


    def displaced(self, tensor, distance: tuple):
        """
        Returns a clone of the current tensor shifted along the x and y
        axies according to distance (x, y). Values shifted off the edge do
        not wrap. All incoming values are zeros.
        """

        if torch.cuda.is_available(): # must convert to cpu if currently on gpu
            tensor = tensor.cpu()

        self.__distance = distance
        self.__tensor = tensor
        self.__validate()

        return self.__thinned(self.__shifted())
