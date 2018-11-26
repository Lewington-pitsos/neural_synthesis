import torch
from torch import nn

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

def clipped_model(model, index):
    """
    index is the index of the last layer we want to keep.
    Assumes that the model.children is a list.
    """
    useful_layers = list(model.children())[:index + 1]

    return nn.Sequential(*useful_layers)