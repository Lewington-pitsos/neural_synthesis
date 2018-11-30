import torch
from torch import nn
from torchvision import models
from utils import layers

def vgg19(normalized=True, avg_pool=True):
  """
  Loads a pre-trained vgg model, loads it to the GPU, freezes it, removes
  all fully connected layers, and optionally adds a normalization layer.,
  """
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  vgg19 = models.vgg19(pretrained=True).to(device)
  for param in vgg19.features.parameters(): # stop training the net
      param.requires_grad = False

  useful_layers = list(list(vgg19.children())[0])

  if normalized:
    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device) 
    normalization_layer = layers.Normalization(normalization_mean, normalization_std).to(device)
    useful_layers.insert(0, normalization_layer)

  if avg_pool:
    for i in range(len(useful_layers)):
        if (useful_layers[i].__class__.__name__ == "MaxPool2d"):
            useful_layers[i] = nn.AvgPool2d(2)
  
  return nn.Sequential(*useful_layers)
