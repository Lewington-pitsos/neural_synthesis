import numpy as np
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt


def image_loader(image_path, transform):
    """load image, returns cuda tensor"""
    image = Image.open(image_path)
    image = transform(image).float()
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()  #assumes that you're using GPU

def show_image(image_tensor):
    np_image = image_tensor.squeeze().cpu().detach().numpy()
    plt.figure()
    plt.imshow(np.transpose(np_image, (1, 2, 0)))