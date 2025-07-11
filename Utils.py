from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def image_loader(image_path, loader, device):
    """Load an image and prepare it for processing"""
    image = Image.open(image_path)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def imshow(tensor, title=None):
    """Display a tensor as an image"""
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
