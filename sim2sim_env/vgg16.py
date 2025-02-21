import torch
import torch.nn as nn
from torchvision.models import vgg16

class CubeDetectorVGG16(nn.Module):
    def __init__(self, pretrained=True):
        super(CubeDetectorVGG16, self).__init__()
        self.vgg = vgg16(pretrained=pretrained)
        self.vgg.classifier[0] = nn.Linear(512 * 7 * 7, 256)
        self.vgg.classifier[3] = nn.Linear(256, 64)
        self.vgg.classifier[6] = nn.Linear(64, 7)  # Output for x, y, z, qx, qy, qz, qw

    def forward(self, x):
        x = self.vgg.features(x)
        x = torch.flatten(x, 1)
        x = self.vgg.classifier(x)
        return x