import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CIFAKECNN(nn.Module):
    def __init__(self, labelnum=1):
        super(CIFAKECNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0)
        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, labelnum)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.sigmoid(self.fc2(x))
        return x


def build_resnet():
    resnet_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in resnet_model.parameters():
        param.requires_grad = False  # freeze all layers
    resnet_model.fc = nn.Sequential(
        nn.Linear(resnet_model.fc.in_features, 1),
        nn.Sigmoid()
    )
    return resnet_model
