
import torch
from torch import nn
from torch.nn import functional as F

class ResNet18_Cifar(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18
        self.model = resnet18(weights=None, num_classes=10)
        self.model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()

    def forward(self, x):
        return self.model(x)

    def head_parameters(self):
        return (self.model.fc.weight, self.model.fc.bias)

class FemnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))

    def head_parameters(self):
        return (self.fc2.weight, self.fc2.bias)

def get_model(dataset):
    return ResNet18_Cifar() if dataset == 'cifar10' else FemnistCNN() if dataset == 'femnist' else None
