import torch
import torch.nn as nn
from torchvision import models


class DogBreedRecognizer(nn.Module):
    def __init__(self, n_classes):
        super(DogBreedRecognizer, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, n_classes)

    def forward(self, x):
        return self.model(x)
